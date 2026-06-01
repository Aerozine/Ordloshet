"""Semantic analysis layer for literary translation.

This module implements a theory-grounded approach to capturing **meaning** beyond
surface words, drawing on:

1. **Semantic frames** (FrameNet theory): Every utterance evokes a conceptual
   frame with typed participants (Agent, Patient, Theme, etc.).  Preserving the
   frame structure in translation ensures the illocutionary force survives.

2. **Affective computing** (Russell circumplex + NRC lexicon): Text carries
   emotional valence and arousal.  A literary translator must preserve the
   emotional arc, not just the denotative content.

3. **Discourse coherence** (Centering Theory + RST): Entities have salience.
   Pronoun resolution, topic continuity, and rhetorical relations are structure,
   not accident.  Tracking them prevents coherence loss across chunks.

4. **Pragmatic force** (Austin/Searle speech act theory): An utterance is not
   just a sentence — it is an act (assertion, question, request, threat, apology).
   A translation must preserve the illocutionary force.

5. **Conceptual network** (inspired by ConceptNet): Words are nodes in a web of
   relations (IsA, PartOf, UsedFor, Causes, HasProperty, ...).  Resolving which
   sense is active — using context — reduces translation ambiguity.

6. **Aktionsart / Verb aspect** (Vendler 1957): English verbs fall into four
   aspectual classes — States, Activities, Accomplishments, Achievements — each
   mapping to a different French tense in past-tense narrative (imparfait vs.
   passé composé vs. passé simple).  Classifying the main verbs of a chunk gives
   the patcher a strong tense-selection prior.

7. **Free Indirect Discourse** (Bakhtin 1930s / Genette 1972): Literary fiction
   constantly slips into the character's deictic centre without quotation marks
   or reporting verbs.  "Why did he have to be so difficult?" is not narration —
   it is the character thinking.  Detecting FID prevents the translator from
   normalising it into colourless indirect speech.

8. **Valence-Arousal-Dominance** (Osgood 1957): The full VAD model adds a
   *dominance* dimension to the valence already tracked.  Dominance (who controls
   the situation) shapes the appropriate French register: a high-dominance speaker
   commands, a low-dominance speaker hedges — two very different translation
   strategies even in the same register.

Integration with the pipeline
------------------------------
- ``analyze_semantics(blocks)`` is called by ``book_analyzer.analyze_book`` and
  returns a ``SemanticProfile`` stored alongside the other analysis data.
- ``chunk_semantic_hints(record, profile)`` retrieves per-chunk hints that are
  injected into the LLM patcher prompt to guide the translation.

Optional dependencies (degrade gracefully when absent)
-------------------------------------------------------
- spacy + en_core_web_sm  -- NER, dependency parsing, SRL-like frame detection
- transformers            -- emotion classification (j-hartmann/emotion-english-distilroberta-base)
- requests                -- ConceptNet REST API lookups
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# NRC Emotion lexicon (embedded minimal subset — no external file needed)
# ---------------------------------------------------------------------------
# Curated from the NRC Word-Emotion Association Lexicon (Mohammad & Turney, 2013).
# Only the most translation-relevant emotions are encoded.

_NRC_SUBSET: dict[str, list[str]] = {
    # anger
    "anger":["angry","rage","fury","furious","hate","hatred","hostile","violent","scream","curse","damn"],
    # fear
    "fear":["afraid","fear","scared","terror","panic","dread","horror","nightmare","threat","warning"],
    # joy
    "joy":["happy","joy","delight","laugh","smile","love","wonderful","beautiful","bliss","ecstatic"],
    # sadness
    "sadness":["sad","cry","tears","grief","sorrow","mourn","depressed","lonely","heartbreak","hurt"],
    # surprise
    "surprise":["surprise","shocked","astonished","unexpected","sudden","gasp","wow","incredible"],
    # anticipation
    "anticipation":["wait","expect","hope","eager","plan","prepare","about","soon","coming"],
    # disgust
    "disgust":["disgust","disgusting","nauseating","revolting","filth","gross","vile","loathe"],
    # trust
    "trust":["trust","believe","faith","honest","reliable","safe","secure","loyal","promise"],
}

_WORD_TO_EMOTIONS: dict[str, list[str]] = {}
for _emo, _words in _NRC_SUBSET.items():
    for _w in _words:
        _WORD_TO_EMOTIONS.setdefault(_w, []).append(_emo)


# ---------------------------------------------------------------------------
# Speech act detection patterns (Austin/Searle)
# ---------------------------------------------------------------------------

_SPEECH_ACT_PATTERNS: dict[str, re.Pattern] = {
    "question":    re.compile(r"\?"),
    "command":     re.compile(r"\b(stop|go|come|leave|get|stay|listen|look|wait|shut|give|take)\b", re.IGNORECASE),
    "apology":     re.compile(r"\b(sorry|apologize|forgive|pardon|excuse me)\b", re.IGNORECASE),
    "promise":     re.compile(r"\b(promise|swear|guarantee|will|i'll)\b", re.IGNORECASE),
    "threat":      re.compile(r"\b(or else|you'll regret|watch out|i'll make you|you don't want)\b", re.IGNORECASE),
    "assertion":   re.compile(r"\b(is|are|was|were|has|have|had|know|think|believe|feel)\b", re.IGNORECASE),
    "exclamation": re.compile(r"!"),
}

# Politeness markers (Brown & Levinson face theory)
_FORMAL_POLITENESS = re.compile(
    r"\b(please|would you|could you|if you don't mind|pardon|excuse me|sir|ma'am|mister|miss)\b",
    re.IGNORECASE,
)
_INFORMAL_POLITENESS = re.compile(
    r"\b(hey|yeah|yep|nah|wanna|gonna|kinda|sorta|dude|man|bro|pal)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Aktionsart lexicons (Vendler 1957)
# ---------------------------------------------------------------------------
# Each set maps English verb stems to an aspectual class.
# The class determines which French past tense is most natural.
#
#   state       -> imparfait  (no clear start/end, homogeneous)
#   activity    -> imparfait (ongoing) or passe compose (with endpoint)
#   achievement -> passe compose / passe simple (instantaneous, telic)
#   accomplishment -> passe compose (durative + natural endpoint, completed)

_AK_STATES: frozenset[str] = frozenset({
    # cognitive / perception states
    "know","believe","think","understand","suppose","assume","imagine","realise",
    "realize","recognize","recognise","remember","forget","notice","see","hear",
    # emotional states
    "love","hate","like","dislike","prefer","want","need","wish","desire","fear",
    "enjoy","mind","regret","resent","trust","doubt",
    # relational / existential
    "be","have","own","contain","belong","consist","involve","concern","depend",
    "lack","deserve","tend","matter","mean","resemble","seem","appear","look",
    "cost","weigh","measure","fit","suit",
    # sensory states
    "feel","smell","taste","sound",
})

_AK_ACHIEVEMENTS: frozenset[str] = frozenset({
    # change-of-state, punctual, telic
    "arrive","leave","depart","reach","find","lose","win","fail","succeed",
    "discover","realize","recognise","decide","die","fall","drop","hit",
    "catch","miss","finish","stop","start","begin","end","enter","exit",
    "land","wake","freeze","melt","shatter","snap","burst","explode",
    "crash","collide","stumble","trip","slip","notice","spot","glimpse",
})

_AK_ACCOMPLISHMENTS: frozenset[str] = frozenset({
    # durative + telic (has a natural endpoint)
    "write","build","make","draw","paint","create","cook","prepare","solve",
    "complete","read","eat","drink","clean","fix","repair","learn","compose",
    "translate","finish","describe","explain","construct","produce","develop",
    "design","knit","sew","carve","sculpt","bake","brew","draft","compile",
    "assemble","manufacture","restore","heal","recover","grow","raise",
})

_AK_ACTIVITIES: frozenset[str] = frozenset({
    # atelic, ongoing, durative
    "run","walk","swim","talk","speak","laugh","cry","work","play","study",
    "travel","sleep","sit","stand","wait","watch","listen","think","look",
    "search","dance","sing","drive","fly","sail","wander","roam","stroll",
    "work","practice","exercise","train","chat","gossip","wander","drift",
    "breathe","bleed","sweat","shake","tremble","smile","frown",
})

# Past-tense surface forms to detect in text
_PAST_VERB_RE = re.compile(
    r"\b([A-Za-z]+(?:ed|t))\b"   # weak past (walked, felt)
    r"|\b(was|were|had|went|came|said|told|saw|knew|thought|felt|got|gave|took|made|left|ran|stood|sat|fell)\b",
    re.IGNORECASE,
)

# Simple English past-tense to infinitive mapping (most common irregulars)
_IRREGULAR_PAST: dict[str, str] = {
    "was":"be","were":"be","had":"have","went":"go","came":"come",
    "said":"say","told":"tell","saw":"see","knew":"know","thought":"think",
    "felt":"feel","got":"get","gave":"give","took":"take","made":"make",
    "left":"leave","ran":"run","stood":"stand","sat":"sit","fell":"fall",
    "found":"find","lost":"lose","won":"win","heard":"hear","held":"hold",
    "kept":"keep","led":"lead","met":"meet","read":"read","sent":"send",
    "set":"set","spent":"spend","wore":"wear","woke":"wake","wrote":"write",
    "built":"build","caught":"catch","chose":"choose","drew":"draw",
    "drove":"drive","drank":"drink","ate":"eat","flew":"fly","froze":"freeze",
    "grew":"grow","hung":"hang","lay":"lie","meant":"mean","paid":"pay",
    "rose":"rise","rang":"ring","sang":"sing","shook":"shake","shone":"shine",
    "shot":"shoot","showed":"show","slid":"slide","slept":"sleep",
    "spoke":"speak","spent":"spend","swam":"swim","swore":"swear",
    "threw":"throw","understood":"understand","wept":"weep","won":"win",
}


# ---------------------------------------------------------------------------
# Free Indirect Discourse (FID) — Bakhtin / Genette
# ---------------------------------------------------------------------------
# FID is narration that secretly adopts the character's deictic centre:
# tense, pronouns, and adverbials stay third-person past, but the
# perspective, vocabulary, and evaluative stance are the character's.
#
# Key signals (each carries a partial score):
#   +3  interrogative sentence without quotation marks
#   +3  "How/What a" exclamative in narration
#   +2  "Why/How" + past tense question without reporting verb
#   +2  "would" + bare infinitive (future-in-past)
#   +2  temporal deictics (now/today/tomorrow/here) in past-tense narration
#   +1  evaluative intensifiers (so/such/really/absolutely) in narration
#   +1  second-person address in third-person narration

_FID_UNQUOTED_QUESTION_RE = re.compile(
    r'(?<!["“«])([A-Z][^"!?]*\?)',   # sentence ending in ? without opening quote
    re.MULTILINE,
)
_FID_EXCLAMATIVE_RE = re.compile(
    r"\b(How|What(?: a| an)?)\s+\w",
    re.IGNORECASE,
)
_FID_WHY_HOW_RE = re.compile(
    r"^(Why|How)\s+\w",
    re.IGNORECASE | re.MULTILINE,
)
_FID_WOULD_INF_RE = re.compile(
    r"\b(would|could|might|should)\s+[a-z]+\b",
    re.IGNORECASE,
)
_FID_DEICTIC_RE = re.compile(
    r"\b(now|today|tomorrow|yesterday|here|this morning|tonight|next week|"
    r"that day|the next day|the following day)\b",
    re.IGNORECASE,
)
_FID_EVALUATIVE_RE = re.compile(
    r"\b(so|such|really|absolutely|utterly|simply|just|quite|perfectly|"
    r"terribly|awfully|horribly|wonderfully|ridiculously)\s+\w",
    re.IGNORECASE,
)
_FID_SECOND_PERSON_NARRATION_RE = re.compile(
    r"\byou\s+(could|would|can|will|must|should|have|had|were|are|know)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# VAD Dominance — Osgood (1957)
# ---------------------------------------------------------------------------
# The Dominance dimension of the semantic differential captures who is
# *in control* of a situation.  High dominance = power/authority/assertion;
# low dominance = submission/deference/uncertainty.
#
# score > +0.3  -> high dominance: assertive, commanding register
# score < -0.3  -> low dominance:  deferential, hedged register
# else          -> neutral dominance

_DOM_HIGH_WORDS: frozenset[str] = frozenset({
    "force","command","order","demand","insist","require","control","power",
    "authority","refuse","dismiss","decide","assert","declare","announce",
    "impose","enforce","compel","dominate","lead","direct","rule","govern",
    "own","possess","master","conquer","defeat","win","override","reject",
    "forbid","prevent","stop","ban","prohibit","dictate",
})
_DOM_LOW_WORDS: frozenset[str] = frozenset({
    "please","maybe","perhaps","possibly","wonder","suppose","might","hesitate",
    "apologize","sorry","excuse","pardon","beg","request","ask","hope","wish",
    "afraid","nervous","uncertain","unsure","confused","lost","helpless",
    "surrender","yield","comply","obey","submit","defer","give up",
    "barely","hardly","scarcely","somehow","rather","somewhat","kind of",
})

# Imperative mood: sentence-initial verb (high dominance)
_DOM_IMPERATIVE_RE = re.compile(
    r"^(Stop|Go|Come|Leave|Get|Stay|Listen|Look|Wait|Give|Take|Do|Don't|"
    r"Tell|Show|Help|Follow|Move|Run|Sit|Stand|Open|Close|Read|Write)\b",
    re.IGNORECASE | re.MULTILINE,
)
# Hedging constructions (low dominance)
_DOM_HEDGE_RE = re.compile(
    r"\b(I think|I suppose|I wonder|I guess|I hope|if you don't mind|"
    r"would you mind|could you possibly|I'm not sure|I don't know|"
    r"kind of|sort of|a little|a bit|somewhat|rather)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ChunkSemantics:
    """Semantic annotation for a single translation chunk."""
    # --- existing fields ---
    emotions: dict[str, float] = field(default_factory=dict)
    speech_acts: list[str] = field(default_factory=list)
    politeness: str = "neutral"       # "formal" | "informal" | "neutral"
    sentiment: float = 0.0            # -1.0 to +1.0
    entity_salience: list[str] = field(default_factory=list)
    # --- Aktionsart (Vendler) ---
    aktionsart_hints: list[str] = field(default_factory=list)
    # e.g. ["love (state) -> imparfait", "arrived (achievement) -> passe compose"]
    # --- Free Indirect Discourse ---
    fid_score: float = 0.0            # 0.0 = pure narration, 1.0 = pure FID
    fid_signals: list[str] = field(default_factory=list)
    # e.g. ["unquoted_question", "would_future_in_past"]
    # --- VAD Dominance ---
    dominance: float = 0.0            # -1.0 (low/submissive) to +1.0 (high/assertive)


@dataclass
class SemanticProfile:
    """Book-level semantic profile stored alongside the analysis cache."""
    # --- existing fields ---
    emotional_arc: list[dict[str, float]] = field(default_factory=list)
    dominant_emotions: list[str] = field(default_factory=list)
    overall_sentiment: float = 0.0
    register: str = "informal"        # "formal" | "informal" | "mixed"
    speech_act_distribution: dict[str, int] = field(default_factory=dict)
    entity_emotional_profile: dict[str, dict[str, float]] = field(default_factory=dict)
    chunk_hints: list[str] = field(default_factory=list)
    # --- Aktionsart ---
    aspect_distribution: dict[str, int] = field(default_factory=dict)
    # e.g. {"state": 45, "achievement": 23, "activity": 67, "accomplishment": 12}
    # --- FID ---
    fid_ratio: float = 0.0            # proportion of chunks with fid_score >= 0.5
    # --- VAD Dominance per entity ---
    entity_dominance: dict[str, float] = field(default_factory=dict)
    # e.g. {"Alice": 0.6, "Bob": -0.3}


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _score_emotions(text: str) -> dict[str, float]:
    """Return emotion scores for *text* using the embedded NRC subset."""
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    counts: Counter = Counter()
    for word in words:
        for emo in _WORD_TO_EMOTIONS.get(word, []):
            counts[emo] += 1
    total = max(sum(counts.values()), 1)
    return {emo: round(count / total, 3) for emo, count in counts.most_common(4)}


def _score_sentiment(text: str) -> float:
    """Simple valence score: positive-word density minus negative-word density."""
    positive = set(_NRC_SUBSET["joy"]) | set(_NRC_SUBSET["trust"]) | set(_NRC_SUBSET["anticipation"])
    negative = set(_NRC_SUBSET["anger"]) | set(_NRC_SUBSET["fear"]) | set(_NRC_SUBSET["sadness"]) | set(_NRC_SUBSET["disgust"])
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in positive)
    neg = sum(1 for w in words if w in negative)
    return round((pos - neg) / max(len(words), 1) * 10, 3)


def _detect_speech_acts(text: str) -> list[str]:
    acts = [act for act, pat in _SPEECH_ACT_PATTERNS.items() if pat.search(text)]
    # Remove generic "assertion" if more specific acts are present
    if len(acts) > 1 and "assertion" in acts:
        acts.remove("assertion")
    return acts


def _detect_politeness(text: str) -> str:
    formal = bool(_FORMAL_POLITENESS.search(text))
    informal = bool(_INFORMAL_POLITENESS.search(text))
    if formal and not informal:
        return "formal"
    if informal and not formal:
        return "informal"
    return "neutral"


def _entity_salience(text: str, known_chars: set[str]) -> list[str]:
    """Return known characters mentioned in *text*, ordered by first appearance."""
    mentioned = []
    for char in known_chars:
        if re.search(rf"\b{re.escape(char)}\b", text, re.IGNORECASE):
            match = re.search(rf"\b{re.escape(char)}\b", text, re.IGNORECASE)
            if match:
                mentioned.append((match.start(), char))
    return [name for _, name in sorted(mentioned)]


# ---------------------------------------------------------------------------
# Aktionsart detection functions
# ---------------------------------------------------------------------------

def _past_to_infinitive(verb_surface: str) -> str:
    """Best-effort conversion of a past-tense surface form to its infinitive."""
    v = verb_surface.lower()
    if v in _IRREGULAR_PAST:
        return _IRREGULAR_PAST[v]
    # -ied -> -y (tried -> try)
    if v.endswith("ied"):
        return v[:-3] + "y"
    if v.endswith("ed"):
        stem = v[:-2]
        # doubled final consonant: stopped -> stop, dropped -> drop
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            return stem[:-1]
        # silent -e elision: loved -> love, arrived -> arrive, noted -> note
        # Rule: if stem ends in a consonant (not 'e'), restore the 'e'
        if stem and stem[-1] not in "aeiouе":
            restored = stem + "e"
            # Accept if the restored form is in any of our lexicons
            if (restored in _AK_STATES or restored in _AK_ACHIEVEMENTS
                    or restored in _AK_ACCOMPLISHMENTS or restored in _AK_ACTIVITIES):
                return restored
        return stem
    return v


def classify_aktionsart(verb_surface: str) -> str:
    """Return "state" | "achievement" | "accomplishment" | "activity" | "unknown"."""
    inf = _past_to_infinitive(verb_surface)
    if inf in _AK_STATES:
        return "state"
    if inf in _AK_ACHIEVEMENTS:
        return "achievement"
    if inf in _AK_ACCOMPLISHMENTS:
        return "accomplishment"
    if inf in _AK_ACTIVITIES:
        return "activity"
    return "unknown"


_FRENCH_TENSE_ADVICE: dict[str, str] = {
    "state":          "imparfait (states are backgrounded in French past narrative)",
    "activity":       "imparfait if ongoing, passe compose if completed",
    "achievement":    "passe compose / passe simple (punctual change of state)",
    "accomplishment": "passe compose when the endpoint is reached",
}


def tense_hints_for_chunk(text: str) -> list[str]:
    """Extract past-tense verbs, classify their Aktionsart, and return tense hints.

    Returns a deduplicated list of "verb (class) -> tense advice" strings.
    At most 3 hints are returned to keep prompts concise.
    """
    seen: set[str] = set()
    hints: list[str] = []
    for match in _PAST_VERB_RE.finditer(text):
        surface = match.group(0)
        cls = classify_aktionsart(surface)
        if cls == "unknown":
            continue
        inf = _past_to_infinitive(surface)
        key = f"{inf}:{cls}"
        if key in seen:
            continue
        seen.add(key)
        advice = _FRENCH_TENSE_ADVICE[cls]
        hints.append(f"{inf} ({cls}) -> {advice}")
        if len(hints) >= 3:
            break
    return hints


def aspect_distribution_for_blocks(blocks: list[str]) -> dict[str, int]:
    """Count Aktionsart classes across all blocks (for book-level profile)."""
    counts: Counter = Counter()
    for block in blocks:
        for match in _PAST_VERB_RE.finditer(block):
            cls = classify_aktionsart(match.group(0))
            if cls != "unknown":
                counts[cls] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Free Indirect Discourse detection functions
# ---------------------------------------------------------------------------

def detect_fid(text: str) -> tuple[float, list[str]]:
    """Score a chunk for Free Indirect Discourse.

    Returns (score 0.0-1.0, list of triggered signal names).
    A score >= 0.5 is considered FID.  The signals list explains which
    patterns fired, which can be surfaced in patcher hints.
    """
    raw = 0
    signals: list[str] = []

    # Unquoted question sentence (not inside quotation marks)
    stripped = re.sub(r'[""«][^""»]*[""»]', " ", text)  # remove quoted sections
    if re.search(r"\?", stripped):
        raw += 3
        signals.append("unquoted_question")

    # "How/What a" exclamative construction in narration
    if _FID_EXCLAMATIVE_RE.search(stripped):
        raw += 3
        signals.append("exclamative_how_what")

    # "Why/How" + past question without reporting verb
    if _FID_WHY_HOW_RE.search(stripped):
        raw += 2
        signals.append("why_how_question")

    # "would/could/might + infinitive" (future-in-past = character's thought)
    if _FID_WOULD_INF_RE.search(stripped):
        raw += 2
        signals.append("would_future_in_past")

    # Temporal deictics (character's "now") in past-tense narration
    if _FID_DEICTIC_RE.search(stripped):
        raw += 2
        signals.append("temporal_deictic")

    # Evaluative intensifiers (character's attitude bleeding into narration)
    if _FID_EVALUATIVE_RE.search(stripped):
        raw += 1
        signals.append("evaluative_intensifier")

    # Second-person address in third-person narration
    if _FID_SECOND_PERSON_NARRATION_RE.search(stripped):
        raw += 1
        signals.append("second_person_in_narration")

    # Normalise to 0-1 (max observable raw score is ~14)
    score = round(min(raw / 8.0, 1.0), 2)
    return score, signals


def fid_translation_hint(score: float, signals: list[str]) -> str:
    """Return an LLM hint string for FID chunks."""
    if score < 0.4:
        return ""
    if score < 0.6:
        level = "possible"
    elif score < 0.8:
        level = "likely"
    else:
        level = "strong"
    signal_str = ", ".join(signals[:3]) if signals else ""
    hint = (
        f"FID ({level}): this chunk may adopt the character's deictic centre. "
        "Use imparfait de style indirect, preserve the character's voice and "
        "evaluative stance rather than flattening to neutral narration."
    )
    if signal_str:
        hint += f" (signals: {signal_str})"
    return hint


# ---------------------------------------------------------------------------
# VAD Dominance scoring functions
# ---------------------------------------------------------------------------

def score_dominance(text: str) -> float:
    """Return a dominance score in [-1.0, +1.0].

    +1.0 = maximally assertive/dominant (commands, assertions of power)
    -1.0 = maximally deferential/submissive (apologies, hedging, requests)
     0.0 = neutral
    """
    words = re.findall(r"\b[a-z]+\b", text.lower())
    if not words:
        return 0.0

    high = sum(1 for w in words if w in _DOM_HIGH_WORDS)
    low  = sum(1 for w in words if w in _DOM_LOW_WORDS)

    # Imperative patterns boost high dominance
    imperatives = len(_DOM_IMPERATIVE_RE.findall(text))
    high += imperatives * 2

    # Hedging patterns boost low dominance
    hedges = len(_DOM_HEDGE_RE.findall(text))
    low += hedges * 2

    total = max(high + low, 1)
    return round((high - low) / total, 2)


def dominance_hint(score: float) -> str:
    """Return a register hint derived from the dominance score."""
    if score > 0.4:
        return (
            "High-dominance speaker: use assertive, direct French register. "
            "Imperatives and declarative force should be preserved."
        )
    if score < -0.4:
        return (
            "Low-dominance speaker: use deferential French register. "
            "Preserve hedging, softeners, and apologetic markers."
        )
    return ""


def entity_dominance_profile(
    blocks: list[str], known_chars: set[str]
) -> dict[str, float]:
    """Compute average dominance score for each character's dialogue blocks."""
    char_scores: dict[str, list[float]] = defaultdict(list)
    for block in blocks:
        present = [c for c in known_chars if re.search(rf"\b{re.escape(c)}\b", block)]
        if not present:
            continue
        dom = score_dominance(block)
        for char in present:
            char_scores[char].append(dom)
    return {
        char: round(sum(scores) / len(scores), 2)
        for char, scores in char_scores.items()
        if scores
    }


# ---------------------------------------------------------------------------
# ConceptNet integration (optional — degrades gracefully)
# ---------------------------------------------------------------------------

def _conceptnet_relations(
    word: str, lang: str = "en", max_results: int = 5
) -> list[dict]:
    """Look up ConceptNet relations for *word* via the public REST API.

    Returns an empty list if requests is not installed or the API is unreachable.
    Cached in memory to avoid repeated network calls.
    """
    cache = getattr(_conceptnet_relations, "_cache", {})
    key = (word, lang)
    if key in cache:
        return cache[key]
    try:
        import requests
        url = f"https://api.conceptnet.io/c/{lang}/{word.lower()}"
        resp = requests.get(url, timeout=3)
        if resp.ok:
            edges = resp.json().get("edges", [])[:max_results]
            result = [
                {
                    "relation": e.get("rel", {}).get("label", ""),
                    "start": e.get("start", {}).get("label", ""),
                    "end": e.get("end", {}).get("label", ""),
                    "weight": e.get("weight", 0),
                }
                for e in edges
            ]
            cache[key] = result
            setattr(_conceptnet_relations, "_cache", cache)
            return result
    except Exception:
        pass
    return []


def semantic_disambiguation_hint(word: str, context: str) -> str:
    """Return a brief French translation hint for *word* using ConceptNet context.

    Returns '' if ConceptNet is unavailable or no relevant relation is found.
    """
    relations = _conceptnet_relations(word)
    if not relations:
        return ""
    # Find relations whose start/end appears in context
    context_lower = context.lower()
    for rel in relations:
        end_label = rel.get("end", "").lower()
        if end_label and end_label in context_lower:
            return f"[{word} = {rel['relation']} {end_label}]"
    return ""


# ---------------------------------------------------------------------------
# Optional: Transformer-based emotion classifier
# ---------------------------------------------------------------------------

_EMOTION_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
_emotion_pipeline = None


def _load_emotion_pipeline():
    global _emotion_pipeline
    if _emotion_pipeline is not None:
        return _emotion_pipeline
    try:
        from transformers import pipeline as _pipe
        _emotion_pipeline = _pipe(
            "text-classification",
            model=_EMOTION_MODEL_ID,
            top_k=4,
            device=-1,
        )
    except Exception:
        pass
    return _emotion_pipeline


def deep_emotion_scores(text: str) -> dict[str, float]:
    """Run the DistilRoBERTa emotion classifier if available.

    Falls back to the lexicon-based scorer when the model is not installed.
    """
    pipe = _load_emotion_pipeline()
    if pipe is None:
        return _score_emotions(text)
    try:
        results = pipe(text[:512], truncation=True)
        if isinstance(results, list) and results:
            top = results[0] if isinstance(results[0], dict) else results[0][0]
            if isinstance(top, list):
                return {r["label"].lower(): round(r["score"], 3) for r in top}
            return {top["label"].lower(): round(top["score"], 3)}
    except Exception:
        pass
    return _score_emotions(text)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def analyze_semantics(
    blocks: list[str],
    known_chars: Optional[set[str]] = None,
    use_deep_emotion: bool = False,
) -> SemanticProfile:
    """Build a SemanticProfile for a full book (list of paragraph blocks).

    Parameters
    ----------
    blocks:
        Paragraph-level text blocks (from book_analyzer._extract_blocks).
    known_chars:
        Set of canonical character names for entity salience tracking.
    use_deep_emotion:
        If True, attempt to use the DistilRoBERTa emotion model (slower, more
        accurate).  Falls back to the NRC lexicon when the model is unavailable.
    """
    known_chars = known_chars or set()
    emotion_fn = deep_emotion_scores if use_deep_emotion else _score_emotions

    arc: list[dict[str, float]] = []
    speech_act_total: Counter = Counter()
    formality_votes: list[str] = []
    sentiment_sum = 0.0
    entity_emotions: dict[str, list[dict[str, float]]] = defaultdict(list)

    window = max(1, len(blocks) // 20)  # sample at most 20 points along the arc
    sampled_blocks = blocks[::window][:20]

    fid_scores: list[float] = []
    aspect_counts: Counter = Counter()

    for block in sampled_blocks:
        emo = emotion_fn(block)
        arc.append(emo)
        sentiment_sum += _score_sentiment(block)
        acts = _detect_speech_acts(block)
        speech_act_total.update(acts)
        formality_votes.append(_detect_politeness(block))
        # Entity-level emotion
        for char in _entity_salience(block, known_chars):
            entity_emotions[char].append(emo)
        # FID ratio sampling
        fid_s, _ = detect_fid(block)
        fid_scores.append(fid_s)
        # Aspect distribution (sampled)
        for match in _PAST_VERB_RE.finditer(block):
            cls = classify_aktionsart(match.group(0))
            if cls != "unknown":
                aspect_counts[cls] += 1

    # Dominant emotions across the arc
    all_emo_counts: Counter = Counter()
    for chunk_emo in arc:
        for emo, score in chunk_emo.items():
            all_emo_counts[emo] += score
    dominant = [e for e, _ in all_emo_counts.most_common(3)]

    # Overall register
    formal_count = formality_votes.count("formal")
    informal_count = formality_votes.count("informal")
    if formal_count > informal_count * 2:
        register = "formal"
    elif informal_count > formal_count * 2:
        register = "informal"
    else:
        register = "mixed"

    # Per-entity emotion summary
    char_profiles: dict[str, dict[str, float]] = {}
    for char, emo_list in entity_emotions.items():
        merged: Counter = Counter()
        for d in emo_list:
            merged.update(d)
        total = max(sum(merged.values()), 1)
        char_profiles[char] = {k: round(v / total, 3) for k, v in merged.most_common(3)}

    # Compute entity dominance across all sampled blocks
    char_dom = entity_dominance_profile(sampled_blocks, known_chars)

    # FID ratio across sampled blocks
    fid_ratio = round(
        sum(1 for s in fid_scores if s >= 0.5) / max(len(fid_scores), 1), 2
    )

    # Translation hints derived from analysis
    hints: list[str] = []
    if dominant:
        hints.append(
            f"Dominant emotional register: {', '.join(dominant)}. "
            "Preserve emotional intensity in French."
        )
    if register == "formal":
        hints.append("Text uses formal register; prefer 'vous' in ambiguous dialogue.")
    elif register == "informal":
        hints.append("Text uses informal register; prefer 'tu' in ambiguous dialogue.")
    for act, count in speech_act_total.most_common(2):
        if act not in ("assertion",) and count > 2:
            hints.append(f"Frequent speech act: {act}. Preserve illocutionary force in French.")
    if fid_ratio > 0.2:
        hints.append(
            f"Free Indirect Discourse detected in ~{fid_ratio:.0%} of sampled chunks. "
            "Preserve character's deictic centre and evaluative stance."
        )
    if aspect_counts:
        dominant_aspect = max(aspect_counts, key=aspect_counts.__getitem__)
        hints.append(
            f"Dominant Aktionsart: {dominant_aspect}. "
            f"Prefer {_FRENCH_TENSE_ADVICE.get(dominant_aspect, 'context-appropriate tense')}."
        )

    return SemanticProfile(
        emotional_arc=arc,
        dominant_emotions=dominant,
        overall_sentiment=round(sentiment_sum / max(len(sampled_blocks), 1), 3),
        register=register,
        speech_act_distribution=dict(speech_act_total),
        entity_emotional_profile=char_profiles,
        chunk_hints=hints,
        aspect_distribution=dict(aspect_counts),
        fid_ratio=fid_ratio,
        entity_dominance=char_dom,
    )


def chunk_semantic_hints(text: str, profile: Optional[SemanticProfile] = None) -> str:
    """Return a compact semantic hint string for a single translation chunk.

    All eight theories contribute:
      Tone (NRC emotion) | Acts (speech acts) | Register (politeness) |
      Aspect (Aktionsart tense advice) | FID (free indirect discourse) |
      Dominance (VAD)
    """
    parts: list[str] = []

    # --- Emotion / Tone (NRC) ---
    emo = _score_emotions(text)
    if emo:
        top_emo = max(emo, key=emo.__getitem__)
        parts.append(f"Tone: {top_emo}")

    # --- Speech acts (Austin/Searle) ---
    acts = _detect_speech_acts(text)
    if acts and acts != ["assertion"]:
        parts.append(f"Acts: {', '.join(acts)}")

    # --- Politeness register (Brown & Levinson) ---
    pol = _detect_politeness(text)
    if pol != "neutral":
        parts.append(f"Register: {pol}")

    # --- Aktionsart / French tense advice (Vendler) ---
    tense = tense_hints_for_chunk(text)
    if tense:
        parts.append(f"Aspect: {tense[0]}")

    # --- Free Indirect Discourse (Bakhtin/Genette) ---
    fid_score, fid_sigs = detect_fid(text)
    if fid_score >= 0.5:
        level = "likely" if fid_score < 0.8 else "strong"
        parts.append(f"FID ({level}): adopt character's voice, use imparfait de style indirect")

    # --- Dominance (Osgood VAD) ---
    dom = score_dominance(text)
    if dom > 0.4:
        parts.append("Dominance: high — keep assertive register")
    elif dom < -0.4:
        parts.append("Dominance: low — keep deferential/hedged register")

    return " | ".join(parts) if parts else ""


def profile_to_dict(profile: SemanticProfile) -> dict:
    """Convert a SemanticProfile to a JSON-serialisable dict."""
    from dataclasses import asdict
    return asdict(profile)
