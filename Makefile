PYTHON      ?= python
BOOK        ?= examples/Game Changers 6 The Long Game (Rachel Reid) (Z-Library).epub
BOOK_CONFIG ?= books/game_changers.toml
CHAPTER     ?= 1
XHTML_INDEX ?=
MODEL       ?= nllb
STRATEGY    ?= light
EXTRA_ARGS  ?=
EVAL_REFERENCE ?= right_translation/GC6-fr.epub

CHAPTER_ARGS    = --chapter $(CHAPTER) $(if $(XHTML_INDEX),--xhtml-index $(XHTML_INDEX),)
BENCH_ARGS      = $(CHAPTER_ARGS) --no-print-text $(EXTRA_ARGS)
PATCH_ARGS      = --qe-engine auto --comet-model auto --batch-size 4 --batch-token-limit 1400 \
                  --chunk-preview-chars 0 $(EXTRA_ARGS)
LITERARY_ARGS   = --literary-reread dialogue --literary-window-reread dialogue \
                  --window-size 5 --window-stride 3 --max-repair-rounds 2 \
                  --context-window 3 --reread-window 3 --literary-arbitration \
                  --arbitration-model nllb-ct2 --back-translation-check changed \
                  --strict-acceptance --candidate-reranking --correction-memory \
                  --book-memory --glossary-enforcement --llm-json-output \
                  --no-print-text --eval-reference "$(EVAL_REFERENCE)" \
                  --book-config "$(BOOK_CONFIG)" $(EXTRA_ARGS)

.PHONY: all help init validate clean list-xhtml translate \
        model-test-nllb model-test-nllb-ct2 model-test-madlad model-test-madlad-ct2 \
        model-test-madlad-q4 model-test-madlad-3b model-test-madlad-spec \
        model-test-opus model-test-opus-ct2 model-test-seamless \
        model-test-sonar model-test-gemma model-test-mistral model-test-qwen model-test-tower \
        model-test-mistral-vllm model-test-qwen-vllm \
        model-test-mistral-awq model-test-qwen-awq \
        benchmark_all benchmark_draft benchmark_patch benchmark_literary \
        test test-nllb test-nllb-ct2 test-madlad test-opus test-opus-ct2 test-gemma \
        test-mistral test-fast test-fast-ct2 test-slow test-fast-patch test-fast-mistral \
        test-fast-tower test-fast-qwen test-fast-qe test-fast-ct2-patch test-fast-ct2-mistral \
        test-fast-ct2-tower test-fast-ct2-qwen test-fast-ct2-qe test-slow-patch \
        test-slow-mistral test-slow-tower test-slow-qwen test-slow-qe test-best \
        test-literary translate-best prefetch-best test-patch test-llm test-all compare \
        compare3 compare3-draft compare3-patch compare3-qwen compare3-mistral \
        benchmark-alma benchmark-alma-draft benchmark-alma-qwen benchmark-alma-qwen14b \
        compare-alma compare-alma-draft compare-alma-patch

all: help

help:
	$(PYTHON) epub.py --help

validate:
	$(PYTHON) tests/validation.py || echo "Validation completed with optional deps missing"

init:
	$(PYTHON) -m venv venv
	venv/bin/pip install -r requirements.txt

list-xhtml:
	$(PYTHON) epub.py --model nllb --strategy light --book "$(BOOK)" --list-xhtml --dry-run

clean:
	rm -rf output/

translate: validate
	$(PYTHON) epub.py --model $(MODEL) --strategy $(STRATEGY) --book "$(BOOK)" $(CHAPTER_ARGS) $(EXTRA_ARGS)

# 
# Per-model tests: translate chapter 1 quickly (no-print, no QE scoring).
# Usage: make model-test-nllb
# 

model-test-nllb:
	@echo "=== NLLB-200 3.3B ==="
	$(PYTHON) epub.py --model nllb --strategy light --book "$(BOOK)" $(BENCH_ARGS)

model-test-nllb-ct2:
	@echo "=== NLLB CTranslate2 ==="
	$(PYTHON) epub.py --model nllb-ct2 --strategy light --book "$(BOOK)" $(BENCH_ARGS)

model-test-madlad:
	@echo "=== MADLAD-400 10B ==="
	$(PYTHON) epub.py --model madlad --strategy light --book "$(BOOK)" $(BENCH_ARGS)

model-test-madlad-ct2:
	@echo "=== MADLAD CTranslate2 ==="
	$(PYTHON) epub.py --model madlad-ct2 --strategy light --book "$(BOOK)" $(BENCH_ARGS)

model-test-madlad-q4:
	@echo "=== MADLAD-400 10B (4-bit NF4) ==="
	$(PYTHON) epub.py --model madlad-q4 --strategy light --book "$(BOOK)" $(BENCH_ARGS)

model-test-madlad-3b:
	@echo "=== MADLAD-400 3B ==="
	$(PYTHON) epub.py --model madlad-3b --strategy light --book "$(BOOK)" $(BENCH_ARGS)

model-test-madlad-spec:
	@echo "=== MADLAD speculative (10B NF4 + 3B assistant) ==="
	$(PYTHON) epub.py --model madlad-spec --strategy light --book "$(BOOK)" $(BENCH_ARGS)

model-test-opus:
	@echo "=== OPUS-MT ==="
	$(PYTHON) epub.py --model opus --strategy light --book "$(BOOK)" $(BENCH_ARGS)

model-test-opus-ct2:
	@echo "=== OPUS-MT CTranslate2 ==="
	$(PYTHON) epub.py --model opus-ct2 --strategy light --book "$(BOOK)" $(BENCH_ARGS)

model-test-seamless:
	@echo "=== SeamlessM4T v2 ==="
	$(PYTHON) epub.py --model seamless --strategy light --book "$(BOOK)" $(BENCH_ARGS)

model-test-sonar:
	@echo "=== SONAR ==="
	$(PYTHON) epub.py --model sonar --strategy light --book "$(BOOK)" $(BENCH_ARGS)

model-test-gemma:
	@echo "=== Gemma Translate 12B ==="
	$(PYTHON) epub.py --model gemma --strategy heavy --book "$(BOOK)" $(BENCH_ARGS)

model-test-mistral:
	@echo "=== Mistral-7B (direct) ==="
	$(PYTHON) epub.py --model mistral --strategy heavy --book "$(BOOK)" $(BENCH_ARGS)

model-test-qwen:
	@echo "=== Qwen2.5-7B (direct) ==="
	$(PYTHON) epub.py --model qwen --strategy heavy --book "$(BOOK)" $(BENCH_ARGS)

model-test-tower:
	@echo "=== TowerInstruct-7B (direct) ==="
	$(PYTHON) epub.py --model tower --strategy heavy --book "$(BOOK)" $(BENCH_ARGS)

model-test-mistral-vllm:
	@echo "=== Mistral-7B vLLM (direct) ==="
	$(PYTHON) epub.py --model mistral-vllm --strategy heavy --book "$(BOOK)" $(BENCH_ARGS)

model-test-qwen-vllm:
	@echo "=== Qwen2.5-7B vLLM (direct) ==="
	$(PYTHON) epub.py --model qwen-vllm --strategy heavy --book "$(BOOK)" $(BENCH_ARGS)

model-test-mistral-awq:
	@echo "=== Mistral-7B AWQ (~4GB) ==="
	$(PYTHON) epub.py --model mistral-awq --strategy heavy --book "$(BOOK)" $(BENCH_ARGS)

model-test-qwen-awq:
	@echo "=== Qwen2.5-7B AWQ (~4GB) ==="
	$(PYTHON) epub.py --model qwen-awq --strategy heavy --book "$(BOOK)" $(BENCH_ARGS)

# 
# Benchmark suites — run all models in a category, continue on failure.
# 

benchmark_draft:
	@echo "=== Draft model benchmark ==="
	$(MAKE) model-test-nllb        || echo "NLLB failed; continuing"
	$(MAKE) model-test-nllb-ct2    || echo "NLLB-CT2 failed; continuing"
	$(MAKE) model-test-madlad      || echo "MADLAD failed; continuing"
	$(MAKE) model-test-madlad-ct2  || echo "MADLAD-CT2 failed; continuing"
	$(MAKE) model-test-madlad-q4   || echo "MADLAD-Q4 failed; continuing"
	$(MAKE) model-test-madlad-3b   || echo "MADLAD-3B failed; continuing"
	$(MAKE) model-test-madlad-spec || echo "MADLAD-Spec failed; continuing"
	$(MAKE) model-test-opus        || echo "OPUS failed; continuing"
	$(MAKE) model-test-opus-ct2    || echo "OPUS-CT2 failed; continuing"
	$(MAKE) model-test-seamless    || echo "Seamless failed; continuing"
	$(MAKE) model-test-sonar       || echo "SONAR failed; continuing"
	@echo "=== Draft benchmark complete ==="

benchmark_patch:
	@echo "=== Patch pipeline benchmark (NLLB draft) ==="
	$(PYTHON) epub.py --model nllb           --strategy light  --book "$(BOOK)" $(BENCH_ARGS) $(PATCH_ARGS) || echo "NLLB baseline failed"
	$(PYTHON) epub.py --model nllb-mistral   --strategy heavy  --book "$(BOOK)" $(BENCH_ARGS) $(PATCH_ARGS) || echo "NLLB+Mistral failed"
	$(PYTHON) epub.py --model nllb-qwen      --strategy heavy  --book "$(BOOK)" $(BENCH_ARGS) $(PATCH_ARGS) || echo "NLLB+Qwen failed"
	$(PYTHON) epub.py --model nllb-tower     --strategy heavy  --book "$(BOOK)" $(BENCH_ARGS) $(PATCH_ARGS) || echo "NLLB+Tower failed"
	$(PYTHON) epub.py --model nllb-qe        --strategy heavy  --book "$(BOOK)" $(BENCH_ARGS) $(PATCH_ARGS) || echo "NLLB+QE failed"
	$(PYTHON) epub.py --model nllb-ct2-qwen  --strategy heavy  --book "$(BOOK)" $(BENCH_ARGS) $(PATCH_ARGS) || echo "NLLB-CT2+Qwen failed"
	@echo "=== MADLAD-CT2 draft ==="
	$(PYTHON) epub.py --model madlad-ct2         --strategy light  --book "$(BOOK)" $(BENCH_ARGS) $(PATCH_ARGS) || echo "MADLAD-CT2 baseline failed"
	$(PYTHON) epub.py --model madlad-ct2-qwen    --strategy heavy  --book "$(BOOK)" $(BENCH_ARGS) $(PATCH_ARGS) || echo "MADLAD-CT2+Qwen failed"
	$(PYTHON) epub.py --model madlad-ct2-mistral --strategy heavy  --book "$(BOOK)" $(BENCH_ARGS) $(PATCH_ARGS) || echo "MADLAD-CT2+Mistral failed"
	$(PYTHON) epub.py --model madlad-ct2-tower   --strategy heavy  --book "$(BOOK)" $(BENCH_ARGS) $(PATCH_ARGS) || echo "MADLAD-CT2+Tower failed"
	@echo "=== Patch benchmark complete ==="

benchmark_literary:
	@echo "=== Literary pipeline benchmark ==="
	$(PYTHON) epub.py --model madlad-ct2-literary --strategy heavy --book "$(BOOK)" \
	    $(BENCH_ARGS) $(LITERARY_ARGS) || echo "MADLAD-CT2 literary failed"
	@echo "=== Literary benchmark complete ==="

benchmark_llm:
	@echo "=== Direct LLM benchmark ==="
	$(MAKE) model-test-gemma         || echo "Gemma failed; continuing"
	$(MAKE) model-test-qwen          || echo "Qwen failed; continuing"
	$(MAKE) model-test-qwen-awq      || echo "Qwen-AWQ failed; continuing"
	$(MAKE) model-test-qwen-vllm     || echo "Qwen-vLLM failed; continuing"
	$(MAKE) model-test-mistral       || echo "Mistral failed; continuing"
	$(MAKE) model-test-mistral-awq   || echo "Mistral-AWQ failed; continuing"
	$(MAKE) model-test-mistral-vllm  || echo "Mistral-vLLM failed; continuing"
	$(MAKE) model-test-tower         || echo "Tower failed; continuing"
	@echo "=== LLM benchmark complete ==="

benchmark_all:
	@echo "=============================="
	@echo "  Full benchmark — chapter $(CHAPTER)"
	@echo "=============================="
	$(MAKE) benchmark_draft
	$(MAKE) benchmark_patch
	$(MAKE) benchmark_llm
	@echo "=============================="
	@echo "  benchmark_all complete"
	@echo "=============================="

# 
# Legacy / convenience targets kept for backward compatibility.
# 

test: validate
	$(MAKE) test-nllb

test-nllb:
	$(PYTHON) epub.py --model nllb --strategy light --book "$(BOOK)" $(CHAPTER_ARGS) $(EXTRA_ARGS)

test-nllb-ct2:
	$(PYTHON) epub.py --model nllb-ct2 --strategy light --book "$(BOOK)" $(CHAPTER_ARGS) $(EXTRA_ARGS)

test-madlad:
	$(PYTHON) epub.py --model madlad --strategy light --book "$(BOOK)" $(CHAPTER_ARGS) $(EXTRA_ARGS)

test-opus:
	$(PYTHON) epub.py --model opus --strategy light --book "$(BOOK)" $(CHAPTER_ARGS) $(EXTRA_ARGS)

test-opus-ct2:
	$(PYTHON) epub.py --model opus-ct2 --strategy light --book "$(BOOK)" $(CHAPTER_ARGS) $(EXTRA_ARGS)

test-gemma:
	$(PYTHON) epub.py --model gemma --strategy heavy --book "$(BOOK)" $(CHAPTER_ARGS) $(EXTRA_ARGS)

test-mistral:
	$(PYTHON) epub.py --model mistral --strategy heavy --book "$(BOOK)" $(CHAPTER_ARGS) $(EXTRA_ARGS)

test-fast: test-nllb
test-fast-ct2: test-nllb-ct2
test-slow: test-madlad-ct2

test-fast-patch: test-fast-mistral

test-fast-mistral:
	$(PYTHON) epub.py --model nllb-mistral --strategy heavy --book "$(BOOK)" $(CHAPTER_ARGS) $(EXTRA_ARGS)

test-fast-tower:
	$(PYTHON) epub.py --model nllb-tower --strategy heavy --book "$(BOOK)" $(CHAPTER_ARGS) $(EXTRA_ARGS)

test-fast-qwen:
	$(PYTHON) epub.py --model nllb-qwen --strategy heavy --book "$(BOOK)" $(CHAPTER_ARGS) $(EXTRA_ARGS)

test-fast-qe:
	$(PYTHON) epub.py --model nllb-qe --strategy heavy --book "$(BOOK)" $(CHAPTER_ARGS) $(EXTRA_ARGS)

test-fast-ct2-patch: test-fast-ct2-mistral

test-fast-ct2-mistral:
	$(PYTHON) epub.py --model nllb-ct2-mistral --strategy heavy --book "$(BOOK)" $(CHAPTER_ARGS) $(EXTRA_ARGS)

test-fast-ct2-tower:
	$(PYTHON) epub.py --model nllb-ct2-tower --strategy heavy --book "$(BOOK)" $(CHAPTER_ARGS) $(EXTRA_ARGS)

test-fast-ct2-qwen:
	$(PYTHON) epub.py --model nllb-ct2-qwen --strategy heavy --book "$(BOOK)" $(CHAPTER_ARGS) $(EXTRA_ARGS)

test-fast-ct2-qe:
	$(PYTHON) epub.py --model nllb-ct2-qe --strategy heavy --book "$(BOOK)" $(CHAPTER_ARGS) $(EXTRA_ARGS)

test-slow-patch: test-slow-mistral

test-slow-mistral:
	$(PYTHON) epub.py --model madlad-ct2-mistral --strategy heavy --book "$(BOOK)" $(CHAPTER_ARGS) $(EXTRA_ARGS)

test-slow-tower:
	$(PYTHON) epub.py --model madlad-ct2-tower --strategy heavy --book "$(BOOK)" $(CHAPTER_ARGS) $(EXTRA_ARGS)

test-slow-qwen:
	$(PYTHON) epub.py --model madlad-ct2-qwen --strategy heavy --book "$(BOOK)" $(CHAPTER_ARGS) $(EXTRA_ARGS)

test-slow-qe:
	$(PYTHON) epub.py --model madlad-ct2-qe --strategy heavy --book "$(BOOK)" $(CHAPTER_ARGS) $(EXTRA_ARGS)

test-best:
	$(PYTHON) epub.py --model madlad-ct2-literary --strategy heavy --book "$(BOOK)" \
	    $(CHAPTER_ARGS) --qe-engine auto --comet-model auto --batch-size 4 \
	    --batch-token-limit 1400 $(LITERARY_ARGS)

test-literary: test-best
translate-best: test-best

prefetch-best:
	$(PYTHON) -c "\
	from huggingface_hub import snapshot_download; \
	[snapshot_download(repo_id=m) for m in ( \
	  'google/madlad400-10b-mt', \
	  'facebook/nllb-200-3.3B', \
	  'Qwen/Qwen2.5-7B-Instruct', \
	  'Unbabel/wmt22-cometkiwi-da', \
	  'Unbabel/wmt20-comet-qe-da', \
	)]"

test-patch:
	$(MAKE) benchmark_patch EXTRA_ARGS="$(EXTRA_ARGS)"

test-llm:
	$(MAKE) benchmark_llm EXTRA_ARGS="$(EXTRA_ARGS)"

test-all:
	$(MAKE) benchmark_all EXTRA_ARGS="$(EXTRA_ARGS)"

compare:
	@echo "Your translations in output/"
	@ls output/nllb/light/*.epub 2>/dev/null || echo "Not found"
	@ls output/gemma/heavy/*.epub 2>/dev/null || echo "Not found"

# 
# 3-chapter comparison: run the best viable models on chapters 1-3 with
# --book-config so force_tu and glossary are active. Delete the translation
# cache first (--refresh-translation-cache) so the 6 fixes take effect.
# 

COMPARE3_ARGS = --book "$(BOOK)" --book-config "$(BOOK_CONFIG)" \
                --no-print-text --refresh-translation-cache $(EXTRA_ARGS)

compare3-draft:
	@echo "=== Draft baseline: NLLB-CT2 and MADLAD-CT2, chapters 1-3 ==="
	-$(PYTHON) epub.py --model nllb-ct2    --strategy light --chapter 1 $(COMPARE3_ARGS)
	-$(PYTHON) epub.py --model nllb-ct2    --strategy light --chapter 2 $(COMPARE3_ARGS)
	-$(PYTHON) epub.py --model nllb-ct2    --strategy light --chapter 3 $(COMPARE3_ARGS)
	-$(PYTHON) epub.py --model madlad-ct2  --strategy light --chapter 1 $(COMPARE3_ARGS)
	-$(PYTHON) epub.py --model madlad-ct2  --strategy light --chapter 2 $(COMPARE3_ARGS)
	-$(PYTHON) epub.py --model madlad-ct2  --strategy light --chapter 3 $(COMPARE3_ARGS)

compare3-qwen:
	@echo "=== MADLAD-CT2 + Qwen patcher, chapters 1-3 ==="
	-$(PYTHON) epub.py --model madlad-ct2-qwen --strategy heavy --chapter 1 $(COMPARE3_ARGS)
	-$(PYTHON) epub.py --model madlad-ct2-qwen --strategy heavy --chapter 2 $(COMPARE3_ARGS)
	-$(PYTHON) epub.py --model madlad-ct2-qwen --strategy heavy --chapter 3 $(COMPARE3_ARGS)

compare3-mistral:
	@echo "=== MADLAD-CT2 + Mistral patcher, chapters 1-3 ==="
	-$(PYTHON) epub.py --model madlad-ct2-mistral --strategy heavy --chapter 1 $(COMPARE3_ARGS)
	-$(PYTHON) epub.py --model madlad-ct2-mistral --strategy heavy --chapter 2 $(COMPARE3_ARGS)
	-$(PYTHON) epub.py --model madlad-ct2-mistral --strategy heavy --chapter 3 $(COMPARE3_ARGS)

compare3-patch:
	$(MAKE) compare3-qwen
	$(MAKE) compare3-mistral

compare3: compare3-draft compare3-patch
	@echo "=== Outputs written to output/ — compare EPUBs side by side ==="
	@$(PYTHON) -c "from pathlib import Path; [print(p) for p in sorted(Path('output').rglob('*.epub'))]"

# 
# ALMA benchmark: compare ALMA-7B-R draft vs MADLAD-CT2 draft, and
#                 ALMA+Qwen variants vs the current best MADLAD-CT2+Qwen.
# Run these AFTER re-downloading the MADLAD-CT2 model.bin.
# 

ALMA_ARGS = --book "$(BOOK)" --book-config "$(BOOK_CONFIG)" \
            --no-print-text --refresh-translation-cache $(EXTRA_ARGS)

benchmark-alma-draft:
	@echo "=== Draft quality: NLLB-CT2 vs MADLAD-CT2 vs ALMA-7B-R (ch1) ==="
	-$(PYTHON) epub.py --model nllb-ct2   --strategy light --chapter 1 $(ALMA_ARGS)
	-$(PYTHON) epub.py --model madlad-ct2 --strategy light --chapter 1 $(ALMA_ARGS)
	-$(PYTHON) epub.py --model alma-7b-r  --strategy light --chapter 1 $(ALMA_ARGS)

benchmark-alma-qwen:
	@echo "=== Patched: MADLAD-CT2+Qwen-7B vs ALMA-7B-R+Qwen-7B (ch1) ==="
	-$(PYTHON) epub.py --model madlad-ct2-qwen     --strategy heavy --chapter 1 $(ALMA_ARGS)
	-$(PYTHON) epub.py --model alma-7b-r-qwen       --strategy heavy --chapter 1 $(ALMA_ARGS)

benchmark-alma-qwen14b:
	@echo "=== Better patcher: MADLAD-CT2+Qwen-14B-AWQ vs ALMA-7B-R+Qwen-14B-AWQ (ch1) ==="
	-$(PYTHON) epub.py --model madlad-ct2-qwen-14b-awq --strategy heavy --chapter 1 $(ALMA_ARGS)
	-$(PYTHON) epub.py --model alma-7b-r-qwen-14b-awq   --strategy heavy --chapter 1 $(ALMA_ARGS)

benchmark-alma: benchmark-alma-draft benchmark-alma-qwen benchmark-alma-qwen14b
	@echo "=== ALMA benchmark complete. Compare EPUBs in output/ ==="
