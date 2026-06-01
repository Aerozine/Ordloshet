"""Optional French grammar checking via LanguageTool.

Requires LanguageTool server running locally:
  docker run -p 8081:8081 erikvl87/languagetool

Or set LANGUAGETOOL_URL to a remote instance.
"""
from __future__ import annotations

import urllib.request
import urllib.parse
import json


def check_french(
    text: str,
    url: str = "http://localhost:8081",
    min_length: int = 40,
    max_errors: int = 5,
) -> list[str]:
    """Return a list of grammar issue messages (empty list = no errors or service unavailable)."""
    if not text or len(text) < min_length:
        return []
    try:
        data = urllib.parse.urlencode({
            "text": text,
            "language": "fr",
            "disabledRules": "WHITESPACE_RULE,UNPAIRED_BRACKETS",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{url.rstrip('/')}/v2/check",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            result = json.loads(resp.read())
        matches = result.get("matches", [])[:max_errors]
        return [
            f"[{m.get('rule', {}).get('id', '?')}] {m.get('message', '')} "
            f"(offset {m.get('offset', 0)})"
            for m in matches
        ]
    except Exception:
        return []
