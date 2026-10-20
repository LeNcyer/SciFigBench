"""Model-output parsing for JSON lists, Booleans and integers."""
from __future__ import annotations
import re
import json
from typing import Any

_FENCE_RE = re.compile(
    r"\A(?P<open>`{3,})[ \t]*(?:json[ \t]*)?\r?\n"
    r"(?P<body>.*?)\r?\n(?P<close>`{3,})[ \t]*\Z",
    re.DOTALL | re.IGNORECASE,
)
_BOOL_RE = re.compile(r"\b(true|false)\b", re.IGNORECASE)
_INT_RE = re.compile(r"[-+]?\d+")

def _strip_md_fence(s: str) -> str:
    s = s.strip()
    m = _FENCE_RE.fullmatch(s)
    if m and len(m.group("close")) >= len(m.group("open")):
        return m.group("body").strip()
    return s

def parse_json_list(raw: Any) -> list[str] | None:
    """Extract a JSON list-of-strings from raw model output.

    Fallbacks:
      1. json.loads raw output (preserve JSON string contents)
      2. strip a complete outer ```json``` / ``` fence, then json.loads
      3. first-balanced-list fallback; otherwise None
    """
    if raw is None:
        return None
    # Parse valid JSON before interpreting any Markdown. Backticks inside a
    # JSON string are transcription content, not presentation markup.
    s = str(raw).strip()
    try:
        v = json.loads(s)
    except Exception:
        # Only a complete outer fence may be stripped. Other nonstandard
        # output is handled by the balanced-list fallback below.
        s = _strip_md_fence(s)
        try:
            v = json.loads(s)
        except Exception:
            v = None
    if isinstance(v, list):
        return [str(x) for x in v]
    start = s.find("[")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    v = json.loads(s[start:i + 1])
                    return [str(x) for x in v] if isinstance(v, list) else None
                except Exception:
                    return None
    return None

def parse_bool(raw: Any) -> str | None:
    if raw is None:
        return None
    m = _BOOL_RE.search(str(raw))
    return m.group(1).lower() if m else None

def parse_int(raw: Any) -> int | None:
    if raw is None:
        return None
    m = _INT_RE.search(str(raw))
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None
