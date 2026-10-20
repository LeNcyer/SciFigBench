"""LaTeX normalization + tokenization + token-level ANLS for OCR/VLM eval.

Goal: collapse presentation-level differences between two LaTeX strings without
hiding real OCR/transcription errors. Conservative: prefer false-negative
("they didn't match because the normalizer was too cautious") over false-
positive ("they matched because the normalizer washed out an error").

Public API
----------
    normalize_latex(s)         -> str
    tokenize_latex(s)          -> list[str]
    levenshtein_tokens(a, b)   -> int
    anls(gt, pred, threshold=0.5)        -> float
    set_anls(gt_list, pred_list, ...)    -> float   (Hungarian)
    position_anls(gt_list, pred_list, ...) -> float  (pad-with-empty, max denom)

All functions are pure / deterministic / reproducible.

Run `python _latex_norm.py` to execute the self-test suite.
"""
from __future__ import annotations

import re
from typing import Sequence


# ---------------------------------------------------------------------------
# Unicode → LaTeX (trailing space inserted to prevent \alphax-style merges)
# ---------------------------------------------------------------------------

_UNICODE_TO_LATEX: dict[str, str] = {
    # Relations
    "≥": r"\geq", "≤": r"\leq", "≠": r"\neq",
    "≈": r"\approx", "≡": r"\equiv", "∝": r"\propto",
    "≪": r"\ll", "≫": r"\gg",
    # Arrows
    "→": r"\rightarrow", "←": r"\leftarrow", "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow", "⇐": r"\Leftarrow", "⇔": r"\Leftrightarrow",
    "↦": r"\mapsto", "⟶": r"\longrightarrow",
    # Operators
    "×": r"\times", "÷": r"\div", "·": r"\cdot", "∗": r"\ast",
    "±": r"\pm", "∓": r"\mp",
    "∘": r"\circ", "⊕": r"\oplus", "⊗": r"\otimes",
    # Set theory
    "∈": r"\in", "∉": r"\notin", "∋": r"\ni",
    "⊂": r"\subset", "⊃": r"\supset",
    "⊆": r"\subseteq", "⊇": r"\supseteq",
    "∪": r"\cup", "∩": r"\cap", "∅": r"\emptyset",
    # Big operators
    "∑": r"\sum", "∏": r"\prod", "∐": r"\coprod",
    "∫": r"\int", "∬": r"\iint", "∭": r"\iiint", "∮": r"\oint",
    "⋂": r"\bigcap", "⋃": r"\bigcup",
    # Calculus / analysis
    "∞": r"\infty", "∂": r"\partial", "∇": r"\nabla",
    "√": r"\sqrt", "∝": r"\propto",
    "⟨": r"\langle", "⟩": r"\rangle",
    # Logic
    "∀": r"\forall", "∃": r"\exists", "¬": r"\neg",
    "∧": r"\wedge", "∨": r"\vee",
    # Greek lower
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\epsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    "ϕ": r"\phi", "ϵ": r"\epsilon", "ϑ": r"\theta", "ϱ": r"\rho",
    # Greek upper
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma", "Φ": r"\Phi",
    "Ψ": r"\Psi", "Ω": r"\Omega", "ϒ": r"\Upsilon",
    # Blackboard convenience (rare; preserved as-is if no LaTeX equivalent)
    "ℝ": r"\mathbb{R}", "ℕ": r"\mathbb{N}", "ℤ": r"\mathbb{Z}",
    "ℚ": r"\mathbb{Q}", "ℂ": r"\mathbb{C}",
}


# ---------------------------------------------------------------------------
# Environments to strip headers of (content preserved; structural envs absent)
# ---------------------------------------------------------------------------

_STRIP_ENV_NAMES = (
    "equation", "align", "aligned", "alignat", "gather", "gathered",
    "multline", "split", "eqnarray", "math", "displaymath",
)
_STRIP_ENV_GROUP = "|".join(_STRIP_ENV_NAMES)
_ENV_BEGIN_RE = re.compile(
    rf"\\begin\{{(?:{_STRIP_ENV_GROUP})\*?\}}(?:\s*\{{[^{{}}]*\}})?"
)
_ENV_END_RE = re.compile(rf"\\end\{{(?:{_STRIP_ENV_GROUP})\*?\}}")

# Structural envs that MUST be preserved (matrix family, cases, array)
# - we don't strip them, so their \\ and & survive unchanged.


# ---------------------------------------------------------------------------
# Outer math wrappers (only strip when they wrap the WHOLE remaining string)
# ---------------------------------------------------------------------------

_WRAPPER_PAIRS = [
    (re.compile(r"^\s*\\\(\s*(.*?)\s*\\\)\s*$", re.DOTALL), r"\1"),
    (re.compile(r"^\s*\\\[\s*(.*?)\s*\\\]\s*$", re.DOTALL), r"\1"),
    (re.compile(r"^\s*\$\$\s*(.*?)\s*\$\$\s*$", re.DOTALL), r"\1"),
    (re.compile(r"^\s*\$\s*(.*?)\s*\$\s*$", re.DOTALL), r"\1"),
    (re.compile(r"^\s*`\s*(.*?)\s*`\s*$", re.DOTALL), r"\1"),
]


# ---------------------------------------------------------------------------
# Sizing / spacing / annotation commands to remove
# ---------------------------------------------------------------------------

# Delimiter sizing — drop the command, keep the delimiter that follows
_SIZING_RE = re.compile(
    r"\\(?:left|right|middle"
    r"|big|Big|bigg|Bigg"
    r"|bigl|bigr|bigm|Bigl|Bigr|Bigm"
    r"|biggl|biggr|biggm|Biggl|Biggr|Biggm)(?![A-Za-z])"
)

# Spacing commands → single space (later collapsed)
_SPACE_CMD_RE = re.compile(
    r"\\(?:,|;|:| |!"
    r"|quad|qquad|thinspace|medspace|thickspace|enspace|negthinspace"
    r"|negmedspace|negthickspace)(?![A-Za-z])"
)

# Spacing commands with one braced arg → drop entirely (incl. arg)
_SPACE_ARG_CMD_RE = re.compile(
    r"\\(?:phantom|hphantom|vphantom|smash|hspace|vspace|kern|mkern|mskip)"
    r"\*?\s*\{[^{}]*\}"
)

# Numbering / labeling / display hints — drop entirely
_LABEL_RE = re.compile(
    r"\\(?:label|tag|nonumber|notag|limits|nolimits|displaystyle"
    r"|textstyle|scriptstyle|scriptscriptstyle)(?:\s*\{[^{}]*\})?"
)


# ---------------------------------------------------------------------------
# Fractions / relations / operators — alias to canonical
# ---------------------------------------------------------------------------

# Order matters: longer first so e.g. \longrightarrow isn't shadowed by \to.
_ALIAS_RULES = [
    # Fractions
    (r"\\dfrac(?![A-Za-z])",      r"\\frac"),
    (r"\\tfrac(?![A-Za-z])",      r"\\frac"),
    (r"\\cfrac(?![A-Za-z])",      r"\\frac"),
    (r"\\nicefrac(?![A-Za-z])",   r"\\frac"),
    (r"\\sfrac(?![A-Za-z])",      r"\\frac"),
    # Arrows — longer first
    (r"\\longrightarrow(?![A-Za-z])",     r"\\rightarrow"),
    (r"\\longleftarrow(?![A-Za-z])",      r"\\leftarrow"),
    (r"\\longleftrightarrow(?![A-Za-z])", r"\\leftrightarrow"),
    (r"\\Longrightarrow(?![A-Za-z])",     r"\\Rightarrow"),
    (r"\\Longleftarrow(?![A-Za-z])",      r"\\Leftarrow"),
    (r"\\Longleftrightarrow(?![A-Za-z])", r"\\Leftrightarrow"),
    (r"\\implies(?![A-Za-z])",            r"\\Rightarrow"),
    (r"\\impliedby(?![A-Za-z])",          r"\\Leftarrow"),
    (r"\\iff(?![A-Za-z])",                r"\\Leftrightarrow"),
    (r"\\to(?![A-Za-z])",                 r"\\rightarrow"),
    (r"\\gets(?![A-Za-z])",               r"\\leftarrow"),
    # Relations — canonical = long form (matches OCR GT convention)
    (r"\\le(?![A-Za-z])",   r"\\leq"),
    (r"\\ge(?![A-Za-z])",   r"\\geq"),
    (r"\\ne(?![A-Za-z])",   r"\\neq"),
    # Operators
    (r"\\cdotp(?![A-Za-z])",     r"\\cdot"),
    (r"\\centerdot(?![A-Za-z])", r"\\cdot"),
    # Accents (single-char wide variants → narrow)
    (r"\\widehat(?![A-Za-z])",   r"\\hat"),
    (r"\\widetilde(?![A-Za-z])", r"\\tilde"),
]
_ALIAS_COMPILED = [(re.compile(p), r) for p, r in _ALIAS_RULES]


# ---------------------------------------------------------------------------
# Font handling
#   - \text* family → strip command, keep arg
#   - \mathbb / \mathbf / \mathcal / \mathfrak / \mathds / \mathscr  → PRESERVE
#   - \operatorname{X}  → \mathrm{X}   (both write-styles converge on \mathrm)
#   - \mathrm / \mathit / \mathsf / \mathtt → PRESERVE as-is
# ---------------------------------------------------------------------------

_TEXT_FONT_RE = re.compile(
    r"\\(?:text|textbf|textit|textrm|textsf|texttt|textnormal|emph"
    r"|mbox|hbox)\s*\{([^{}]*)\}"
)
_OPERATORNAME_RE = re.compile(r"\\operatorname\*?\s*\{([^{}]*)\}")


# ---------------------------------------------------------------------------
# Prime normalization: x' / x^\prime / x^{\prime} → x^{\prime}
# Multi-prime: x'' → x^{\prime\prime}
# ---------------------------------------------------------------------------

# Match a run of one or more ASCII apostrophes that is NOT preceded by ^_ {
# (already-attached primes). Replace with ^{\prime...}.
_PRIME_RE = re.compile(r"(?<![\\\^_{])'+")
# Existing x^\prime / x^{\prime\prime} stay valid; we also collapse stray
# spacing inside the brace: x^{ \prime } → x^{\prime}
_PRIME_BRACE_TIDY_RE = re.compile(r"\^\{\s*((?:\\prime\s*)+)\}")


def _collapse_prime_runs(s: str) -> str:
    def repl(m: re.Match) -> str:
        n = len(m.group(0))
        return "^{" + r"\prime" * n + "}"
    s = _PRIME_RE.sub(repl, s)
    # Tidy any pre-existing braced primes
    s = _PRIME_BRACE_TIDY_RE.sub(
        lambda m: "^{" + "".join(m.group(1).split()) + "}", s)
    return s


# ---------------------------------------------------------------------------
# Sub/sup single-token brace wrapping
#
# After `^` or `_`, find the extent of the next token and wrap with `{...}`.
# A "token" is:
#   - already braced group `{...}` (leave alone)
#   - `\command` + optional `*` + any tightly-following `{...}` arg groups
#   - `\<escaped non-letter>` (e.g. `\%`)
#   - single character (letter / digit / other)
# ---------------------------------------------------------------------------

def _next_token_extent(s: str, pos: int) -> int:
    """Return exclusive end index of the token starting at s[pos]. Caller
    guarantees pos < len(s)."""
    c = s[pos]
    if c == "{":
        depth = 1
        i = pos + 1
        while i < len(s) and depth > 0:
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
            i += 1
        return i
    if c == "\\":
        i = pos + 1
        if i < len(s) and not s[i].isalpha():
            return i + 1  # escaped non-letter e.g. \%
        while i < len(s) and s[i].isalpha():
            i += 1
        if i < len(s) and s[i] == "*":
            i += 1
        # Gobble immediate brace-args (no intervening space) so that
        # x^\frac{a}{b} is treated as one extent.
        while i < len(s) and s[i] == "{":
            depth = 1
            j = i + 1
            while j < len(s) and depth > 0:
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                j += 1
            i = j
        return i
    return pos + 1


def _wrap_subsup(s: str) -> str:
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        # Skip escaped underscore/caret: `\_` / `\^` are literal chars, not sub/sup.
        if c == "\\" and i + 1 < n and s[i + 1] in "_^":
            out.append(s[i:i + 2])
            i += 2
            continue
        out.append(c)
        if c == "_" or c == "^":
            i += 1
            while i < n and s[i] == " ":
                i += 1
            if i >= n:
                continue
            end = _next_token_extent(s, i)
            tok = s[i:end]
            if tok.startswith("{"):
                out.append(tok)
            else:
                out.append("{")
                out.append(tok)
                out.append("}")
            i = end
        else:
            i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Top-level: fixpoint-safe normalize
# ---------------------------------------------------------------------------

def _strip_wrappers_fixpoint(s: str) -> str:
    for _ in range(8):
        new = s
        for pat, repl in _WRAPPER_PAIRS:
            new = pat.sub(repl, new).strip()
        if new == s:
            break
        s = new
    return s


def _strip_envs(s: str) -> str:
    s = _ENV_BEGIN_RE.sub("", s)
    s = _ENV_END_RE.sub("", s)
    return s


def _strip_text_fonts(s: str) -> str:
    for _ in range(4):
        new = _TEXT_FONT_RE.sub(r"\1", s)
        if new == s:
            break
        s = new
    return s


def _operatorname_to_mathrm(s: str) -> str:
    return _OPERATORNAME_RE.sub(lambda m: r"\mathrm{" + m.group(1) + "}", s)


def _apply_aliases(s: str) -> str:
    for pat, repl in _ALIAS_COMPILED:
        s = pat.sub(repl, s)
    return s


def _strip_redundant_outer_braces(s: str) -> str:
    """If the whole string is wrapped in braces and the braces are balanced
    around the entire content, drop one pair. Repeat to fixpoint."""
    for _ in range(4):
        s = s.strip()
        if not (s.startswith("{") and s.endswith("}")):
            return s
        depth = 0
        balanced_whole = True
        for i, ch in enumerate(s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    balanced_whole = False
                    break
        if not balanced_whole:
            return s
        s = s[1:-1]
    return s


def normalize_latex(s: str | None) -> str:
    """Conservative Tier 1 + Tier 2 LaTeX normalization.

    Idempotent (multiple applications yield the same result).
    """
    if s is None:
        return ""
    s = str(s)
    # 1. Unicode → LaTeX (pad both sides with space; whitespace gets collapsed later)
    for u, l in _UNICODE_TO_LATEX.items():
        if u in s:
            s = s.replace(u, " " + l + " ")
    # 2. Strip equation-family environment headers (matrix family preserved)
    s = _strip_envs(s)
    # 3. Strip outer math/code wrappers, fixpoint
    s = _strip_wrappers_fixpoint(s)
    # 4. Drop \left \right \big ... \middle ...
    s = _SIZING_RE.sub("", s)
    # 5. Drop spacing-arg commands first (so their {...} arg goes away atomically)
    s = _SPACE_ARG_CMD_RE.sub(" ", s)
    # 6. Drop annotation commands (\label \tag \limits \nolimits ...)
    s = _LABEL_RE.sub("", s)
    # 7. Drop spacing commands (\, \; \quad ...)
    s = _SPACE_CMD_RE.sub(" ", s)
    # 8. Strip text-font wrappers (\text \textbf \mbox ...)
    s = _strip_text_fonts(s)
    # 9. \operatorname{X} → \mathrm{X}  (canonical for OCR fidelity)
    s = _operatorname_to_mathrm(s)
    # 10. Command aliases (frac family, arrow long forms, relation aliases, ...)
    s = _apply_aliases(s)
    # 11. Prime normalization
    s = _collapse_prime_runs(s)
    # 12. Sub/sup single-token brace wrapping
    s = _wrap_subsup(s)
    # 13. Strip redundant whole-string outer braces
    s = _strip_redundant_outer_braces(s)
    # 14. Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"\\[a-zA-Z]+\*?"          # \frac, \alpha, \begin, \frac* etc.
    r"|\\[^a-zA-Z\s]"          # escaped non-letter: \%, \{, \\, \&
    r"|[0-9]"                  # single digit
    r"|[a-zA-Z]"               # single letter
    r"|[^\s]"                  # operators, braces, ^, _, +, =, (, ) ...
)


def tokenize_latex(s: str | None) -> list[str]:
    if s is None:
        return []
    return _TOKEN_RE.findall(str(s))


# ---------------------------------------------------------------------------
# Token-level Levenshtein + ANLS
# ---------------------------------------------------------------------------

def levenshtein_tokens(a: Sequence[str], b: Sequence[str]) -> int:
    """Token-level edit distance (insert/delete/substitute each cost 1)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr[j] = min(ins, dele, sub)
        prev = curr
    return prev[-1]


def anls(gt: str, pred: str, *, normalize: bool = True,
         threshold: float = 0.5) -> float:
    """Token-level ANLS on optionally-normalized LaTeX strings.

    Both empty → 1.0; exactly one empty → 0.0; otherwise
    NLS = 1 - editdist / max(len_tokens_gt, len_tokens_pred), zeroed below
    `threshold` (DocVQA / ANLS convention).
    """
    gt_n = normalize_latex(gt) if normalize else (gt or "")
    pr_n = normalize_latex(pred) if normalize else (pred or "")
    g = tokenize_latex(gt_n)
    p = tokenize_latex(pr_n)
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    d = levenshtein_tokens(g, p)
    nl = 1.0 - d / max(len(g), len(p))
    return nl if nl >= threshold else 0.0


def position_anls(gt_list: list[str], pred_list: list[str], *,
                  normalize: bool = True, threshold: float = 0.5) -> float:
    """Position-aligned ANLS. Pad shorter side with empty strings; both
    "extra" preds and "missing" preds contribute 0 to the average over
    max(len_gt, len_pred). Used for ocr_unified.flow_path where the prompt
    binds position i to color i."""
    n = max(len(gt_list), len(pred_list))
    if n == 0:
        return 1.0
    s = 0.0
    for i in range(n):
        g = gt_list[i] if i < len(gt_list) else ""
        p = pred_list[i] if i < len(pred_list) else ""
        s += anls(g, p, normalize=normalize, threshold=threshold)
    return s / n


def set_anls(gt_list: list[str], pred_list: list[str], *,
             normalize: bool = True, threshold: float = 0.5) -> float:
    """Set-ANLS* (Hungarian). Pad shorter side with empty so extra preds
    contribute 0 (hallucination) and missing gt contributes 0 (recall miss).
    Both empty → 1.0. Used for content_task."""
    if not gt_list and not pred_list:
        return 1.0
    n = max(len(gt_list), len(pred_list))
    gt_p = list(gt_list) + [""] * (n - len(gt_list))
    pr_p = list(pred_list) + [""] * (n - len(pred_list))
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        cost = np.zeros((n, n), dtype=float)
        for i, g in enumerate(gt_p):
            for j, p in enumerate(pr_p):
                cost[i, j] = 1.0 - anls(g, p, normalize=normalize,
                                        threshold=threshold)
        ri, ci = linear_sum_assignment(cost)
        total = sum(1.0 - cost[i, j] for i, j in zip(ri, ci))
    except ImportError:
        # Brute force optimal for small n (n! ≤ 40,320 at n=8 is fine).
        # Falls back to greedy for n > 8 (sub-optimal but reasonable).
        sim = [[anls(g, p, normalize=normalize, threshold=threshold)
                for p in pr_p] for g in gt_p]
        if n <= 8:
            import itertools
            best = -1.0
            for perm in itertools.permutations(range(n)):
                cand = sum(sim[i][perm[i]] for i in range(n))
                if cand > best:
                    best = cand
            total = best
        else:
            used = set()
            total = 0.0
            for i in range(n):
                best_j, best_s = None, -1.0
                for j in range(n):
                    if j in used:
                        continue
                    if sim[i][j] > best_s:
                        best_s, best_j = sim[i][j], j
                if best_j is not None:
                    used.add(best_j)
                    total += best_s
    return total / n


# ---------------------------------------------------------------------------
# Self-test (run: python _latex_norm.py)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    cases: list[tuple[str, str, str]] = [
        # (name, input, expected normalized)
        # ---- Wrappers ----
        ("inline $", r"$x^2$",                            r"x^{2}"),
        ("display $$", r"$$x^2$$",                        r"x^{2}"),
        ("inline \\(\\)", r"\(x^2\)",                     r"x^{2}"),
        ("display \\[\\]", r"\[x^2\]",                    r"x^{2}"),
        ("backtick", r"`\frac{1}{2}`",                    r"\frac{1}{2}"),
        ("nested wrap", r"$$\(x^2\)$$",                   r"x^{2}"),

        # ---- Environments ----
        ("equation env", r"\begin{equation}x^2\end{equation}", r"x^{2}"),
        ("align* env",   r"\begin{align*}x^2\end{align*}",     r"x^{2}"),
        ("aligned",      r"\begin{aligned}x^2\end{aligned}",   r"x^{2}"),

        # ---- Fractions ----
        ("dfrac",  r"\dfrac{1}{2}",  r"\frac{1}{2}"),
        ("tfrac",  r"\tfrac{1}{2}",  r"\frac{1}{2}"),
        ("cfrac",  r"\cfrac{1}{1+\cfrac{1}{x}}",
                                     r"\frac{1}{1+\frac{1}{x}}"),

        # ---- Sizing ----
        ("left/right paren", r"\left( x+y \right)",  r"( x+y )"),
        ("bigl/bigr brack",  r"\bigl[ x \bigr]",     r"[ x ]"),
        ("middle",           r"\left| x \middle| y \right|",
                                                     r"| x | y |"),

        # ---- Sub/sup brace ----
        ("super single digit", r"x^2",          r"x^{2}"),
        ("sub single letter",  r"x_i",          r"x_{i}"),
        ("super command",      r"x^\alpha",     r"x^{\alpha}"),
        ("super with frac",    r"x^\frac{a}{b}", r"x^{\frac{a}{b}}"),
        ("already braced",     r"x^{abc}",      r"x^{abc}"),
        ("trap: x^ab",         r"x^ab",         r"x^{a}b"),
        ("trap: x_12",         r"x_12",         r"x_{1}2"),

        # ---- Primes ----
        ("single prime", r"x'",      r"x^{\prime}"),
        ("double prime", r"x''",     r"x^{\prime\prime}"),
        ("triple prime", r"x'''",    r"x^{\prime\prime\prime}"),

        # ---- Relation aliases ----
        ("\\le",  r"a \le b",  r"a \leq b"),
        ("\\ge",  r"a \ge b",  r"a \geq b"),
        ("\\to",  r"a \to b",  r"a \rightarrow b"),
        ("\\implies", r"P \implies Q", r"P \Rightarrow Q"),
        ("\\iff", r"P \iff Q", r"P \Leftrightarrow Q"),
        ("\\longrightarrow", r"a \longrightarrow b", r"a \rightarrow b"),

        # ---- Unicode → LaTeX ----
        ("alpha",     "α",       r"\alpha"),
        ("geq",       "α≥β",     r"\alpha \geq \beta"),
        ("rightarrow", "x→y",    r"x \rightarrow y"),
        ("bbR",       "ℝ",       r"\mathbb{R}"),

        # ---- Font preservation / stripping ----
        ("mathbb R preserved", r"\mathbb{R}",     r"\mathbb{R}"),
        ("mathbf x preserved", r"\mathbf{x}",     r"\mathbf{x}"),
        ("mathcal L preserved", r"\mathcal{L}",   r"\mathcal{L}"),
        ("text stripped",      r"\text{hello}",   r"hello"),
        ("textbf stripped",    r"\textbf{x}",     r"x"),
        ("mbox stripped",      r"\mbox{x}",       r"x"),
        ("operatorname → mathrm", r"\operatorname{softmax}(x)",
                                  r"\mathrm{softmax}(x)"),
        ("mathrm preserved",   r"\mathrm{d}x",    r"\mathrm{d}x"),

        # ---- Spacing ----
        ("\\,",      r"a \, b",          r"a b"),
        ("\\quad",   r"a \quad b",       r"a b"),
        ("\\phantom", r"a \phantom{xyz} b", r"a b"),
        ("\\!",      r"a \! b",          r"a b"),

        # ---- Limits noise ----
        ("\\limits", r"\sum\limits_{i=1}^n x_i",
                     r"\sum_{i=1}^{n} x_{i}"),
        ("\\nolimits", r"\sum\nolimits_{i=1}^{n} x_{i}",
                       r"\sum_{i=1}^{n} x_{i}"),

        # ---- Escaped chars preserved ----
        ("\\%",  r"50\%",   r"50\%"),
        ("\\_",  r"a\_b",   r"a\_b"),
        ("\\&",  r"a\&b",   r"a\&b"),

        # ---- Redundant outer brace ----
        ("outer brace",         r"{x+y}",        r"x+y"),
        ("double outer brace",  r"{{x+y}}",      r"x+y"),
        ("INNER brace preserved", r"x^{2y}",     r"x^{2y}"),

        # ---- Idempotency probes ----
        ("already canonical", r"\frac{1}{2}", r"\frac{1}{2}"),

        # ---- Real-world example from diag ----
        ("diag GT",  r"\quad Z_{u}",       r"Z_{u}"),
        ("diag PRED",r"\( \hat{Y}_{v} \)", r"\hat{Y}_{v}"),
    ]
    passed = 0
    failed: list[tuple[str, str, str, str]] = []
    for name, inp, exp in cases:
        got = normalize_latex(inp)
        if got == exp:
            passed += 1
        else:
            failed.append((name, inp, exp, got))

    # Idempotency: applying normalize twice == once
    for name, inp, _ in cases:
        once = normalize_latex(inp)
        twice = normalize_latex(once)
        if once != twice:
            failed.append((f"idempotent[{name}]", inp,
                          f"once={once!r}", f"twice={twice!r}"))

    # Tokenizer spot-checks
    tok_cases = [
        (r"\frac{1}{2}",   [r"\frac", "{", "1", "}", "{", "2", "}"]),
        (r"x^{2}+y",       ["x", "^", "{", "2", "}", "+", "y"]),
        (r"\alpha \beta",  [r"\alpha", r"\beta"]),
        ("12",             ["1", "2"]),
        (r"\%",            [r"\%"]),
        (r"\hat{Y}_{v}",   [r"\hat", "{", "Y", "}", "_", "{", "v", "}"]),
        (r"\mathbb{R}",    [r"\mathbb", "{", "R", "}"]),
    ]
    for inp, exp in tok_cases:
        got = tokenize_latex(inp)
        if got != exp:
            failed.append((f"tokenize[{inp}]", inp, str(exp), str(got)))

    # ANLS spot-checks
    anls_cases: list[tuple[str, str, str, float]] = [
        # (description, gt, pred, expected score >= or == comment in print)
        ("identical",            r"\frac{1}{2}", r"\frac{1}{2}", 1.0),
        ("dfrac vs frac",        r"\frac{1}{2}", r"\dfrac{1}{2}", 1.0),
        ("wrapped vs naked",     r"\hat{Y}_{v}", r"\( \hat{Y}_{v} \)", 1.0),
        ("x^2 vs x^{2}",         r"x^{2}",       r"x^2",          1.0),
        ("le vs leq",            r"a \leq b",    r"a \le b",      1.0),
        ("to vs rightarrow",     r"a \rightarrow b", r"a \to b",  1.0),
        ("alpha unicode",        r"\alpha",      "α",             1.0),
    ]
    for desc, gt, pred, exp in anls_cases:
        got = anls(gt, pred)
        if abs(got - exp) > 1e-6:
            failed.append((f"anls[{desc}]", f"{gt!r} vs {pred!r}",
                          f"={exp}", f"got={got:.4f}"))

    # set-ANLS spot-check (user's diag case)
    gt = [r"\quad Z_{u}", r"\hat{Y}_{v}"]
    pred = [r"\( \hat{Y}_{v} \)"]
    got = set_anls(gt, pred)
    expected_min = 0.45  # one perfect match, one missing → ~0.5
    if got < expected_min:
        failed.append(("set_anls[diag]",
                       f"gt={gt} pred={pred}",
                       f">= {expected_min}",
                       f"got={got:.4f}"))

    print(f"\n[_latex_norm self-test] passed: {passed}/{len(cases)} structural cases")
    print(f"                        + {len(tok_cases)} tokenizer cases")
    print(f"                        + {len(anls_cases)+1} anls cases")
    if failed:
        print(f"\nFAILED: {len(failed)}")
        for f in failed:
            print(f"  - {f[0]}")
            print(f"      input:    {f[1]}")
            print(f"      expected: {f[2]}")
            print(f"      got:      {f[3]}")
        raise SystemExit(1)
    print("ALL GREEN")


if __name__ == "__main__":
    _self_test()
