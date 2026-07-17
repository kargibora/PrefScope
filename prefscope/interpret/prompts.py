"""Load WIMHF prompt templates (vendored verbatim) and parse model output."""
from __future__ import annotations

import json
import re
from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(name: str) -> str:
    """Read a vendored prompt template by stem (no .txt)."""
    return (_PROMPT_DIR / f"{name}.txt").read_text()


def _clean_phrase(s: str) -> str:
    s = s.strip()
    if s.startswith("- "):
        s = s[2:]
    if s.startswith("-"):
        s = s[1:]
    return s.strip().strip('"').strip()


# a bare last line that is clearly NOT a feature description: chat role-play, a
# label/letter fragment, a list number, or a question (the model answered the
# examples instead of describing the concept). Chat models like GLM do this when
# they don't complete the WIMHF ``- "`` stub.
_NON_CONCEPT = re.compile(
    r"""^(?:[abq]\s*[:.]            # "A:", "B." label fragments
        | \d+\s*[.)]               # "1.", "2)" list markers
        | (?:hey|hi|hello|sure|okay|ok|as\s+an|i\s*'?m|i\s+am)\b  # role-play openers
    )""", re.IGNORECASE | re.VERBOSE)


def _looks_like_concept(s: str) -> bool:
    """A WIMHF concept is a 3rd-person descriptive phrase; reject obvious non-concepts
    so a chat model's role-play/answer never leaks in as a 'concept'."""
    s = s.strip()
    if len(s.split()) < 2 or s.endswith("?"):
        return False
    return _NON_CONCEPT.match(s) is None


def parse_concept(response: str) -> str:
    """Extract the quoted concept phrase.

    WIMHF emits a single quoted phrase (often as ``- "phrase"``) and GPT-4.1 obeys
    exactly. Reasoning/chat models add `<think>` blocks or ignore the format, so we:
    strip thinking, prefer the WIMHF bullet, then any quoted span, then a bare last
    line — but only if it actually looks like a concept; otherwise return "" (an
    honest abstain that the pipeline flags, never garbage like "A:" or a question).
    """
    response = re.sub(r"(?is)<think>.*?</think>", "", response or "").strip()
    # structured output (preferred): {"concept": "<phrase>"} — robust across chat models
    try:                                   # accept the schema key + common synonyms GLM drifts to
        obj = json.loads(response)
        if isinstance(obj, dict):
            for key in ("concept", "feature", "description", "phrase"):
                v = str(obj.get(key) or "").strip()
                if v:
                    return _clean_phrase(v)
    except Exception:
        pass
    m = re.search(r'"(?:concept|feature|description|phrase)"\s*:\s*"([^"]+)"', response)
    if m:
        return _clean_phrase(m.group(1))
    if response.lstrip().startswith("{"):
        return ""   # JSON-shaped but no usable "concept" (e.g. model dumped reasoning JSON)
    bullets = re.findall(r'(?m)^\s*-\s*"([^"]+)"', response)
    if bullets:
        return _clean_phrase(bullets[-1])
    quotes = re.findall(r'"([^"]+)"', response)
    if quotes:
        c = _clean_phrase(quotes[-1])
        if _looks_like_concept(c):
            return c
    lines = [ln for ln in (l.strip() for l in response.splitlines()) if ln]
    cand = _clean_phrase(lines[-1].rstrip('"')) if lines else ""  # completion stub: 'phrase"'
    return cand if _looks_like_concept(cand) else ""


_STATUSES = ("ok", "polysemantic", "insufficient_evidence")


def parse_concept_result(response: str) -> dict:
    """Parse the structured naming output ``{status, concept, confidence}``.

    Abstention is first-class: ``status`` in {ok, polysemantic, insufficient_evidence}.
    For a non-ok status we return an EMPTY concept — the feature is flagged, never
    force-named. Back-compat: a bare ``{"concept": "..."}`` or a plain phrase parses as
    ``status="ok"`` so old prompts/models keep working."""
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", response or "").strip()
    # An empty reply or the name.py error sentinel (`<<ERROR: ...>>`) is a MISSING naming,
    # not a concept — never let it leak in as a feature label.
    if not cleaned or cleaned.startswith("<<ERROR"):
        return {"status": "insufficient_evidence", "concept": "", "confidence": ""}
    try:
        obj = json.loads(cleaned)
    except Exception:
        obj = None
    if isinstance(obj, dict):
        # Fail CLOSED on a malformed structured response: an unrecognized status is an
        # abstain, not an "ok" (#6) — the old code turned unknown status into ok.
        status = str(obj.get("status") or "ok").strip().lower()
        if status not in _STATUSES:
            status = "insufficient_evidence"
        confidence = str(obj.get("confidence") or "").strip().lower()
        if confidence not in ("high", "medium", "low"):
            confidence = ""                    # validate, don't pass junk through
        c = obj.get("concept")
        concept = "" if c in (None, "", "null") else _clean_phrase(str(c))
        if status != "ok":
            concept = ""                       # abstain -> no forced label
        elif not _looks_like_concept(concept):
            status, concept = "insufficient_evidence", ""   # "ok" but no real phrase -> abstain
        return {"status": status, "concept": concept, "confidence": confidence}
    # non-JSON fallback: reuse the legacy extractor, treat as ok
    concept = parse_concept(response)
    return {"status": "ok" if concept else "insufficient_evidence",
            "concept": concept, "confidence": ""}


def parse_label(raw: str):
    """A -> +1, B -> -1, Tie -> 0. Unparseable/empty -> None (a MISSING observation,
    NOT evidence). Matches the FIRST word only, so "As an evaluator…" is None rather than
    A (the old `startswith("A")` scored that as a real A vote) and "Both" is None, not B."""
    s = (raw or "").strip().strip('"').strip()
    m = re.match(r"[^A-Za-z]*([A-Za-z]+)", s)
    if not m:
        return None
    w = m.group(1).upper()
    return {"TIE": 0, "A": 1, "B": -1}.get(w, None)


def parse_presence(raw: str):
    """Single-text presence: Yes/present -> 1, No/absent -> 0. Unclear / unparseable / empty
    -> None (MISSING, not a No). The old code returned 0 for garbage, biasing fidelity down."""
    s = (raw or "").strip().lower()
    if not s:
        return None
    # Unclear FIRST — before the yes/no prefixes — so "not sure" isn't caught by the "no"
    # prefix ("not".startswith("no")) and "Unclear because no evidence…" isn't read as No.
    if s.startswith(("unclear", "unsure", "not sure", "n/a", "cannot", "can't", "can not")):
        return None
    if s.startswith("yes") or s.startswith("present"):
        return 1
    if s.startswith("no") or s.startswith("absent"):
        return 0
    if re.search(r"\bunclear\b|\bunsure\b|\bnot sure\b|\bcannot tell\b|\bcan.?t tell\b", s):
        return None
    if re.search(r"\byes\b", s):
        return 1
    if re.search(r"\bno\b", s):
        return 0
    return None


def truncate(s: str, n: int) -> str:
    """Cap a response to ~n chars keeping HEAD + TAIL. The defining behaviour is often at
    the END — a final refusal, a conclusion, or the answer after a long preamble — which
    head-only truncation silently erases. Marks the cut so the interpreter knows content
    was omitted. Falls back to head-only when n is too small for the marker."""
    if not isinstance(s, str):
        return ""
    s = s.replace("\r", " ").strip()
    if len(s) <= n:
        return s
    marker = " …[omitted]… "
    budget = n - len(marker)
    if budget <= 8:                       # too small to split usefully
        return s[: n - 1] + "…"
    head = (budget * 7) // 10             # 70% head / 30% tail
    tail = budget - head
    return s[:head].rstrip() + marker + s[len(s) - tail:].lstrip()


def shield(text: str) -> str:
    """Neutralize the <example> delimiter inside UNTRUSTED dataset text, so a response
    can't close the block early and inject instructions after it (prompt-injection guard).
    Paired with the system-prompt rule that <example> content is data, never instructions."""
    return (text or "").replace("</example", "<\\/example").replace("<example", "<\\example")


def fmt_example(idx: int, row: dict) -> str:
    """Render one pair as CONTEXT / RESPONSE A / RESPONSE B, wrapped as an untrusted
    <example> block with its activation."""
    z = row.get("signed_z_diff", 0.0)
    return (
        f'<example idx="{idx}" signed_activation="{z:+.3f}">\n'
        f"CONTEXT (user prompt):\n{shield(row.get('prompt', ''))}\n\n"
        f"RESPONSE A:\n{shield(row.get('completion_a', ''))}\n\n"
        f"RESPONSE B:\n{shield(row.get('completion_b', ''))}\n"
        f"</example>\n"
    )
