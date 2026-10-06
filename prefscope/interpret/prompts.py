"""Load prompt templates adapted from WIMHF and parse model output.

See ``prompts/WIMHF_LICENSE.txt`` for the upstream copyright and license.
"""
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
    lines = [ln for ln in (line.strip() for line in response.splitlines()) if ln]
    cand = _clean_phrase(lines[-1].rstrip('"')) if lines else ""  # completion stub: 'phrase"'
    return cand if _looks_like_concept(cand) else ""


_STATUSES = ("ok", "polysemantic", "insufficient_evidence")


def _evidence_summary(obj: dict) -> str:
    """Normalize a short model-written account of the displayed evidence."""
    value = obj.get("evidence_summary")
    if value in (None, "", "null"):
        return ""
    # This is a visible evidence note, not hidden reasoning. Keep it bounded and flat so
    # one verbose provider cannot dominate checkpoints or CSV exports.
    return " ".join(str(value).strip().split())[:1500]


def _polysemantic_label(concept: str) -> str:
    """Normalize at most three explicitly structured mixed-evidence clusters."""
    clusters = [_clean_phrase(part) for part in concept.split(";")]
    clusters = [part for part in clusters
                if part and not part.endswith("?") and _NON_CONCEPT.match(part) is None]
    return "; ".join(clusters[:3])


def parse_concept_result(response: str) -> dict:
    """Parse structured naming output plus an optional visible evidence summary.

    Older three-field objects remain valid and receive an empty ``evidence_summary``.
    ``ok`` carries one atomic hypothesis. ``polysemantic`` may carry a descriptive
    semicolon-separated label for recurring clusters, but remains an abstention for
    downstream verification. ``insufficient_evidence`` always has an empty concept.
    Back-compat: old null polysemantic outputs, bare concept JSON, and plain phrases work.
    """
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", response or "").strip()
    if not cleaned or cleaned.startswith("<<ERROR"):
        return {"status": "insufficient_evidence", "concept": "", "confidence": "",
                "evidence_summary": ""}
    try:
        obj = json.loads(cleaned)
    except Exception:
        obj = None
    if isinstance(obj, dict):
        status = str(obj.get("status") or "ok").strip().lower()
        if status not in _STATUSES:
            status = "insufficient_evidence"
        confidence = str(obj.get("confidence") or "").strip().lower()
        if confidence not in ("high", "medium", "low"):
            confidence = ""
        c = obj.get("concept")
        concept = "" if c in (None, "", "null") else _clean_phrase(str(c))
        if status == "insufficient_evidence":
            concept = ""
        elif status == "ok" and len([part for part in concept.split(";") if part.strip()]) > 1:
            status, concept = "polysemantic", _polysemantic_label(concept)
        elif status == "ok" and not _looks_like_concept(concept):
            status, concept = "insufficient_evidence", ""
        elif status == "polysemantic":
            concept = _polysemantic_label(concept)
        return {"status": status, "concept": concept, "confidence": confidence,
                "evidence_summary": _evidence_summary(obj)}
    concept = parse_concept(response)
    if len([part for part in concept.split(";") if part.strip()]) > 1:
        return {"status": "polysemantic", "concept": _polysemantic_label(concept),
                "confidence": "", "evidence_summary": ""}
    return {"status": "ok" if concept else "insufficient_evidence",
            "concept": concept, "confidence": "", "evidence_summary": ""}


def fallback_concept_result(candidates: list[dict]) -> dict:
    """Choose a conservative usable proposal when synthesis fails."""
    for status in ("polysemantic", "ok"):
        for candidate in candidates:
            if candidate.get("status") == status and candidate.get("concept"):
                return candidate
    return {"status": "insufficient_evidence", "concept": "", "confidence": "low",
            "evidence_summary": ""}


def parse_synthesis_result(response: str, candidates: list[dict]) -> dict:
    """Parse synthesis, falling back only when its response is malformed or empty."""
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", response or "").strip()
    try:
        obj = json.loads(cleaned)
    except Exception:
        obj = None
    if not isinstance(obj, dict):
        return fallback_concept_result(candidates)
    result = parse_concept_result(cleaned)
    if result["status"] != "insufficient_evidence":
        return result
    status = str(obj.get("status") or "").strip().lower()
    if status and status not in ("ok", "polysemantic"):
        return result
    return fallback_concept_result(candidates)


def parse_support_audit(response: str, *, n_active: int, n_control: int) -> dict:
    """Parse the independent literal-presence audit for a proposed concept."""
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", response or "").strip()
    try:
        obj = json.loads(cleaned)
    except Exception:
        obj = None

    def bool_list(key: str, expected: int):
        values = obj.get(key) if isinstance(obj, dict) else None
        if (not isinstance(values, list) or len(values) != expected
                or any(type(v) is not bool for v in values)):
            return None
        return values

    active = bool_list("active_matches", n_active)
    control = bool_list("control_matches", n_control)
    valid = active is not None and control is not None
    passed = bool(valid and n_active > 0 and n_control > 0
                  and all(active) and not any(control))
    return {
        "active_matches": active,
        "control_matches": control,
        "valid": valid,
        "pass": passed,
        "active_support": int(sum(active)) if active is not None else 0,
        "control_violations": int(sum(control)) if control is not None else 0,
    }


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
    import re
    return re.sub(r"<\s*(/?)\s*example", r"<\\\1example", text or "",
                  flags=re.IGNORECASE)


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
