"""Load OpenJury annotation JSON(s) into a normalized battle table.

Drops rows with parse errors, missing completions/prompt, or missing judge
preference. Quarter-tie judge preferences (0.25/0.75) collapse to 0.5. A missing
label is never imputed as a tie — the row is dropped. Deduplicates on
instruction_id.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from prefscope.config import CONFIG

_QUARTER_TIES = set(CONFIG.quarter_ties)


def _collapse(p: float | None) -> float | None:
    if p is None:
        return None
    return 0.5 if float(p) in _QUARTER_TIES else float(p)


def _judge_pref(sample: dict) -> float | None:
    for key in ("judge_pref", "preference"):
        if sample.get(key) is not None:
            return _collapse(sample[key])
    return None


def _lang(sample: dict) -> str | None:
    md = sample.get("instruction_metadata")
    if isinstance(md, dict):
        return md.get("lang") or md.get("language") or sample.get("lang")
    return sample.get("lang")


def load_battles(paths: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    rows: list[dict] = []
    for p in paths:
        data = json.loads(Path(p).read_text())
        if isinstance(data, dict):
            # agreement output uses 'per_sample'; annotate output uses 'matches'
            if "per_sample" in data:
                samples = data["per_sample"]
            elif "matches" in data:
                samples = data["matches"]
            else:
                raise ValueError(
                    f"{p}: JSON object lacks a 'per_sample' or 'matches' key; "
                    "expected OpenJury format or a bare list of samples"
                )
        else:
            samples = data
        for s in samples:
            if s.get("parse_error"):
                continue
            ca, cb = s.get("completion_a"), s.get("completion_b")
            prompt = s.get("instruction") or s.get("prompt")
            yj = _judge_pref(s)
            if not ca or not cb or not prompt or yj is None:
                continue
            # OpenJury encodes preference as P(B preferred) (pref=0.0 => A wins).
            # Flip to the Bradley-Terry convention y = P(A preferred) that orient/
            # diagnose assume, so win-rates and outcome associations come out right.
            y_judge = 1.0 - yj
            rows.append({
                "instruction_id": str(s.get("instruction_id")),
                "model_a": s.get("model_a"),
                "model_b": s.get("model_b"),
                "prompt": prompt,
                "completion_a": ca,
                "completion_b": cb,
                "y_judge": y_judge,
                "judge_label": s.get("judge_label"),
                "scores_a": s.get("scores_a"),
                "scores_b": s.get("scores_b"),
                "len_a": s.get("len_a"),
                "len_b": s.get("len_b"),
                "lang": _lang(s),
                "source_path": str(p),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates("instruction_id").reset_index(drop=True)
    return df
