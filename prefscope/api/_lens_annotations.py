"""Internal annotation-table loading for the Lens facade."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prefscope.artifacts import (
    FEATURE_CALIBRATION,
    FEATURE_CLUSTERS,
    FEATURE_CONTEXT,
    FEATURE_FIDELITY,
    FEATURE_NAMES,
    FEATURE_ROLES,
    PROMPT_FEATURE_CLUSTERS,
    PROMPT_FEATURE_FIDELITY,
    PROMPT_FEATURE_NAMES,
)

_COMMON_ANNOTATIONS = (FEATURE_CALIBRATION, FEATURE_CONTEXT)
_RESPONSE_ANNOTATIONS = (
    FEATURE_NAMES, FEATURE_FIDELITY, FEATURE_ROLES, FEATURE_CLUSTERS,
    *_COMMON_ANNOTATIONS,
)
_PROMPT_ANNOTATIONS = (
    PROMPT_FEATURE_NAMES, PROMPT_FEATURE_FIDELITY, PROMPT_FEATURE_CLUSTERS,
    *_COMMON_ANNOTATIONS,
)

def _annotation_paths(lens_dir: Path, input_rep: str, annotations=None) -> list[Path]:
    """Canonical annotation files in the lens plus optional external files/dirs."""
    names = _PROMPT_ANNOTATIONS if input_rep == "prompt" else _RESPONSE_ANNOTATIONS
    roots = [lens_dir]
    explicit: list[Path] = []
    if annotations is not None:
        values = ([annotations] if isinstance(annotations, (str, Path))
                  else list(annotations))
        for value in values:
            path = Path(value)
            if path.is_dir():
                roots.append(path)
            elif path.is_file():
                if path.name not in names:
                    raise ValueError(
                        f"annotation file {path} has a noncanonical name; use one of "
                        f"{list(names)} or pass its containing directory")
                explicit.append(path)
            else:
                raise FileNotFoundError(f"annotation path does not exist: {path}")
    paths = [root / name for root in roots for name in names if (root / name).is_file()]
    paths.extend(explicit)
    # Preserve priority: bundled files first, explicit/external files later. A repeated
    # path should only be read once.
    return list(dict.fromkeys(path.resolve() for path in paths))


def _load_feature_table(lens_dir: Path, input_rep: str, m_total: int,
                        annotations=None) -> pd.DataFrame | None:
    """Merge names, fidelity, calibration, context and clusters by feature id.

    Later sources override earlier non-null values, so ``annotations=...`` can attach a
    fresh interpretation directory without copying it into a trained lens first.
    """
    paths = _annotation_paths(lens_dir, input_rep, annotations)
    if not paths:
        return None
    table = pd.DataFrame({"feature_id": np.arange(int(m_total), dtype=int)})
    for path in paths:
        frame = pd.read_csv(path)
        if "feature_id" not in frame.columns:
            raise ValueError(f"annotation file {path} has no feature_id column")
        frame = frame.copy()
        frame["feature_id"] = pd.to_numeric(frame["feature_id"], errors="raise").astype(int)
        frame = frame.drop_duplicates("feature_id", keep="last")
        invalid = frame.loc[
            (frame["feature_id"] < 0) | (frame["feature_id"] >= int(m_total)),
            "feature_id",
        ].tolist()
        if invalid:
            raise ValueError(
                f"annotation file {path} contains feature ids outside [0, {m_total}): "
                f"{invalid[:10]}"
            )
        indexed = frame.set_index("feature_id")
        table = table.set_index("feature_id")
        for col in indexed.columns:
            incoming = indexed[col].reindex(table.index)
            if col in table.columns:
                table[col] = incoming.combine_first(table[col])
            else:
                table[col] = incoming
        table = table.reset_index()
    return table
