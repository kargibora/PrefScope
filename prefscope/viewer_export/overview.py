"""Dataset-shape artifacts: concept prevalence and concept co-activation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prefscope.analysis.presence import annotation_flag
from prefscope.analysis.distribution import concept_coactivation, concept_distribution
from prefscope.artifacts import BATTLES, Z_A, Z_DIFF, Z_PROMPT

_CODE_FILES = (Z_A, Z_DIFF, Z_PROMPT)
GROUP_COLUMNS = ("language", "lang", "source", "model_a")
TEXT_GROUP_COLUMNS = ("language", "lang", "source")


def _text_group(record) -> dict:
    for column in TEXT_GROUP_COLUMNS:
        value = record.get(column)
        if pd.notna(value) and str(value).strip():
            return {"group": str(value), "group_column": column}
    return {}


class _StackedRows:
    """A sliceable, bounded-memory vertical view over same-width code arrays."""

    ndim = 2

    def __init__(self, arrays):
        self.arrays = list(arrays)
        widths = {int(array.shape[1]) for array in self.arrays}
        if not self.arrays or len(widths) != 1:
            raise ValueError("stacked code arrays must be non-empty and have one width")
        self.shape = (sum(int(array.shape[0]) for array in self.arrays), widths.pop())

    def __getitem__(self, key):
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("stacked code view supports contiguous row slices")
        start, stop, _ = key.indices(self.shape[0])
        chunks = []
        offset = 0
        for array in self.arrays:
            end = offset + int(array.shape[0])
            lo, hi = max(start, offset), min(stop, end)
            if lo < hi:
                chunks.append(np.asarray(array[lo - offset:hi - offset]))
            offset = end
            if offset >= stop:
                break
        if not chunks:
            return np.empty((0, self.shape[1]), dtype=np.float32)
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


def _visible_features(features: pd.DataFrame | None, width: int):
    """Feature columns safe to describe as concepts, preferring verified names."""
    if features is None or "feature_id" not in features.columns:
        return np.arange(width, dtype=int), "all"
    table = features.copy()
    table["feature_id"] = pd.to_numeric(table["feature_id"], errors="coerce")
    table = table.dropna(subset=["feature_id"])
    if "concept" in table.columns:
        table = table[table["concept"].fillna("").astype(str).str.strip().ne("")]
    selection = "named"
    if "fidelity_pass" in table.columns and table["fidelity_pass"].notna().any():
        table = table[table["fidelity_pass"].map(annotation_flag)]
        selection = "verified"
    ids = table["feature_id"].astype(int).drop_duplicates().to_numpy()
    ids = ids[(ids >= 0) & (ids < width)]
    return ids, selection


def _codes(lens: Path):
    if (lens / Z_A).exists():
        sides = [np.load(lens / Z_A, mmap_mode="r")]
        source = Z_A
        if (lens / "z_b.npy").exists():
            sides.append(np.load(lens / "z_b.npy", mmap_mode="r"))
            source = f"{Z_A}+z_b.npy"
        return (sides[0] if len(sides) == 1 else _StackedRows(sides)), source
    for name in _CODE_FILES:
        if (lens / name).exists():
            return np.load(lens / name, mmap_mode="r"), name
    return None, None


def _groups(lens: Path, n_rows: int):
    path = lens / BATTLES
    if not path.exists():
        return None, None
    meta = pd.read_parquet(path)
    paired = n_rows == 2 * len(meta)
    if len(meta) != n_rows and not paired:
        return None, None
    for column in GROUP_COLUMNS:
        if column in meta.columns and meta[column].astype(str).str.strip().ne("").any():
            if meta[column].nunique(dropna=True) > 1:
                first = meta[column].astype(str).to_numpy()
                if paired:
                    if column == "model_a" and "model_b" in meta:
                        second = meta["model_b"].astype(str).to_numpy()
                    else:
                        second = first
                    first = np.concatenate([first, second])
                return first, column
    return None, None


def export_concept_distribution(lens, features: pd.DataFrame, *, chunk_rows: int = 50_000):
    """Per-concept prevalence, per-row concept counts, and per-group fire rates."""
    lens = Path(lens)
    codes, source = _codes(lens)
    if codes is None:
        return None
    groups, group_column = _groups(lens, codes.shape[0])
    ids, selection = _visible_features(features, codes.shape[1])
    payload = concept_distribution(
        codes, columns=ids, feature_ids=ids, groups=groups, chunk_rows=chunk_rows)
    payload["code_array"] = source
    payload["group_column"] = group_column
    payload["selection"] = selection
    payload["n_total_features"] = int(codes.shape[1])
    names = {}
    if features is not None and {"feature_id", "concept"} <= set(features.columns):
        names = {int(r.feature_id): ("" if pd.isna(r.concept) else str(r.concept))
                 for r in features.itertuples()}
    for row in payload["features"]:
        row["concept"] = names.get(row["feature_id"], "")
    return payload


def export_prompt_concept_distribution(prompt_lens, features: pd.DataFrame, *,
                                       chunk_rows: int = 50_000):
    """Per-concept prevalence and per-prompt concept counts for a prompt lens.

    This deliberately requires ``z_prompt.npy`` instead of using the generic code-array
    fallback.  A directory that also contains response codes must never yield a plausible
    but semantically wrong prompt distribution.
    """
    lens = Path(prompt_lens)
    path = lens / Z_PROMPT
    if not path.exists():
        return None
    codes = np.load(path, mmap_mode="r")
    groups, group_column = _groups(lens, codes.shape[0])
    ids, selection = _visible_features(features, codes.shape[1])
    payload = concept_distribution(
        codes, columns=ids, feature_ids=ids, groups=groups, chunk_rows=chunk_rows)
    payload["code_array"] = Z_PROMPT
    payload["group_column"] = group_column
    payload["selection"] = selection
    payload["n_total_features"] = int(codes.shape[1])
    names = {}
    if features is not None and {"feature_id", "concept"} <= set(features.columns):
        names = {int(r.feature_id): ("" if pd.isna(r.concept) else str(r.concept))
                 for r in features.itertuples()}
    for row in payload["features"]:
        row["concept"] = names.get(row["feature_id"], "")
    return payload


def _example_texts(lens: Path, corpus_path, rows: set[int]) -> dict[int, dict]:
    """Prompt/response snippets for the given row indices, in lens row order."""
    if not rows or not corpus_path:
        return {}
    from prefscope.interpret.io import load_lens_battles
    battles, _, _ = load_lens_battles(lens, corpus=corpus_path)
    out = {}
    n = len(battles)
    paired = (lens / "z_b.npy").exists()
    for row in sorted(rows):
        side_b = paired and row >= n
        local = row - n if side_b else row
        if local >= n:
            continue
        record = battles.iloc[int(local)]
        out[int(row)] = {
            "prompt": _clip(record.get("prompt", ""), 300),
            "response": _clip(record.get("completion_b" if side_b else "completion_a", ""), 600),
            **_text_group(record),
        }
    return out


def _prompt_example_texts(lens: Path, corpus_path, rows: set[int]) -> dict[int, dict]:
    """Prompt snippets for prompt-lens row indices, aligned by instruction id."""
    if not rows or not corpus_path:
        return {}
    from prefscope.data.corpus import load_corpus

    meta = pd.read_parquet(lens / BATTLES)
    corpus = load_corpus(corpus_path)
    meta_ids = meta["instruction_id"].astype(str).tolist()
    corpus["instruction_id"] = corpus["instruction_id"].astype(str)
    indexed = corpus.set_index("instruction_id")
    missing = [instruction_id for instruction_id in meta_ids
               if instruction_id not in indexed.index]
    if missing:
        raise ValueError(
            f"{len(missing)} prompt-lens rows missing from corpus "
            f"(e.g. {missing[:3]})"
        )
    aligned = indexed.loc[meta_ids].reset_index()
    out = {}
    for row in sorted(rows):
        if row < 0 or row >= len(aligned):
            continue
        record = aligned.iloc[int(row)]
        out[int(row)] = {"prompt": _clip(record.get("prompt", ""), 600),
                         **_text_group(record)}
    return out


def _attach_example_activations(payload: dict, codes) -> None:
    """Attach the two positive activation values to every retained text example.

    Text is keyed globally by row because the same response may support multiple pairs.
    The sparse activation map lets every pair retrieve its own two values without
    duplicating the transcript in the bundle.
    """
    examples = payload.get("examples", {})
    if not examples:
        return
    row_features: dict[int, set[int]] = {}
    for pair in payload.get("pairs", []):
        for row in pair.get("rows", []):
            row_features.setdefault(int(row), set()).update(
                (int(pair["a"]), int(pair["b"]))
            )
    for row, feature_ids in row_features.items():
        example = examples.get(str(row))
        if example is None or row < 0 or row >= codes.shape[0]:
            continue
        ordered = sorted(feature_ids)
        values = np.asarray(codes[row:row + 1])[0, ordered]
        example["activations"] = {
            str(feature_id): round(float(value), 6)
            for feature_id, value in zip(ordered, values)
        }


def _clip(value, limit: int) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


def export_coactivation(lens, features: pd.DataFrame, *, top_k: int = 20,
                        max_pairs: int = 20_000, n_examples: int = 6,
                        corpus_path: str = "", chunk_rows: int = 50_000):
    """Concept pairs that co-fire more than independence predicts, with example rows."""
    lens = Path(lens)
    codes, source = _codes(lens)
    if codes is None:
        return None
    # Co-activation is an activation-level statistic, so retain every SAE axis. Named
    # and verified status remain visible in features.json; dropping unverified axes here
    # would make atlas points appear to have no neighbors merely because interpretation
    # is incomplete.
    ids = np.arange(codes.shape[1], dtype=int)
    selection = "all_axes"
    payload = concept_coactivation(
        codes, columns=ids, feature_ids=ids, top_k=top_k, max_pairs=max_pairs,
        n_examples=n_examples, chunk_rows=chunk_rows)
    payload["code_array"] = source
    payload["selection"] = selection
    payload["n_total_features"] = int(codes.shape[1])
    wanted = {row for pair in payload["pairs"] for row in pair["rows"]}
    payload["examples"] = {
        str(row): text for row, text in _example_texts(lens, corpus_path, wanted).items()
    }
    _attach_example_activations(payload, codes)
    if features is not None and {"feature_id", "concept"} <= set(features.columns):
        names = {int(r.feature_id): ("" if pd.isna(r.concept) else str(r.concept))
                 for r in features.itertuples()}
        for pair in payload["pairs"]:
            pair["a_concept"] = names.get(pair["a"], "")
            pair["b_concept"] = names.get(pair["b"], "")
    return payload


def export_prompt_coactivation(prompt_lens, features: pd.DataFrame, *,
                               top_k: int = 20, max_pairs: int = 20_000,
                               n_examples: int = 6,
                               corpus_path: str = "",
                               chunk_rows: int = 50_000):
    """Prompt-axis pairs that co-fire above independence in the prompt corpus."""
    lens = Path(prompt_lens)
    path = lens / Z_PROMPT
    if not path.exists():
        return None
    codes = np.load(path, mmap_mode="r")
    ids = np.arange(codes.shape[1], dtype=int)
    payload = concept_coactivation(
        codes, columns=ids, feature_ids=ids, top_k=top_k, max_pairs=max_pairs,
        n_examples=n_examples, chunk_rows=chunk_rows)
    payload["code_array"] = Z_PROMPT
    payload["selection"] = "all_axes"
    payload["n_total_features"] = int(codes.shape[1])
    wanted = {row for pair in payload["pairs"] for row in pair["rows"]}
    payload["examples"] = {
        str(row): text
        for row, text in _prompt_example_texts(lens, corpus_path, wanted).items()
    }
    _attach_example_activations(payload, codes)
    if features is not None and {"feature_id", "concept"} <= set(features.columns):
        names = {int(r.feature_id): ("" if pd.isna(r.concept) else str(r.concept))
                 for r in features.itertuples()}
        for pair in payload["pairs"]:
            pair["a_concept"] = names.get(pair["a"], "")
            pair["b_concept"] = names.get(pair["b"], "")
    return payload
