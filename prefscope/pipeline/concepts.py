"""Export every active prompt/response concept in a filterable long table."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from prefscope.analysis.presence import annotation_flag, concept_presence
from prefscope.data import pair_schema
from prefscope.pipeline.encode_dataset import _load_table, _nonempty


def _resolve_column(df: pd.DataFrame, column, *defaults):
    if column is not None:
        alias = pair_schema.ENCODE_ALIASES.get(column)
        if column not in df.columns and alias in df.columns:
            return alias
        return column
    return next((name for name in defaults if name in df.columns), None)


def _normalise_dtypes(frame: pd.DataFrame, feature_table: pd.DataFrame) -> pd.DataFrame:
    """Give every streamed chunk the same Arrow/CSV-friendly nullable dtypes."""
    frame = frame.copy()
    bool_columns = {"semantic_present", "concept_pole_matches_name"}
    numeric_columns = {
        "row_id", "rank", "feature_id", "activation", "abs_activation",
        "activation_threshold",
    }
    for col in feature_table.columns:
        non_null = feature_table[col].dropna()
        bool_like = (
            len(non_null) > 0
            and bool(non_null.map(lambda value: isinstance(value, (bool, np.bool_))).all())
        )
        if is_bool_dtype(feature_table[col].dtype) or bool_like:
            bool_columns.add(col)
        elif is_numeric_dtype(feature_table[col].dtype):
            numeric_columns.add(col)
    for col in frame.columns:
        if col in bool_columns:
            frame[col] = frame[col].astype("boolean")
        elif col in numeric_columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            frame[col] = frame[col].astype("string")
    return frame


class _ChunkWriter:
    def __init__(self, path, feature_table: pd.DataFrame) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.feature_table = feature_table
        self.kind = self.path.suffix.lower()
        if self.kind not in {".parquet", ".csv", ".jsonl"}:
            raise ValueError("concept output must end in .parquet, .csv, or .jsonl")
        self._writer = None
        self._wrote = False

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        frame = _normalise_dtypes(frame, self.feature_table)
        if self.kind == ".parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(frame, preserve_index=False)
            if self._writer is None:
                self._writer = pq.ParquetWriter(
                    self.path, table.schema, compression="zstd")
            self._writer.write_table(table)
        elif self.kind == ".csv":
            frame.to_csv(
                self.path, mode="a" if self._wrote else "w",
                header=not self._wrote, index=False)
        else:
            frame.to_json(
                self.path, orient="records", lines=True,
                mode="a" if self._wrote else "w")
        self._wrote = True

    def close(self, empty: pd.DataFrame) -> None:
        if self._writer is not None:
            self._writer.close()
        if self._wrote:
            return
        empty = _normalise_dtypes(empty, self.feature_table)
        if self.kind == ".parquet":
            empty.to_parquet(self.path, index=False)
        elif self.kind == ".csv":
            empty.to_csv(self.path, index=False)
        else:
            self.path.write_text("")


def export_concepts(
    lens,
    data,
    out,
    *,
    prompt_col: str = "prompt",
    response_col: str | None = "response",
    response2_col: str | None = None,
    batch_size: int = 128,
    active_only: bool = True,
    pole: str = "any",
    min_abs_activation: float = 0.0,
    top_k: int | None = None,
    fidelity_only: bool = False,
    semantic_presence_only: bool = False,
    include_text: bool = False,
    log=print,
) -> dict:
    """Encode an arbitrary table and stream its active concepts to long form.

    Prompt lenses emit one ``side="prompt"`` set per row. Individual response
    lenses emit ``side="a"`` and, when present, ``side="b"``. Every emitted row
    keeps the source ``row_id``, feature id, raw activation, within-item rank, and
    bundled name/fidelity/calibration/context columns.
    """
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if lens.input_rep not in {"prompt", "individual"}:
        raise ValueError(
            "concept export needs a prompt or individual lens; a difference lens "
            "cannot encode a lone prompt/response")

    frame = _load_table(Path(data))
    if prompt_col not in frame.columns:
        raise ValueError(f"data has no prompt column {prompt_col!r}")
    response_col = _resolve_column(
        frame, response_col, "response", pair_schema.RESPONSE_A)
    response2_col = _resolve_column(
        frame, response2_col, "response_2", pair_schema.RESPONSE_B)
    if lens.input_rep == "individual":
        if response_col is None or response_col not in frame.columns:
            raise ValueError(
                f"individual lens needs response column {response_col!r}")
    if response2_col is not None and response2_col not in frame.columns:
        raise ValueError(f"data has no second response column {response2_col!r}")

    source = frame.assign(_prefscope_row_id=np.arange(len(frame), dtype=int))
    feature_table = lens.feature_table
    if lens.activation_polarity == "signed" and pole == "any":
        log(
            "warning: this is a signed lens; negative rows are opposite-axis "
            "activations, so concept_pole_matches_name=False and semantic_present=False")
    writer = _ChunkWriter(out, feature_table)
    n_concepts = 0
    n_items = 0

    if lens.input_rep == "prompt":
        sides = [("prompt", None)]
    else:
        sides = [("a", response_col)]
        if response2_col is not None:
            sides.append(("b", response2_col))

    empty_codes = np.empty((0, int(lens.projector.m_total)), dtype=np.float32)
    empty = lens.concept_activations(
        empty_codes, row_ids=[], active_only=active_only, pole=pole,
        min_abs_activation=min_abs_activation, top_k=top_k,
        fidelity_only=fidelity_only,
        semantic_presence_only=semantic_presence_only)
    empty.insert(1, "side", pd.Series(dtype="string"))
    if include_text:
        empty["prompt"] = pd.Series(dtype="string")
        empty["completion"] = pd.Series(dtype="string")

    try:
        for start in range(0, len(source), int(batch_size)):
            batch = source.iloc[start:start + int(batch_size)]
            for side, completion_col in sides:
                keep = _nonempty(batch[prompt_col])
                if completion_col is not None:
                    keep &= _nonempty(batch[completion_col])
                selected = batch[keep]
                if selected.empty:
                    continue
                prompts = selected[prompt_col].astype(str).tolist()
                completions = (
                    selected[completion_col].astype(str).tolist()
                    if completion_col is not None else None)
                codes = lens.encode(prompts, completions)
                long = lens.concept_activations(
                    codes, row_ids=selected["_prefscope_row_id"].tolist(),
                    active_only=active_only, pole=pole,
                    min_abs_activation=min_abs_activation, top_k=top_k,
                    fidelity_only=fidelity_only,
                    semantic_presence_only=semantic_presence_only)
                if long.empty:
                    n_items += len(selected)
                    continue
                long.insert(1, "side", side)
                if include_text:
                    indexed = selected.set_index("_prefscope_row_id")
                    long["prompt"] = long["row_id"].map(indexed[prompt_col])
                    long["completion"] = (
                        long["row_id"].map(indexed[completion_col])
                        if completion_col is not None else pd.NA)
                writer.write(long)
                n_concepts += len(long)
                n_items += len(selected)
    finally:
        writer.close(empty)

    result = {
        "input_rows": int(len(source)),
        "encoded_items": int(n_items),
        "concept_rows": int(n_concepts),
        "lens_kind": lens.input_rep,
        "output": str(Path(out)),
        "active_only": bool(active_only),
        "pole": pole,
        "min_abs_activation": float(min_abs_activation),
        "top_k": int(top_k) if top_k is not None else None,
        "fidelity_only": bool(fidelity_only),
        "semantic_presence_only": bool(semantic_presence_only),
    }
    log(
        f"wrote {n_concepts} active concept rows from {n_items} encoded items "
        f"to {out}")
    return result


def export_concepts_from_codes(
    lens,
    codes_dir,
    out,
    *,
    presence_policy: str = "mixed",
    fidelity_only: bool = True,
    named_only: bool = True,
    top_k: int | None = None,
    include_text: bool = False,
    chunk_size: int = 4096,
    log=print,
) -> dict:
    """Export semantic concept presence from an existing encoded bundle.

    Unlike :func:`export_concepts`, this function does not embed the dataset again. It
    consumes ``z_prompt.npy`` or ``z_a.npy``/``z_b.npy`` plus ``meta.parquet`` written
    by :func:`prefscope.pipeline.encode_dataset.run_encode_dataset`. This is the
    efficient path used by the config-driven ``prefscope analyze`` workflow.
    """
    codes_dir = Path(codes_dir)
    out = Path(out)
    if top_k is not None and int(top_k) <= 0:
        raise ValueError("top_k must be positive when set")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    meta_path = codes_dir / "meta.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"encoded bundle is missing {meta_path}")
    meta = pd.read_parquet(meta_path)

    if lens.input_rep == "prompt":
        sides = [("prompt", codes_dir / "z_prompt.npy", None)]
    elif lens.input_rep == "individual":
        sides = [("a", codes_dir / "z_a.npy", pair_schema.RESPONSE_A)]
        if (codes_dir / "z_b.npy").exists():
            sides.append(("b", codes_dir / "z_b.npy", pair_schema.RESPONSE_B))
    else:
        raise ValueError(
            "concept export needs a prompt or individual lens; a difference lens "
            "only represents a response contrast")
    missing = [str(path) for _, path, _ in sides if not path.exists()]
    if missing:
        raise FileNotFoundError(f"encoded bundle is missing {missing}")

    features = lens.feature_table.drop_duplicates("feature_id", keep="last").copy()
    features["feature_id"] = pd.to_numeric(
        features["feature_id"], errors="raise").astype(int)
    keep = pd.Series(True, index=features.index)
    if named_only:
        if "concept" not in features.columns:
            raise ValueError("named_only needs bundled concept names")
        keep &= features["concept"].notna() & (
            features["concept"].astype(str).str.strip() != "")
    if fidelity_only:
        if "fidelity_pass" not in features.columns:
            raise ValueError("fidelity_only needs bundled fidelity annotations")
        keep &= features["fidelity_pass"].map(annotation_flag)
    selected = features.loc[keep].sort_values("feature_id").reset_index(drop=True)
    feature_ids = selected["feature_id"].astype(int).tolist()
    indexed = selected.set_index("feature_id", drop=False)

    base_columns = [
        "row_id", "side", "rank", "feature_id", "activation",
        "semantic_present", "presence_basis", "activation_threshold",
    ]
    annotation_columns = [
        column for column in selected.columns if column not in base_columns
    ]
    optional_columns = []
    if "battle_id" in meta.columns:
        optional_columns.append("battle_id")
    if include_text:
        optional_columns.extend(["prompt", "completion"])
    empty = pd.DataFrame(columns=[
        *base_columns,
        *[column for column in annotation_columns if column != "feature_id"],
        *optional_columns,
    ])
    writer = _ChunkWriter(out, selected)
    concept_rows = 0
    encoded_items = 0

    try:
        for side, path, completion_col in sides:
            codes = np.load(path, mmap_mode="r")
            if len(codes) != len(meta):
                raise ValueError(
                    f"{path.name} has {len(codes)} rows but metadata has {len(meta)}")
            for start in range(0, len(meta), int(chunk_size)):
                stop = min(start + int(chunk_size), len(meta))
                block = np.asarray(codes[start:stop])
                presence = concept_presence(
                    block, selected, feature_ids=feature_ids,
                    policy=presence_policy)
                rows = []
                for offset in range(stop - start):
                    present_pos = np.flatnonzero(presence.values[offset])
                    if not len(present_pos):
                        continue
                    values = block[offset, presence.feature_ids[present_pos]]
                    order = np.argsort(-values, kind="stable")
                    if top_k is not None:
                        order = order[:int(top_k)]
                    source_row = meta.iloc[start + offset]
                    row_id = source_row.get("row_id", start + offset)
                    for rank, position in enumerate(present_pos[order], start=1):
                        feature_id = int(presence.feature_ids[position])
                        annotation = indexed.loc[feature_id].to_dict()
                        row = {
                            "row_id": row_id,
                            "side": side,
                            "rank": rank,
                            "feature_id": feature_id,
                            "activation": float(block[offset, feature_id]),
                            "semantic_present": True,
                            "presence_basis": str(presence.basis[position]),
                            "activation_threshold": float(
                                presence.thresholds[position]),
                            **{
                                key: value for key, value in annotation.items()
                                if key != "feature_id"
                            },
                        }
                        if "battle_id" in meta.columns:
                            row["battle_id"] = source_row["battle_id"]
                        if include_text:
                            row["prompt"] = source_row.get(pair_schema.PROMPT, pd.NA)
                            row["completion"] = (
                                source_row.get(completion_col, pd.NA)
                                if completion_col is not None else pd.NA)
                        rows.append(row)
                if rows:
                    chunk = pd.DataFrame(rows, columns=empty.columns)
                    writer.write(chunk)
                    concept_rows += len(chunk)
                encoded_items += stop - start
    finally:
        writer.close(empty)

    result = {
        "input_rows": int(len(meta)),
        "encoded_items": int(encoded_items),
        "concept_rows": int(concept_rows),
        "lens_kind": lens.input_rep,
        "output": str(out),
        "presence_policy": presence_policy,
        "fidelity_only": bool(fidelity_only),
        "named_only": bool(named_only),
        "top_k": int(top_k) if top_k is not None else None,
        "include_text": bool(include_text),
    }
    log(
        f"wrote {concept_rows} concept rows from {encoded_items} encoded items "
        f"to {out}")
    return result
