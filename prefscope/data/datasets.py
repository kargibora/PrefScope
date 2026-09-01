"""Public local and Hugging Face ``PairItem`` dataset adapters.

The implementations live outside ``prefscope.adapters`` so importing the lightweight
public API does not trigger registration of torch-backed SAE components.
"""
from __future__ import annotations

import os
from typing import Iterator

import pandas as pd

from prefscope.core import registry
from prefscope.core.dataset import Dataset
from prefscope.core.types import PairItem
from prefscope.data.tabular import (
    ColumnMapping, canonicalize_table, load_hf_table, load_local_table,
)


@registry.register("dataset", "table")
class CsvDataset(Dataset):
    """Map a DataFrame or local table into ``PairItem`` objects."""

    def __init__(
        self,
        source,
        *,
        prompt: str,
        a: str,
        b: str | None = None,
        pref: str | None = None,
        model_a: str | None = None,
        model_b: str | None = None,
        id: str | None = None,
        group_id: str | None = None,
        metadata=(),
        prompt_role: str | None = None,
        a_role: str | None = None,
        b_role: str | None = None,
        label_mode: str | None = None,
        a_values=(),
        b_values=(),
        tie_values=(),
    ) -> None:
        if isinstance(source, pd.DataFrame):
            frame = source.reset_index(drop=True)
        else:
            frame = load_local_table(source)
        # Backward-compatible default: the historical table adapter accepted a numeric
        # preference column and interpreted it directly as P(A preferred).
        if pref is not None and label_mode is None:
            label_mode = "probability"
        mapping = ColumnMapping(
            prompt=prompt,
            response_a=a,
            response_b=b,
            label=pref,
            model_a=model_a,
            model_b=model_b,
            item_id=id,
            group_id=group_id,
            metadata=tuple(metadata),
            prompt_role=prompt_role,
            response_a_role=a_role,
            response_b_role=b_role,
            label_mode=label_mode,
            a_values=tuple(a_values),
            b_values=tuple(b_values),
            tie_values=tuple(tie_values),
            auto_pair=b is not None,
        )
        self._df, self.summary = canonicalize_table(frame, mapping)
        self._metadata_columns = tuple(metadata)

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self) -> Iterator[PairItem]:
        for row in self._df.to_dict("records"):
            pref = row.get("human_pref")
            pref = None if pd.isna(pref) else float(pref)
            yield PairItem(
                id=str(row["item_id"]),
                x=str(row["prompt"]),
                y_a=str(row["completion_a"]),
                y_b=(
                    str(row["completion_b"])
                    if "completion_b" in row and not pd.isna(row["completion_b"])
                    else None
                ),
                pref=pref,
                model_a=(
                    str(row["model_a"])
                    if "model_a" in row and not pd.isna(row["model_a"]) else None),
                model_b=(
                    str(row["model_b"])
                    if "model_b" in row and not pd.isna(row["model_b"]) else None),
                meta={
                    "source_row_id": int(row["row_id"]),
                    **(
                        {"group_id": str(row["group_id"])}
                        if "group_id" in row and not pd.isna(row["group_id"])
                        else {}
                    ),
                    **{
                        name: row[name]
                        for name in self._metadata_columns if name in row
                    },
                },
            )


# More accurate public spelling; registry name and historical class stay compatible.
TableDataset = CsvDataset


@registry.register("dataset", "huggingface")
class HuggingFaceDataset(CsvDataset):
    """Load one Hub split and expose it as ``PairItem`` objects."""

    def __init__(
        self,
        dataset_id: str,
        *,
        prompt: str,
        a: str,
        b: str | None = None,
        pref: str | None = None,
        model_a: str | None = None,
        model_b: str | None = None,
        id: str | None = None,
        group_id: str | None = None,
        metadata=(),
        prompt_role: str | None = None,
        a_role: str | None = None,
        b_role: str | None = None,
        label_mode: str | None = None,
        a_values=(),
        b_values=(),
        tie_values=(),
        name: str | None = None,
        split: str = "train",
        revision: str | None = None,
        token=None,
        token_env: str | None = None,
        streaming: bool = False,
        limit: int | None = None,
    ) -> None:
        if token is not None and token_env is not None:
            raise ValueError("pass token or token_env, not both")
        if token_env is not None:
            token = os.environ.get(token_env)
        frame = load_hf_table(
            dataset_id,
            name=name,
            split=split,
            revision=revision,
            token=token,
            streaming=streaming,
            limit=limit,
        )
        super().__init__(
            frame,
            prompt=prompt,
            a=a,
            b=b,
            pref=pref,
            model_a=model_a,
            model_b=model_b,
            id=id,
            group_id=group_id,
            metadata=metadata,
            prompt_role=prompt_role,
            a_role=a_role,
            b_role=b_role,
            label_mode=label_mode,
            a_values=a_values,
            b_values=b_values,
            tie_values=tie_values,
        )
        self.dataset_id = dataset_id
        self.split = split
