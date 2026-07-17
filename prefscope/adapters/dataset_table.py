"""Tabular importer: a DataFrame, CSV, or parquet of battles -> PairItems.

The reference "bring your own data" adapter. Map your columns to the PairItem
fields; only ``prompt`` and ``a`` are required. ``pref`` is P(A preferred).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from prefscope.core import registry
from prefscope.core.dataset import Dataset
from prefscope.core.types import PairItem

_OPTIONAL = ("b", "pref", "model_a", "model_b", "id")


@registry.register("dataset", "table")
class CsvDataset(Dataset):
    def __init__(self, source, *, prompt: str, a: str, b: str | None = None,
                 pref: str | None = None, model_a: str | None = None,
                 model_b: str | None = None, id: str | None = None) -> None:
        if isinstance(source, pd.DataFrame):
            df = source.reset_index(drop=True)
        else:
            p = Path(source)
            df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        self._df = df
        self._cols = {"prompt": prompt, "a": a, "b": b, "pref": pref,
                      "model_a": model_a, "model_b": model_b, "id": id}
        missing = [c for c in (prompt, a) if c not in df.columns]
        missing += [self._cols[k] for k in _OPTIONAL
                    if self._cols[k] is not None and self._cols[k] not in df.columns]
        if missing:
            raise ValueError(f"columns not found in data: {missing}")

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self) -> Iterator[PairItem]:
        cols = self._cols
        # itertuples RENAMES non-identifier column names ("user prompt" -> "_0"), so
        # row._asdict()[col] KeyErrors on such columns. Index by integer POSITION resolved
        # from the true column name instead, and use plain tuples (name=None) (#7).
        pos = {name: self._df.columns.get_loc(name)
               for name in cols.values() if name is not None}
        for i, row in enumerate(self._df.itertuples(index=False, name=None)):

            def get(key):
                col = cols[key]
                if col is None:
                    return None
                v = row[pos[col]]
                return None if (v is None or (isinstance(v, float) and pd.isna(v))) else v

            pid = get("id")
            pref = get("pref")
            yield PairItem(
                id=str(pid) if pid is not None else str(i),
                x=row[pos[cols["prompt"]]],
                y_a=row[pos[cols["a"]]],
                y_b=get("b"),
                pref=None if pref is None else float(pref),
                model_a=get("model_a"),
                model_b=get("model_b"),
            )
