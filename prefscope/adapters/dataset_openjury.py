"""Dataset adapter over OpenJury annotation JSON (wraps data/ingest.load_battles)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from prefscope.core import registry
from prefscope.core.dataset import Dataset
from prefscope.core.types import PairItem
from prefscope.data.ingest import load_battles

_META_COLS = ("scores_a", "scores_b", "len_a", "len_b", "lang")


@registry.register("dataset", "openjury")
class OpenJuryDataset(Dataset):
    def __init__(self, paths: str | Path | Iterable[str | Path]) -> None:
        self._df: pd.DataFrame = load_battles(paths)

    def __iter__(self) -> Iterator[PairItem]:
        for row in self._df.itertuples(index=False):
            d = row._asdict()
            meta = {c: d.get(c) for c in _META_COLS if c in d}
            pref = d.get("y_judge")
            yield PairItem(
                id=str(d["instruction_id"]),
                x=d["prompt"],
                y_a=d["completion_a"],
                y_b=d.get("completion_b"),
                pref=None if pref is None else float(pref),
                model_a=d.get("model_a"),
                model_b=d.get("model_b"),
                meta=meta,
            )

    def __len__(self) -> int:
        return len(self._df)
