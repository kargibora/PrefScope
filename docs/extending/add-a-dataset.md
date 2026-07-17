# Add a dataset adapter

A **dataset adapter** feeds your data into PrefScope as a stream of `PairItem`s —
PrefScope's analysis reads `PairItem` objects, not any fixed file format, so you
adapt your data once and every downstream stage reads it through the same interface.
PrefScope ships two
(`table` for tabular files/DataFrames, `openjury` for OpenJury annotations); this
guide shows how to add your own.

Read [the registry](the-registry.md) first if you haven't — it explains how
components are registered. This guide assumes you know that part.

> **Honesty note.** Unlike interpreters/verifiers/clusterers, the `dataset` registry
> is **not** name-selected by the live CLI or config build path — `build-lens` reads
> a corpus parquet or annotation JSON directly. A custom `Dataset` is used
> **programmatically**: instantiate it and hand it to `Lens.train(data, ...)`,
> `lens.encode_pairs(data)`, or `lens.encode_items(data)`. For tabular data the built-in
> `CsvDataset` (registered as `table`) usually suffices. See
> [`../how-to/bring-your-own-dataset.md`](../how-to/bring-your-own-dataset.md) for the
> end-to-end walkthrough.

## The contract

A dataset subclasses `Dataset` (`prefscope/core/dataset.py`) and yields
`PairItem`s (`prefscope/core/types.py`):

```python
class Dataset(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[PairItem]:
        ...

    def __len__(self) -> int:           # optional; default raises TypeError (unsized)
        ...
```

Each yielded `PairItem` is one comparison:

```python
PairItem(id: str, x: str, y_a: str, y_b: str | None = None, pref: float | None = None,
         model_a=None, model_b=None, meta={})
```

### `PairItem` fields

| field | type | meaning |
|-------|------|---------|
| `id` | `str` | any stable per-row id |
| `x` | `str` | the prompt |
| `y_a` | `str` | response A — by convention the model **under study** ("self") |
| `y_b` | `str \| None` | response B ("other"); `None` for a single response |
| `pref` | `float \| None` | **P(A preferred)** ∈ [0, 1] — `0.0` = B wins, `0.5` = tie, `1.0` = A wins |
| `model_a`, `model_b` | `str \| None` | which model produced each side (needed by `diagnose`) |
| `meta` | `dict` | free-form extras |

`__post_init__` raises if `pref` is outside `[0, 1]`, so map your label onto that
range. Single-response items (`y_b=None`) are accepted by `encode_items()` with an
individual lens; `encode_pairs()` rejects them because it returns A/B contrasts.
Preference diagnosis likewise requires paired data.

## A minimal dataset

This adapter reads a JSONL file where each line is `{"prompt", "self", "other",
"win"}` (`win` is 1 if self wins, 0 if other wins).

```python
import json
from typing import Iterator
from prefscope.core import registry
from prefscope.core.dataset import Dataset
from prefscope.core.types import PairItem


@registry.register("dataset", "jsonl-battles")
class JsonlBattles(Dataset):
    def __init__(self, path: str) -> None:
        self._rows = [json.loads(ln) for ln in open(path) if ln.strip()]

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[PairItem]:
        for i, r in enumerate(self._rows):
            yield PairItem(
                id=str(r.get("id", i)),
                x=r["prompt"],
                y_a=r["self"],                 # model under study
                y_b=r["other"],
                pref=float(r["win"]),          # 1.0 self wins, 0.0 other wins
            )
```

## Register and use it

The `@registry.register("dataset", "jsonl-battles")` decorator registers it — but
the decorator only runs if the module is **imported**. Add it to
`prefscope/adapters/__init__.py`, or import it before use.

Because the build path does not name-select datasets, you **use it
programmatically** — instantiate and pass to a loaded lens:

```python
from prefscope import Lens

data = JsonlBattles("my_battles.jsonl")          # any iterable of PairItem works
lens = Lens.load("lenses/mylens")
codes, meta = lens.encode_pairs(data)            # (N, M) self-minus-other codes + meta
diag = lens.diagnose(codes, meta)
```

For tabular data you usually don't need a custom class — the built-in `table`
adapter maps your columns onto `PairItem` fields:

```python
from prefscope.adapters.dataset_table import CsvDataset

data = CsvDataset("battles.parquet", prompt="question", a="resp_self",
                  b="resp_other", pref="p_self_wins", model_a="model")
codes, meta = lens.encode_pairs(data)
```

## Test it

```python
def test_jsonl_battles(tmp_path):
    import json
    from prefscope.core.types import PairItem
    p = tmp_path / "b.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"prompt": "q1", "self": "a", "other": "b", "win": 1},
        {"prompt": "q2", "self": "c", "other": "d", "win": 0},
    ]))
    items = list(JsonlBattles(str(p)))
    assert len(items) == 2
    assert all(isinstance(it, PairItem) for it in items)
    assert items[0].pref == 1.0 and items[1].pref == 0.0
```

See [`../how-to/bring-your-own-dataset.md`](../how-to/bring-your-own-dataset.md) for
the full walkthrough, and [`add-an-interpreter.md`](add-an-interpreter.md) /
[`add-a-clusterer.md`](add-a-clusterer.md) for the registry-selected sibling
components.
