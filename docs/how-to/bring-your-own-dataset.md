# Bring your own dataset

PrefScope's analysis reads `PairItem` objects, not a fixed file format. You adapt
your data to `PairItem` once; this guide shows three ways (a tabular adapter, the
OpenJury adapter, and a custom `Dataset`), then how to feed it to a lens.

## The contract: `PairItem`

```python
from prefscope.core import PairItem

PairItem(
    id="row-1",                 # any stable id
    x="the prompt",             # the question
    y_a="response A",           # by convention the model under study ("self")
    y_b="response B",           # the comparison ("other"); None for single-response data
    pref=0.8,                   # P(A preferred): 1.0 = A wins, 0.0 = B wins, 0.5 = tie
    model_a="my-model",         # which model produced y_a (needed by diagnose)
    model_b="baseline",         # which model produced y_b
)
```

`pref` is only needed for preference-grounded steps (diagnosis and win-relevance);
feature-name verification uses held-out text and does not require preference labels.

## Option A — the built-in tabular adapter

If your data is a CSV / parquet / DataFrame, `CsvDataset` maps columns to
`PairItem` fields, no code:

```python
from prefscope.adapters.dataset_table import CsvDataset

data = CsvDataset("mine.csv",
                  prompt="question", a="answer_a", b="answer_b",
                  pref="human_choice")     # column names in YOUR file
```

OpenJury annotation JSON has a built-in adapter too:

```python
from prefscope.core import registry
import prefscope.adapters                  # registers the built-ins
data = registry.get("dataset", "openjury")("/path/to/annotations.json")
```

## Option B — a custom `Dataset`

For anything else (a database, an API, a bespoke JSON), implement
`Dataset.__iter__` to yield `PairItem`s:

```python
from prefscope.core.dataset import Dataset
from prefscope.core import PairItem

class MyDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __iter__(self):
        for r in self.rows:
            yield PairItem(id=r["uid"], x=r["prompt"],
                           y_a=r["winner_text"], y_b=r["loser_text"],
                           pref=1.0, model_a=r["winner"], model_b=r["loser"])
```

## Using it

Feed any of these straight into a trained lens for inference (see
[diagnose a model](diagnose-a-model.md)):

```python
from prefscope import Lens
lens = Lens.load("lenses/mylens")
codes, meta = lens.encode_pairs(data)      # accepts any iterable of paired PairItems
diag = lens.diagnose(codes, meta)
```

For homogeneous single-response data (`y_b=None`), use an individual lens and the
mode-aware entry point:

```python
codes, meta = lens.encode_items(data)      # absolute response codes, not A/B contrasts
```

You can name, verify, cluster, and browse those response concepts. Diagnosis,
win-relevance, and preference prediction remain pairwise by definition.

## Training a lens on your own data

**Training** a lens (the embed + SAE step) accepts the same `Dataset` object through
Python, or a corpus parquet through the CLI:

```python
from prefscope import Lens, TrainConfig
lens = Lens.train(data, config=TrainConfig(device="cuda"), out="lenses/mine")
```

`Lens.train` accepts either all paired rows or all single-response rows. Single rows
require `SAEConfig(input_rep="individual")` (the default); mixed rows are rejected so
one artifact never combines absolute response codes with pair contrasts.

For the CLI, convert the data to the corpus schema once:

`battle_id · source · language · prompt · model_a · model_b · completion_a ·
completion_b` (+ optional `human_pref`)

then:

```bash
prefscope build-lens --corpus my_corpus.parquet \
    --input-rep individual --out lenses/mine --device cuda
```

`build-corpus` produces this schema from public arenas; for a custom source,
write the eight columns to a parquet yourself (a tiny pandas script) and point
`build-lens` at it. From there, [build and analyze a lens](build-and-analyze-a-lens.md)
applies unchanged.
