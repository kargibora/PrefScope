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

## Recommended: one config for the complete inference workflow

If the prompt and response lenses are already published, start from
[`examples/workflows/analyze-published-lenses.yaml`](../../examples/workflows/analyze-published-lenses.yaml):

```bash
prefscope analyze --config analysis.yaml
```

The config combines the source adapter, column mapping, lens locations, concept-presence
policy, optional analyses, and viewer export. It accepts local files or Hugging Face
datasets. It does not retrain the SAE or rerun interpretation.

Override common values directly or use a dotted config key:

```bash
prefscope analyze --config analysis.yaml \
  --hf-dataset organization/new-data --out analysis/new-data \
  --set data.source.split=test \
  --set data.columns.response_a=answer \
  --set data.mode=single
```

The output contains the canonical dataset, reusable sparse-code bundles, long prompt
and response concept tables, prompt–response relationships, applicable paired or
preference analyses, and `viewer-data/`. `analysis_state.json` makes the same command
resumable; changed settings require another output directory or `--fresh`.

The lower-level commands below remain useful when you want to schedule stages
separately or replace one analysis component.

## Analyze ratings or several outcome attributes

Keep rating columns in the encoded bundle metadata, then use the generic outcome API or
CLI. It supports binary, bounded probability/preference, continuous, and multi-continuous
outcomes without forcing all tasks into a winner label:

```bash
prefscope associate-outcomes --encoded-dir codes \
  --outcome-col helpfulness --outcome-col correctness \
  --outcome-kind multi_continuous --out associations.csv
```

Continuous attributes z-score by default; bounded outcomes remain on their natural scale.
Missing values are handled per attribute. Repeated prompts are averaged within group before
association so each independent prompt receives equal weight. These are descriptive
associations for this dataset, not causal claims or universal quality judgments.

## Option A — built-in local and Hugging Face adapters

If your data is a CSV / parquet / JSON / DataFrame, `TableDataset` maps columns
to `PairItem` fields:

```python
from prefscope import TableDataset

data = TableDataset(
    "mine.csv",
    prompt="question", a="answer_a", b="answer_b",
    pref="preference_probability",          # already P(A preferred)
)
```

The Hugging Face adapter uses the same mapping. Mutable branches and tags are resolved to
an immutable commit SHA before loading. The prepared sidecar records requested and
resolved revisions plus an ordered `canonical_table_hash`, so resume and audit bind the
exact retained rows rather than only a dataset name:

```python
from prefscope import HuggingFaceDataset

data = HuggingFaceDataset(
    "organization/dataset",
    name="configuration", split="train",
    prompt="question", a="answer_a", b="answer_b",
    pref="winner", label_mode="winner",
    a_values=("model_a",), b_values=("model_b",), tie_values=("tie",),
)
```

OpenJury annotation JSON has a built-in adapter too:

```python
from prefscope.core import registry
import prefscope.adapters                  # registers the built-ins
data = registry.get("dataset", "openjury")("/path/to/annotations.json")
```

`label_mode="winner"` deliberately requires the A/B token mapping. PrefScope
does not guess whether `0`, `1`, `left`, or `right` means A; reversing that
mapping reverses every downstream preference conclusion.

For a chosen/rejected dataset, make the chosen answer A and set
`label_mode="a-wins"` without a label column. If `chosen` and `rejected` are chat
message lists, select their contents explicitly:

```python
data = HuggingFaceDataset(
    "organization/chat-preferences",
    prompt="chosen", a="chosen", b="rejected",
    prompt_role="user:first",
    a_role="assistant:last", b_role="assistant:last",
    label_mode="a-wins",
)
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

## Materialize once from the CLI

For repeatable command-line analyses, map the source once:

```bash
prefscope prepare-dataset \
  --hf-dataset organization/dataset --split train \
  --prompt-col question \
  --response-col answer_a --response-2-col answer_b \
  --label-col winner --label-mode winner \
  --a-wins-value model_a --b-wins-value model_b --tie-value tie \
  --out analysis/data.parquet
```

This writes canonical text columns and numeric
`human_pref = P(A preferred)`, plus `analysis/data.prefscope.json` containing
the resolved source and mapping. A local source uses `--data mine.parquet`
instead. Large Hub sources support a bounded `--streaming --limit N` sample.

For mappings you will reuse, put the source, columns, structured-message
selectors, and label convention in YAML:

```yaml
source:
  type: huggingface
  path: organization/chat-preferences
  split: train
columns:
  prompt: chosen
  response_a: chosen
  response_b: rejected
text:
  prompt_role: user:first
  response_a_role: assistant:last
  response_b_role: assistant:last
label:
  mode: a-wins
```

Then run:

```bash
prefscope prepare-dataset --spec dataset.yaml --out analysis/data.parquet
```

## Apply prompt and response lenses

The three supported data situations are:

| input | available outputs |
|---|---|
| `(prompt, response_a)` | prompt concepts, response-A concepts |
| `(prompt, response_a, response_b)` | both sides' concepts and A-minus-B concept contrasts |
| pair + `human_pref` | all above, raw win rates and concept–preference associations |

Apply a pretrained individual response lens and retain every nonzero activation:

```bash
prefscope concepts \
  --lens hf://owner/lenses/completion \
  --data analysis/data.parquet \
  --response-col completion_a --response-2-col completion_b \
  --out analysis/response_concepts.parquet
```

Apply the prompt lens to the same row ids:

```bash
prefscope concepts \
  --lens hf://owner/lenses/prompt \
  --data analysis/data.parquet \
  --out analysis/prompt_concepts.parquet
```

The long tables include the raw activation, concept name, feature id, rank, side,
and bundled verification/calibration fields. `top_k` is not imposed by default.
For reusable matrix calculations, also save all sparse vectors:

```bash
prefscope encode-dataset \
  --lens hf://owner/lenses/completion \
  --data analysis/data.parquet \
  --response-col completion_a --response-2-col completion_b \
  --label-col human_pref \
  --out analysis/codes --device cuda
```

The same command with the prompt lens writes the row-aligned prompt vectors:

```bash
prefscope encode-dataset \
  --lens hf://owner/lenses/prompt \
  --data analysis/data.parquet \
  --out analysis/prompt_codes --device cuda
```

Because both bundles retain the same stable item ids, preference-independent
prompt→response co-activation is then:

```bash
prefscope elicit \
  --completion-lens analysis/codes \
  --prompt-lens analysis/prompt_codes \
  --out analysis/prompt_response_edges.csv
```

For a labeled pair, compute overall A win rate and which response concepts are
associated with winning:

```bash
prefscope win-relevance \
  --encoded-dir analysis/codes --all-features \
  --out analysis/win_relevance.csv
```

This association describes the supplied human/judge label; it is not an
objective good/bad judgment. Unlabeled pairs still produce `z_a`, `z_b`, and
`z_diff`, but correctly skip win-relevance. In a chosen/rejected dataset where
the chosen answer is always A, the raw A win rate is trivially 100% and an
A-vs-B outcome regression is unidentified. `win_relevance.csv` still reports
`preferred_minus_rejected_mean`, `preferred_side_rate`, and an exact sign test
from winner-oriented contrasts; the length-controlled logistic fields remain
missing until both A-win and B-win rows exist.

If you mapped both model columns, the same command also writes
`win_relevance_models.csv` with each model's wins, losses, ties, and win rate.

## Using adapters from Python

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

Prompt→response co-activation also accepts this single-response layout. Build an
aligned prompt lens and run `prefscope elicit`; `z_b.npy` is optional because this
analysis does not use preference labels.

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

For the CLI, a `prepare-dataset` table feeds `build-lens` directly:

```bash
prefscope build-lens --corpus analysis/data.parquet \
    --input-rep individual --out lenses/mine --device cuda
```

Loading requires only `prompt` and `completion_a`. `battle_id`, `source`, and
`language` are synthesized when absent, `model_a`/`model_b` are optional, and
`completion_b` is what marks a row as paired. `build-corpus` produces the fuller
`battle_id · source · language · prompt · model_a · model_b · completion_a ·
completion_b` (+ optional `human_pref`) schema from public arenas, which is required
only by the analyses that compare two responses or read preferences. From there,
[build and analyze a lens](build-and-analyze-a-lens.md) applies unchanged.
