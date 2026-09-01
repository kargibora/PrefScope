# Extract concepts from every prompt

Use this guide when you already have a prompt lens and a dataset.

## Before you start

The lens must have `input_rep: prompt` in `manifest.json`. A response lens describes
responses. A difference lens describes the difference between two responses. Neither is
a valid prompt-concept lens.

Your dataset needs a prompt column. A stable row ID is also useful because it lets you
join the result back to the source data.

## Command line

```bash
prefscope concepts \
  --lens /path/to/prompt-lens \
  --data dataset.parquet \
  --prompt-col prompt \
  --semantic-presence-only \
  --out prompt_concepts.parquet
```

For a Hub lens, use a source such as
`--lens hf://owner/repository/prompt-lens`. Use `--annotations PATH` when the concept
names and calibration tables are outside the lens directory.

The output is a long table. Each row is one detected concept for one source row. It
includes the source row ID, feature ID, concept name, activation, and calibration data.
A prompt with no detected concept has no row in this table.

`--semantic-presence-only` is important. It uses the calibrated threshold for each
feature. Raw nonzero activation is numerical activity, not proof that the named concept
is present.

## Python

```python
import pandas as pd
import prefscope

source = pd.read_parquet("dataset.parquet")
if "row_id" not in source:
    source["row_id"] = [f"row-{index}" for index in range(len(source))]
source["row_id"] = source["row_id"].astype(str)
if source["row_id"].duplicated().any():
    raise ValueError("row_id must be unique")

lens = prefscope.Lens.from_dir("/path/to/prompt-lens", device="cpu")
if lens.input_rep != "prompt":
    raise ValueError(f"Expected a prompt lens, found {lens.input_rep!r}")

codes = lens.encode(source["prompt"].tolist())
concepts = lens.concept_activations(
    codes,
    row_ids=source["row_id"],
    semantic_presence_only=True,
)
concepts.to_parquet("prompt_concepts.parquet", index=False)
```

`codes` is the numerical prompt-by-feature matrix. `concepts` is the readable long
table with names and thresholds.

To add a list of detected concepts to every source row:

```python
detected = (
    concepts.dropna(subset=["concept"])
    .groupby("row_id", sort=False)["concept"]
    .agg(list)
)

result = source.copy()
result["concepts"] = [detected.get(row_id, []) for row_id in result["row_id"]]
```

## If the result is empty

Check these points:

1. The lens is a prompt lens.
2. The lens contains concept names.
3. The lens contains passing semantic calibration from
   `prefscope interpret calibrate-presence`.
4. The prompt column is not empty.

For exploration only, omit `--semantic-presence-only` or call
`lens.concept_activations(codes)`. This returns numerical activations. Do not describe
all of them as semantic concept presence.

See the [Python API reference](../reference/python-api.md#feature_table-concept_activations-filterable-activations)
and [semantic presence explanation](../explanation/presence-and-context.md).
