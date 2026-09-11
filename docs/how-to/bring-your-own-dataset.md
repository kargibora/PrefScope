# Bring your own dataset

PrefScope accepts local tables and optional Hugging Face datasets through dataset adapters.
Map source columns explicitly instead of relying on name guessing.

```python
from prefscope import TableDataset

dataset = TableDataset(
    "data/comparisons.parquet",
    id="example_id",
    prompt="instruction",
    a="candidate_a",
    b="candidate_b",
    pref="probability_a_preferred",
)
items = list(dataset)
```

Preference values mean `P(A preferred)`. A chosen/rejected layout must preserve which text
became A. Do not silently reverse the pair.

Feature the normalized items with any supported lens:

```python
batch = lens.featurize(items, views=("response_a", "response_b"))
```

Retain scalar metadata only when the downstream study needs it. Keep private or sensitive
fields out of artifacts you intend to share; PrefScope does not classify privacy for
arbitrary columns.

For command-line preparation, use:

```bash
prefscope prepare-dataset --help
```

Analysis is a direct Python step over `FeatureMatrix`, not a YAML-dispatched workflow.
