# PrefScope

PrefScope is a framework for analyzing post-training preference data by concept.
Reusable lenses turn prompts and model responses into feature activations. You can
inspect these features, compare responses, and study how features relate to preference.

Train a sparse autoencoder lens, load a pretrained SAELens checkpoint, or connect your
own backend. Each lens returns feature matrices with row and feature IDs:

```text
PairItem -> Lens.featurize(...) -> FeatureBatch / FeatureMatrix
```

Use these matrices to summarize feature activity, find high-activation examples, measure
feature overlap, and run your own analyses. Feature catalogs attach names and annotations;
reports save results as JSON and CSV; the Viewer displays exported feature data.

## Install

Clone the repository and install with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/kargibora/PrefScope.git
cd PrefScope
uv sync
source .venv/bin/activate  # macOS/Linux
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

Choose optional capabilities with extras:

| Extra | Capability |
|-------|------------|
| `saelens` | SAELens checkpoints |
| `cpu` | Native lens training and inference on CPU or Apple MPS |
| `arena` | Hugging Face dataset adapters |

For example, `uv sync --extra cpu --extra arena` installs native lens support and dataset
adapters. The default inference examples use SAELens: use `uv sync --extra saelens`,
adding `--extra arena` for the Hugging Face dataset example. Repeat all extras you want
to keep in later sync commands.

A plain `import prefscope` stays Torch-free.

## Featurize response pairs

```python
from prefscope import Lens, PairItem

lens = Lens.from_pretrained("owner/repository", subfolder="response-lens")
items = [
    PairItem(
        id="example-1",
        x="Explain the result.",
        y_a="First response",
        y_b="Second response",
    )
]
features = lens.featurize(items, views=("response_a", "response_b"))
z_a = features.matrix("z_a")
z_b = features.matrix("z_b")
```

Supported backends can provide prompt, response-A, response-B, or response-difference
views according to their declared capabilities. Every contrast is oriented A minus B.
An individual lens derives `z_diff = f(e_A) - f(e_B)`; a direct-difference lens computes
`z_diff = f(e_A - e_B)`.

You can also use precomputed representations or construct a lens from a custom
`LensBackend`. See the
[Python API](docs/reference/python-api.md) and
[representation guide](docs/explanation/representations.md).

## Summarize feature activity

Run numerical functions directly on a `FeatureMatrix`:

```python
from prefscope import activation_summary, top_activating_rows

summary = activation_summary(z_a)
top_rows = top_activating_rows(z_a, k=5)
```

Summaries retain feature IDs. Top-row results also include row IDs for joining to source
examples. Coactivation helpers measure feature overlap from boolean matrices using a
membership rule you choose. See
[Numerical analysis](docs/reference/python-api.md#numerical-analysis) for the full list.

## Attach feature names

`FeatureCatalog` stores feature names and source annotations separately from numerical
activations. Join them through `feature_id`:

```python
from prefscope import FeatureCatalog, feature_activation_table

catalog = FeatureCatalog.from_mapping({0: "brevity", 4: "step-by-step structure"})
table = feature_activation_table(z_a, catalog=catalog)
```

Catalog joins use feature IDs and check feature-space identity when available.

## Save analysis results

```python
from prefscope import Report

report = Report(
    title="Response feature summary",
    metrics={"n_pairs": len(items)},
    tables={"activity": summary, "top_rows": top_rows},
    metadata={"study": "example"},
)
report.save("results/report")
```

`Report` saves your metrics and metadata in `report.json`, with pandas tables in separate
CSV files. Use a new output directory for each report.

## Specialized recipes

`prefscope.recipes` contains preference statistics, outcome associations, context analysis,
feature graphs, clustering, and reporting tools. Import the modules directly or adapt them
for your study. See [Specialized analyses](docs/recipes/specialized-analysis.md).

## Visualization bridge

`prefscope.viewer_export.export_viewer_bundle(...)` packages feature batches, optional
catalogs, and analysis tables with a built `@prefscope/viewer` static site. Compute your
tables and maps in Python, then pass them to the exporter along with the Viewer build.
The resulting directory contains the site, `data/viewer-data.json`, and a hashed file
inventory.

## CLI

Use the CLI to prepare datasets, build and inspect lenses, and name and verify features.
Use Python functions for analysis.

```bash
prefscope --help
prefscope init-demo --out demo
```

See the [CLI reference](docs/reference/cli.md).

## Documentation

- [Documentation index](docs/index.md)
- [Architecture](docs/explanation/architecture.md)
- [Python API](docs/reference/python-api.md)
- [API stability](docs/reference/api-stability.md)
- [Lens directory format](docs/reference/lens-directory.md)
- [SAELens integration](docs/how-to/use-saelens.md)
- [Add a lens backend](docs/extending/add-a-lens-backend.md)
- [Reports](docs/reference/report-bundle.md)
- [Viewer data bridge](docs/reference/viewer-bundle.md)
