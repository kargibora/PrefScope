# PrefScope

This branch prepares the breaking `0.3.0` alpha API. See
[Migrate from 0.2](docs/how-to/migrate-from-0.2.md) before upgrading an existing study.
It has not yet been published to PyPI.

PrefScope turns paired model responses into feature activations through a reusable lens.
Its supported flow is deliberately small:

```text
PairItem -> Lens.featurize(...) -> FeatureBatch / FeatureMatrix
```

The package also provides a few numerical helpers, explicit feature annotations, and a
small `Report` container. It does not decide what features mean, run an automatic
analysis plan, or turn activity into scientific claims.

> PrefScope is a research tool. Its outputs are not evidence of model quality, reward,
> causality, safety, or generator behavior without a separate study design.

## Install

For this development version, install from a source checkout:

```bash
git clone https://github.com/kargibora/PrefScope.git
cd PrefScope
uv sync
source .venv/bin/activate  # macOS/Linux
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.
`pip install prefscope` installs the latest published version, which may have a different
API.

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

## Run small numerical operations

The supported analysis surface consists of ordinary functions over `FeatureMatrix`:

```python
from prefscope import activation_summary, coactivation_pairs, top_activating_rows

summary = activation_summary(z_a)
top_rows = top_activating_rows(z_a, k=5)

from prefscope import FeatureMatrix

presence = FeatureMatrix(
    values=caller_defined_presence,
    row_ids=z_a.row_ids,
    feature_ids=z_a.feature_ids,
    role="presence",
    orientation="absolute",
    activation_polarity="boolean",
    code_semantics="semantic_presence",
    provenance={"presence_basis": ["defined by the caller"]},
)
overlaps = coactivation_pairs(presence, min_count=2)
```

These functions preserve `feature_id` and `row_id`. They do not name features, select a
statistical protocol, infer causality, or define semantic presence from raw activity.
The full supported list is in [Numerical analysis](docs/reference/python-api.md#numerical-analysis).

## Keep annotations separate

`FeatureCatalog` stores display annotations. Activations and annotations join explicitly
through `feature_id`:

```python
from prefscope import FeatureCatalog, feature_activation_table

catalog = FeatureCatalog.from_mapping({0: "brevity", 4: "step-by-step structure"})
table = feature_activation_table(z_a, catalog=catalog)
```

A catalog label is a proposed annotation. It is not a verified scientific conclusion.
Feature-space identity prevents accidental joins between different coordinate systems.

## Collect caller-owned results

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

`Report` stores finite numerical metrics, pandas tables, and JSON metadata. It does not
compile analyses, filter private data, render narrative, or interpret columns. The caller
owns those choices.

## Specialized recipes

Specialized preference, outcome, context, graph, clustering, and reporting code is kept
under `prefscope.recipes`. Recipes are direct Python modules that can be copied or adapted.
They are not exported from `prefscope`, registered as analysis components, or run
automatically. Their APIs may change between releases.

## Visualization bridge

`prefscope.viewer_export.export_viewer_bundle(...)` packages already-computed
`FeatureBatch`, optional `FeatureCatalog`, and caller-provided tables with an explicit,
already-built `@prefscope/viewer` static site. It copies that build unchanged, adds
`data/viewer-data.json`, and writes a hashed file inventory. It does not build the Viewer
or calculate maps, distributions, examples, coactivation, or report semantics.

## CLI

The CLI covers dataset preparation, lens construction, feature extraction, naming, and
interpretation. Analysis and report orchestration commands were intentionally removed.
Use direct Python functions for analysis.

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
