# Reporting examples

For caller-owned classification of existing prompt concepts into subject, intent,
audience, constraint, language and other, see the experimental
[prompt taxonomy recipe](prompt_taxonomy.md). It supports offline dry runs and
resumable OpenRouter pilots; it does not change the core catalog or train a lens.

`compile_report.py` saves caller-supplied metrics and tables as a report. To package
already-computed feature batches for the shared static Viewer instead, use:

```python
from prefscope import load_feature_batch, load_feature_catalog
from prefscope.viewer_export import export_viewer_bundle

features = load_feature_batch("artifacts/features")
prompt_features = load_feature_batch("artifacts/prompt_features")
prompt_catalog = load_feature_catalog("artifacts/prompt_feature_catalog.json")

export_viewer_bundle(
    features,
    "example-output/reporting/viewer-site",
    viewer_dist="path/to/viewer/dist",
    prompt_features=prompt_features,
    prompt_catalog=prompt_catalog,
)
```

Prompt inputs are optional. Prompt rows must exactly match response rows in order.
The prompt catalog belongs only to the prompt feature space. The destination must not
exist. Supply a dataset-free Viewer build that supports `prefscope.viewer_data` v2;
export does not install or build the Viewer and does not compute analyses.

See [the Viewer bundle reference](../../docs/reference/viewer-bundle.md) for the CLI,
metadata keys, and caller-computed `feature_map` and `prompt_feature_map` tables.

## Example activation UMAPs

`example_umap.py` computes caller-owned tables before the normal export. It needs
optional `umap-learn` (tested with version `0.5.12`); the base PrefScope install and
serializer do not require or install it.

```python
from examples.reporting.example_umap import example_umap_tables

export_viewer_bundle(
    features,
    "example-output/reporting/viewer-umap",
    viewer_dist="path/to/viewer/dist",
    prompt_features=prompt_features,
    tables=example_umap_tables(features, prompt_features, seed=42),
)
```

Prompt vectors get a separate UMAP. Real A/B answer vectors in the same response
feature space are stacked as separate observations in **one shared answer UMAP**,
not averaged and not fit in separate coordinate frames. A normal individual-lens
batch contains `z_a`, `z_b`, and `z_diff`. When both response sides are present,
the helper selects response-role views (`z_a` and `z_b` in that batch) and excludes
the derived difference from the answer fit. The original batch remains unchanged
for export. A difference-only batch gets one point per original raw pair code,
without preference orientation. Single
response and prompt-only batches also work. All feature columns, rows, signs, and
zero vectors are retained. No labels, preference scores, sampling, normalization,
SVD fallback, or artificial coordinate offsets enter the fit.

The root table `example_umap_points` contains `projection_id` (`prompt`, `answers`,
or `pair`), `space` (`main` or `prompt`), exact `view`, canonical `row_id`, `x`, `y`,
and `zero_vector`. Identity is `(projection_id, row_id, view)`, so two answers
from one row remain separate points. Join text and supplied preference by canonical
row ID, not point-table position. Changing concept color does not change geometry.

`example_umap_meta` records the full-vector basis, source views and semantics,
ordered feature IDs, input hash, input/zero-point counts, and actual parameters and
package versions. `n_rows` counts original rows; `n_points` and `n_zero_rows` count
fitted observations (including both sides in a shared answer map). Fit order is
original row order, then batch view order. The input hash covers canonical source
metadata and ordered IDs followed by a NUL and the full little-endian float32
matrix bytes. It does not depend on labels or preference values.

The fit uses Euclidean distance over unscaled float32 vectors, 2 components,
`min(15, n_points - 1)` neighbors, `min_dist=0.1`, spectral initialization, 500 epochs,
seed 42, and one job. Missing UMAP, fewer than four points, or an all-identical
matrix fails explicitly instead of inventing a layout. Deterministic results are
tested in the recorded environment, not promised across versions or platforms.
UMAP can spread identical/zero vectors during optimization; such spacing is not
measured activation difference. UMAP distance is exploratory, not a calibrated
semantic distance, and prompt versus answer/pair maps do not share a coordinate
frame.
