# Python API

The supported analysis path uses direct objects and functions. `import prefscope` does not
import Torch.

## Items and lenses

```python
from prefscope import Lens, PairItem

item = PairItem(id="1", x="prompt", y_a="response A", y_b="response B")
lens = Lens.from_pretrained("owner/repo", subfolder="lens")
batch = lens.featurize([item], views=("response_a", "response_b"))
```

`Lens` supports local native lenses, published native lenses, precomputed representations,
custom `LensBackend` implementations, and the optional SAELens integration. A backend
must declare supported views through `LensCapabilities`.

Key entry points:

- `Lens.load(path, ...)`
- `Lens.from_pretrained(repo_id, ...)`
- `Lens.from_backend(backend)`
- `Lens.from_saelens(...)`
- `Lens.featurize(items, *, views=None, feature_ids=None, batch_size=None)`
- `load_lens(path, ...)`

`Lens` does not expose analysis-plan, report-compilation, or component-registry methods.
Use `featurize` for feature extraction; the older `encode`, `encode_items`, and
`encode_pairs` methods are no longer public methods on `Lens`.

### Batching

Batching belongs to the backend or representation source. For a native lens, set the
embedding batch size when loading it:

```python
lens = Lens.load("lenses/response", embed_batch_size=8)
batch = lens.featurize(items)
```

The same `embed_batch_size` option is available on `Lens.from_pretrained`. For an
injected `RepresentationSource`, configure its batching before passing it to `Lens`.
Native representation lenses reject `featurize(..., batch_size=...)` rather than
silently ignoring it.

SAELens supports `featurize(..., batch_size=8)` as a per-call text batch size. Custom
backends must honor an explicit `batch_size` or raise `ValueError`. Leave it as `None`
to use the backend's configured batching. This option does not make `featurize` a
streaming API; it still returns the complete `FeatureBatch`.

## Feature data

### `FeatureBatch`

A batch contains several aligned arrays plus:

- `row_ids`
- `feature_ids`
- `roles`
- `orientations`
- row-aligned `metadata`
- `activation_polarity`
- `code_semantics`
- `provenance`

Use `batch.array("z_a")` for a read-only NumPy array or
`batch.matrix("z_a")` for a `FeatureMatrix`.

### `FeatureMatrix`

A matrix holds one two-dimensional array and the same identity and semantics fields. The
feature IDs are coordinates, not column positions in an unrelated matrix.

### Saving feature data

```python
from prefscope import load_feature_batch, save_feature_batch

save_feature_batch(batch, "artifacts/features")
restored = load_feature_batch("artifacts/features")
```

The directory contains one JSON manifest and one `.npy` file per view. Use `overwrite=True`
explicitly to replace an existing destination. This is a local artifact format, not a
hostile-input storage or policy framework.

## Feature annotations

```python
from prefscope import FeatureCatalog, feature_activation_table

catalog = FeatureCatalog.from_mapping({0: "concise", 4: "stepwise"})
table = feature_activation_table(batch.matrix("z_a"), catalog=catalog)
```

`FeatureCatalog` holds proposed display annotations. It never changes activation data.
`feature_activation_table(...)` joins by `feature_id`.

Catalog JSON helpers:

- `encode_feature_catalog(catalog)`
- `decode_feature_catalog(data)`
- `save_feature_catalog(catalog, path, *, overwrite=False)`
- `load_feature_catalog(path)`

## Numerical analysis

All supported functions accept `FeatureMatrix` directly.

### `activation_summary(matrix)`

Returns one row per feature with nonzero count, nonzero rate, mean, mean when nonzero, and
maximum. “Nonzero” is a numerical property, not semantic presence.

### `coactivation_counts(presence)`

Returns a square joint-count table for an explicitly boolean `FeatureMatrix`.

### `coactivation_pairs(presence, *, min_count=1)`

Returns unordered boolean-overlap pairs with counts and Jaccard overlap.

### `cross_coactivation_counts(left, right)`

Returns cross-counts for two boolean matrices with identical ordered row IDs.

### `top_activating_rows(matrix, *, k=10)`

Returns each feature's largest signed values with stable ties and explicit row IDs.

Coactivation functions require an explicitly boolean `FeatureMatrix`. The caller owns the
rule that created that matrix. Raw activity must not be silently presented as semantic
presence. Historical calibration helpers are recipe code, not supported analysis API.

## Reports

```python
from prefscope import Report

report = Report(
    title="Example",
    metrics={"n_rows": batch.matrix("z_a").n_rows},
    tables={"activity": activation_summary(batch.matrix("z_a"))},
    metadata={"note": "caller-owned"},
)
report.add_metric("selected_features", 12)
report.add_table("custom", custom_dataframe)
report.save("results/report")
```

Metrics must be finite real numbers. Tables are pandas DataFrames. Metadata must be JSON
serializable. `Report` does not infer table schemas, run analyses, select statistics,
filter private fields, or generate narrative.

## Visualization data

```python
from prefscope.viewer_export import export_viewer_bundle

export_viewer_bundle(
    batch,
    "results/viewer-site",
    viewer_dist="path/to/viewer/dist",
    catalog=catalog,
    tables={"activity": activity_table},
)
```

The required Viewer directory must already be built and declare support for
`prefscope.viewer_data` v1 in its root `viewer-build.json`. The new output directory is a
self-contained static site with copied Viewer assets, serialized supplied data, and a
hashed inventory. The bridge does not build the Viewer or derive maps, examples,
distributions, or coactivation.

## Specialized recipes

Modules below `prefscope.recipes.analysis`, `prefscope.recipes.pipeline`, and
`prefscope.recipes.viewer_export` retain specialized implementations for reuse. They are
not exported from `prefscope`, covered by the stable API, registered, or automatically
orchestrated. Prefer copying or wrapping the exact recipe your study needs.
