# Migrate from PrefScope 0.2

PrefScope 0.3 has a smaller public API. This is a breaking change: removed imports,
methods, and commands do not have compatibility aliases.

Keep a `prefscope==0.2.0` environment for existing studies until you have checked their
outputs under the new API. Do not overwrite old artifacts during migration. The `0.3.0`
package version in this branch is not a published release until a release is announced.

## Use one extraction method

Use `Lens.featurize(...)` for native, published, precomputed, custom, and SAELens lenses.
It always returns a `FeatureBatch`. Select a named matrix or NumPy array explicitly:

```python
from prefscope import Lens, PairItem

lens = Lens.load("lenses/response", embed_batch_size=8)
items = [PairItem(id="row-1", x="prompt", y_a="answer A", y_b="answer B")]
batch = lens.featurize(items, views=("response_a", "response_b", "response_difference"))
z_a = batch.matrix("z_a")       # FeatureMatrix, with row and feature identity
values = batch.array("z_diff")  # read-only NumPy array
```

This example requires an individual-response lens. A direct-difference lens does not
provide separate A/B response views. Check `lens.capabilities.views` before requesting
views from another backend.

| Old API | New path |
|---------|----------|
| `LoadedLens` | `Lens` |
| `encode`, `encode_one`, `encode_items`, `encode_pairs`, `project` | Create `PairItem` rows and call `featurize`; select the returned array or matrix |
| `Lens.project_representations` | Supply a `RepresentationSource` when constructing a lens, then call `featurize` |
| Lens inspection and analysis methods | Use `FeatureMatrix`, explicit `FeatureCatalog` joins, and direct numerical functions |
| Custom backend registry configuration | Construct the backend and pass it to `Lens.from_backend` |

Native representation lenses reject `featurize(..., batch_size=...)`. Set
`embed_batch_size` when loading a native lens, or configure the injected representation
source. Backends that support a per-call batch size, such as SAELens, still accept it.

## Call analyses directly

`AnalysisDataset`, `AnalysisPlan`, `AnalysisComponent`, `OutcomeSpec`,
`DatasetAnalysisResult`, `TableContract`, `AnalyzeConfig`, `analyze_dataset`, and
`run_analysis` are no longer public framework objects. Fixed analysis-result schemas,
analysis registries, and automatic analysis selection have been removed.

The supported numerical functions accept `FeatureMatrix` directly:

```python
from prefscope import activation_summary, top_activating_rows

summary = activation_summary(z_a)
examples = top_activating_rows(z_a, k=5)
```

Preference statistics, outcome associations, presence calibration, context analysis,
clustering, and historical reporting code remain under `prefscope.recipes`. Import the
specific recipe you need, or adapt it in your study. Recipes are not a stable API. See
[specialized analyses](../recipes/specialized-analysis.md).

Do not treat an activity summary as a replacement for a statistical test. The caller
chooses the outcome, comparison, membership rule, missing-data policy, grouping, and
uncertainty calculation. Raw activity is not semantic presence.

## Keep annotations separate

Use `FeatureCatalog` for proposed names and display annotations. Join by `feature_id`
through `feature_activation_table(...)`, or validate the catalog against the matrix
before a caller-owned join. Labels do not change numerical activations.

Native lens publication includes a catalog tied to the feature space. Write naming
outputs outside the source lens and publish them with `Lens.save(..., annotations=...)`.
Do not edit a lens's annotation files in place while keeping a stale catalog.

## Save caller-owned results

The unmerged reporting development branch also contained a report compiler, privacy
policy objects, a report-bundle reader, and observability wrappers. These are not part
of `0.3`. `Report` only stores the metrics, tables, and metadata you provide:

```python
from prefscope import Report

Report(
    title="Feature activity",
    tables={"summary": summary, "examples": examples},
    metadata={"meaning": "numerical activity, not semantic presence"},
).save("results/new-report")
```

Use a new output directory for each report. Review the tables and metadata before
sharing them. The container does not remove private text or choose what is safe to
publish. `PREFSCOPE_EVENTS_*` settings no longer create run-observability records.

## Re-create feature and viewer bundles

Native lens directories remain loadable when they satisfy the lens manifest contract.
The simplified saved `FeatureBatch` format is a separate local format: a JSON manifest
with `views` and one `.npy` file per view. Old encoded directories and the removed
`FeatureBundleReader` interface are not supported by the new loader. Re-extract features
with the new API into a new directory, or read old artifacts in the matching old
environment and convert them explicitly. Do not change a schema version by hand.

Viewer export now packages existing feature batches, an optional catalog, and
caller-provided tables with an already-built compatible Viewer. It does not calculate
maps, choose examples, run analyses, build the Viewer, or start a server. Old viewer
bundles are not inputs to this exporter. See the
[Viewer data bridge](../reference/viewer-bundle.md) for the required build and data
versions. The Streamlit application and `viewer` extra have been removed.

## Replace removed CLI workflows

Keep the CLI for data preparation, lens training, feature extraction, naming, and
verification. Use direct Python calls for analyses and reports.

The `run`, `analyze`, `report`, `diagnose`, `concepts`, `extract-concepts`,
`win-relevance`, `associate-outcomes`, `compare-responses`, `context-profile`,
`elicit`, `conditional-delta`, and related analysis commands have been removed.
Interpretation role classification and presence-calibration CLI commands are also
removed; retained specialized implementations are recipe code.

`prefscope-view` and `prefscope-viewer` are removed. The remaining
`prefscope-export-viewer` command takes saved feature data and an explicit Viewer build.
Use `prefscope --help` and the [CLI reference](../reference/cli.md) for the current
command list. `init-demo` now creates only the synthetic corpus, not a workflow YAML.
