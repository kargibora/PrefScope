# Python API Reference

PrefScope is usable as a library. The core object is a **`Lens`**: a saved SAE encoder,
a manifest, and optional concept annotations. Its lifecycle is
`train → save → load → encode → analyze`.

Top-level imports (`import prefscope`):

```python
from prefscope import (
    Lens, LoadedLens, load_lens,          # local / Hugging Face lens loaders
    PairItem, Dataset,                    # data contracts
    ColumnMapping, TableDataset, HuggingFaceDataset, prepare_dataset,
    AnalyzeConfig, run_analysis,          # complete frozen-lens workflow
    SAEConfig, TrainConfig, SAELensProjector, SAELensTextBackend,
    LensBackend, LensCapabilities,             # extensible lens backends
    diagnose, evaluate_preference, feature_preference_relevance,  # analyses
    normalize_outcomes, associate_outcomes, associate_outcomes_by_group,
    NormalizedOutcomes, OutcomeAssociationResult,
    RepresentationBatch, RepresentationSource, CallableRepresentationSource,
    EmbeddingRepresentationSource, PrecomputedRepresentationSource,
    FeatureBatch, FeatureMatrix, FeatureCatalog, feature_activation_table,
    TableContract,
    OutcomeSpec, AnalysisDataset, AnalysisArtifact, AnalysisComponent, AnalysisPlan,
    DatasetAnalysisResult, AnalysisDatasetReference, LoadedAnalysisResult,
    FeatureArtifactDiagnostics, OutcomeAssociations,
    PairedConceptShift, PairedOutcomeSpec, PairedOutcomeShifts,
    PromptConditionedOutcomeShifts, PreferenceLengthConfounds, analyze_dataset,
    load_analysis_result, save_analysis_result,
    preference_relevance, load_feature_batch, save_feature_batch,
    concept_presence, paired_concept_shift, compare_encoded_responses,
    registry, load_plugins,               # explicit plug-in activation
)
```

For new typed library code, `prefscope.api` is the canonical import surface. The
same names remain available from top-level `prefscope` for convenience and compatibility.
See [API stability](api-stability.md) for the supported layers.

`import prefscope` is **torch-free**: the heavy `Embedder` / `SAEProjector` /
`build_lens` imports happen lazily inside `Lens` methods, so importing the package
never pulls in torch. `LoadedLens` is a back-compat alias for `Lens`.

`TableDataset` and `HuggingFaceDataset` map arbitrary source columns into
`PairItem`. `ColumnMapping` is the shared mapping, structured-message, and
winner-label contract used by `prepare_dataset(...)` and the
`prefscope prepare-dataset` CLI. `TableDataset(group_id=..., metadata=...)` preserves
independent groups and caller columns in the resulting feature batch. Normalized labels always mean P(A preferred);
winner tokens are declared explicitly rather than inferred.

For the same high-level workflow as `prefscope analyze`:

```python
from prefscope import AnalyzeConfig, run_analysis

config = AnalyzeConfig.load("analysis.yaml", overrides=["data.source.limit=1000"])
outputs = run_analysis(config)             # resumes matching partial output
```

Sources of truth: the facades `prefscope/api/loaded_lens.py` and
`prefscope/api/analysis.py`, their focused `_lens_*` and `analysis_*` implementation
modules, `prefscope/api/analysis_io.py`, `prefscope/api/config.py`,
`prefscope/analysis/__init__.py`, `prefscope/reporting/`,
`prefscope/observability/`, and the documented pipeline runners.

---

## Interchangeable representations and typed analysis

### Canonical basic featurization

Load one strict lens config, normalize rows as `PairItem`, featurize them, and save the
aligned result:

```python
from prefscope import Lens, PairItem, save_feature_batch

lens = Lens.from_config("completion-lens.yaml")
items = [
    PairItem(
        id="row-1",
        x="Explain the result.",
        y_a="Response A",
        y_b="Response B",
        pref=1.0,  # P(A preferred); 0.0 means B, 0.5 means a tie
        meta={"split": "validation"},
    )
]
features = lens.featurize(items, views=("response_a", "response_b"))
save_feature_batch(features, "features/one-row")
```

For a local or Hugging Face table, use the same composition:

```python
from prefscope import (
    HuggingFaceDataset, Lens, TableDataset, save_feature_batch,
)

lens = Lens.from_config("completion-lens.yaml")
dataset = TableDataset(
    "preferences.parquet",
    prompt="prompt", a="answer_a", b="answer_b",
    pref="p_a", id="example_id", group_id="prompt_id",
    metadata=("split",),
)
# For Hub input, choose this adapter instead:
# dataset = HuggingFaceDataset(
#     "owner/dataset", split="train", revision="<commit-sha>", limit=1000,
#     prompt="prompt", a="answer_a", b="answer_b",
#     pref="p_a", id="example_id", group_id="prompt_id",
# )
features = lens.featurize(dataset)
save_feature_batch(features, "features/dataset")
```

A numeric `pref` column is interpreted as P(A preferred). Categorical winner columns need
an explicit `label_mode` and declared A/B/tie tokens; PrefScope does not guess orientation.

Each call uses one lens and one feature space. In particular, apply native prompt and
individual/completion lenses in separate runs and save separate bundles. Inspect
`lens.capabilities.views` before selecting views. The canonical mapping is `prompt` →
`z_prompt`, `response_a` → `z_a`, `response_b` → `z_b`, and
`response_difference` → `z_diff`. The declared difference capability also matters:
`f(e_A - e_B)` from direct contrast projection is distinct from
`f(e_A) - f(e_B)` after separate encoding.

This path performs featurization only. Each view declares its `code_semantics`; built-in
native and SAELens text backends return numerical activity here. The path does not run
analysis, propose names, calibrate semantic presence, or assign a quality score. A saved
`FeatureBatch` retains aligned row IDs, arrays, and all supplied metadata;
`Lens.featurize(...)` supplies prompt/response text and labels. Consequently, bundles from
this path are **local, private artifacts**, not shareable or redacted bundles.

The basic adapters and `Lens.featurize(...)` materialize data in memory. Bound work with
source limits, selected feature IDs, selected views, and a suitable batch size. The
schema-2 writer rejects more than 1,000,000 rows or more than 100,000,000 aggregate
array elements across all saved views. These are safety ceilings, not recommended working
sizes. Small source-checkout compositions are in
[`single_item.py`](../../examples/inference/single_item.py),
[`local_dataset.py`](../../examples/inference/local_dataset.py),
[`inspect_local_features.py`](../../examples/analysis/inspect_local_features.py), and
[`huggingface_dataset.py`](../../examples/inference/huggingface_dataset.py).

### Feature catalogs and activation tables

Numerical coordinates and display annotations have separate lifecycles. A
`FeatureMatrix` owns codes and ordered feature IDs. A `FeatureCatalog` owns proposed
names/descriptions plus their provenance and feature-space identity:

```python
from prefscope import feature_activation_table
from prefscope.integrations import NeuronpediaProvider
from prefscope.presentation import FeatureTableRenderer

matrix = features.matrix("z_a")
catalog = lens.feature_catalog
rows = feature_activation_table(matrix, top_k=5)

provider = NeuronpediaProvider.from_lens(lens)
if not catalog.labels and provider is not None:
    catalog = catalog.merge(provider.fetch(tuple(dict.fromkeys(int(value) for value in rows["feature_id"]))))

rows = feature_activation_table(matrix, catalog=catalog, top_k=5)
FeatureTableRenderer(max_rows=5).print(rows)
```

`FeatureCatalog` accepts only proposed display names/descriptions and their source fields;
fidelity, calibrated presence, context, model tendency, and outcome association remain
separate evidence artifacts. Catalog IDs are strict nonnegative integers. Duplicate,
boolean, fractional, and reserved activation columns fail closed. `select(...)` preserves
an explicit requested order, and `merge(...)` uses right-nonmissing precedence while
retaining source provenance.

Native lens catalogs bind to an exact SAE-weights digest. Current built-in SAELens
coordinates record `declared_unpinned`; the identity contract reserves
`declared_pinned_coordinate` for integrations that can prove a pinned external
coordinate. `feature_activation_table(...)` validates a catalog against the
matrix feature-space identity when both sides declare one and always joins by
`FeatureMatrix.feature_ids`, so selected or reordered features remain aligned.

`NeuronpediaProvider` is an explicit network adapter. It reads the checkpoint-declared
`neuronpedia_id`, requests only selected feature IDs, and returns a catalog snapshot with
source URLs, retrieval status, timestamps, evidence layer, and response digests. Loading
a lens and featurizing text never fetch Neuronpedia automatically. There is no standalone
durable catalog file schema in `0.2`; `to_frame()` is an in-memory compatibility view,
not a versioned persistence contract. Catalogs are not silently added to schema-2 feature
bundles.

`FeatureTableRenderer` is presentation-only. Its deterministic plain formatter does not
import Rich; terminal `style="auto"` lazily uses Rich when available and otherwise falls
back to plain text. It bounds rows, row identifiers, and descriptions and sanitizes
control characters. One-row tables stay compact; visible multi-row tables add `row_id` and
`rank` columns automatically. Observability progress remains a separate event-log
presentation channel.

`Lens.feature_table` and `Lens.concept_names` remain compatibility views.
`Lens.concept_activations(...)` now also accepts a selected/reordered `FeatureMatrix` and
uses the common feature-activation table builder before applying its existing scientific
filters.

A vector source, lens policy, and projected feature view are separate contracts. The
backend-neutral dataset operation is:

```python
features = lens.featurize(dataset)  # aligned FeatureBatch
```

Native embedding lenses use the existing fixed-width representation path internally.
Direct backends, including SAELens and custom hosted systems, implement `LensBackend`.
Both paths expose declared `LensCapabilities` and produce the same role-aware output.

A vector source can still be controlled explicitly:

```python
from prefscope import EmbeddingRepresentationSource, Lens, load_feature_batch

source = EmbeddingRepresentationSource(my_embedder)
representations = source.encode(dataset)       # RepresentationBatch
lens = Lens.load("lenses/completion")
features = lens.project_representations(representations)  # FeatureBatch
response_difference = features.matrix("z_diff")           # FeatureMatrix

# The same typed contract loads an existing encode-dataset directory.
encoded = load_feature_batch("analysis/codes", arrays=["z_diff"])
```

The built-in source accepts any duck-typed embedder with `encode_prompts` and
`encode`. Custom sources implement `RepresentationSource` or use
`CallableRepresentationSource`. Batches retain unique row IDs, aligned metadata,
role, orientation, polarity, code semantics, and credential-free provenance. Metadata
columns use one portable scalar type (string, boolean, or finite number) plus optional
missing values. Saved schema-2 bundles preserve boolean semantic-presence arrays and
float32 activity arrays without changing their dtype.
The current source contract is fixed-width and item-level; pool residual activations in
the source. Token/ragged activation batches are not yet part of this API.

Run a task-centered outcome analysis without depending on pipeline internals:

```python
from prefscope import OutcomeSpec, analyze_dataset

result = analyze_dataset(
    {"response": features.matrix("z_a")},
    {
        "reward": OutcomeSpec(rewards, row_ids=features.row_ids, kind="continuous"),
        "correct": OutcomeSpec(correct, row_ids=features.row_ids, kind="binary"),
    },
    group_ids=prompt_ids,                 # omit when feature metadata has prompt/group_id
)
associations = result.outcome_associations
artifact = result.artifact("outcome_associations")
manifest = result.to_manifest()         # portable summary; tables remain DataFrames
print(artifact.estimand, artifact.metadata)
```

Built-in components attach a versioned `TableContract` to every artifact. The
contract declares required columns, logical pandas types, a unique key, direction, and
units. Construction validates the table without casting or reordering it. Custom
components can omit a contract or provide their own:

```python
from prefscope import AnalysisArtifact, TableContract

schema = TableContract(
    schema_name="my_analysis",
    schema_version=1,
    required_columns=("feature_id", "estimate"),
    dtypes={"feature_id": "integer", "estimate": "float"},
    unique_key=("feature_id",),
    orientation="as_declared",
    units={"estimate": "unitless"},
)
artifact = AnalysisArtifact(
    "my_analysis", table, "declared estimand", table_contract=schema)
```

The artifact manifest includes this schema. `TableContract.from_manifest(...)` strictly
parses the exact eight-field portable form emitted by `to_manifest()`. Result tables
remain ordinary pandas DataFrames.

The result distinguishes descriptive Pearson/OLS effects from its optional
range-midpoint Fisher inference, identifies each BH correction family, and records
whether rows or equal-weight independent groups define the estimand.

For a custom reusable task:

```python
from prefscope import AnalysisArtifact, AnalysisComponent, AnalysisPlan

class MyAnalysis(AnalysisComponent):
    name = "my_analysis"
    def run(self, dataset):
        return AnalysisArtifact(
            name=self.name,
            table=my_table(dataset),
            estimand="explicit description of the analysis unit and quantity",
        )

plan = AnalysisPlan.from_names(["outcome-associations"])
# Or compose instances: AnalysisPlan((OutcomeAssociations(), MyAnalysis()))
```

Label-free checkpoint comparison requires calibrated semantic presence rather than raw
nonzero codes:

```python
presence_a = FeatureMatrix.from_presence(
    lens.presence(z_a), row_ids=features.row_ids, role="response_a")
presence_b = FeatureMatrix.from_presence(
    lens.presence(z_b), row_ids=features.row_ids, role="response_b")
plan = AnalysisPlan((PairedConceptShift(side_a="before", side_b="after"),))
result = analyze_dataset(
    {"before": presence_a, "after": presence_b},
    group_ids=prompt_ids,
    plan=plan,
)
```

This table explicitly reports `delta_b_minus_a`. It refuses raw positive-nonzero codes,
so an exploratory activation fallback cannot silently become a semantic behavior claim.
`FeatureMatrix.from_presence(...)` itself requires every feature to have a confirmed
`semantic_threshold`; a mixed or positive-nonzero fallback fails closed.

Compare aligned checkpoint or response-set outcomes on their raw scale without adding a
project-specific script:

```python
paired_quality = PairedOutcomeSpec(
    quality_before,
    quality_after,
    row_ids=features.row_ids,
    kind="probability",
    side_a="before",
    side_b="after",
    interpretation="higher values mean higher task quality",
)
plan = AnalysisPlan((
    PairedOutcomeShifts(),
    PromptConditionedOutcomeShifts(prompt_features="prompt_concepts"),
))
result = analyze_dataset(
    {"prompt_concepts": prompt_presence},
    paired_outcomes={"quality": paired_quality},
    group_ids=prompt_group_ids,
    plan=plan,
)
```

The overall table estimates the equal-independent-group mean B-minus-A change. The
prompt-conditioned table estimates a real heterogeneity contrast: B-minus-A change when
the calibrated prompt concept is present minus the same change when it is absent. It does
not infer heterogeneity by comparing two separate significance decisions. Binary,
probability, and preference outcomes use bounded finite-sample inference after support
gates. Unbounded continuous outcomes remain descriptive. Missingness is pairwise per
outcome attribute. Generic and paired preference outcomes retain `0.5` ties as neutral values. The binary
logistic preference table drops ties and says so. Its descriptive table retains them as
neutral. Each table states direction, outcome scale, support, test, confidence interval,
and its BH family. Prompt-conditioned BH adjustment covers every prompt-feature and
outcome-attribute test within one paired outcome set. Paired outcome-only analysis does
not need a dummy feature matrix.

`FeatureArtifactDiagnostics` gives every aligned feature set a deterministic density,
L0, zero-row, never-active/always-active, and value-range summary. It deliberately labels
nonzero activity as numerical artifact health, not semantic presence.

`PreferenceLengthConfounds` exposes the existing sensitivity screen through the same plan
contract. It requires explicit `a_minus_b` orientation for both the response features and
length difference plus a declared P(A preferred) outcome. Its result is an entanglement
screen, not evidence that a concept is bad, biased, or causal.


A complete Torch-free composition example is in
[`examples/advanced/custom_analysis_api.py`](../../examples/advanced/custom_analysis_api.py). It injects a
precomputed source, projects it through a duck-typed lens, composes built-in and custom
analysis components, and compares paired outcomes without dataset-specific framework code.

### Durable analysis-result I/O

```text
save_analysis_result(result, out) -> Path
load_analysis_result(path, *, dataset=None)
```

Saving requires contracted artifacts and publishes a strict schema-1 directory of Parquet
tables plus `manifest.json`. Loading without `dataset=` returns a distinct
`LoadedAnalysisResult`: validated tables plus `AnalysisDatasetReference`, but no input
arrays. Passing a complete `AnalysisDataset` reattaches only after exact ordered row-ID,
group-source, and group-partition checks. See [Durable analysis results](analysis-result.md).

### Reporting foundation (experimental)

Schema-2 feature bundles can be accessed without materializing a `FeatureBatch`:

```python
from prefscope.reporting import FeatureBundleReader

source = FeatureBundleReader.open("encoded/dataset")
for chunk in source.iter_chunks(4096, views=("z_a",)):
    consume(chunk.array("z_a"))
```

The reader validates the full bundle on open, then keeps live read-only memory maps.
Its canonical per-view semantics property is `code_semantics_by_view`;
`code_semantics` remains a reader compatibility alias. Schema 1 needs explicit migration
through `load_feature_batch(...)` and `save_feature_batch(...)`. The latter compatibility
loader is eager and validates/loads all declared arrays within fixed budgets. See
[Feature bundle reader](feature-bundle-reader.md).

`prefscope.reporting` also exports `ReportDataset`, mandatory `ReportLineage` and its
dataset/source/compiler/sampling types, typed report v3 contracts, recursive privacy
policies, canonical JSON-table helpers, and strict bundle I/O. Each artifact names its
lineage `source_refs`. A v3 writer expects raw JSON to pass through `json_payload(...)`;
direct object/table payloads are already sanitized and are not transformed twice.

`prefscope.observability` exports automatic and manual event-schema-v1 JSONL recording:

```python
from prefscope import load_analysis_result
from prefscope.observability import observe_run

with observe_run("events.jsonl", pretty=True):
    result = load_analysis_result("results/analysis")
```

The context auto-instruments supported public Lens and durable-artifact operations. No
manual `record(...)` calls are needed. It records correlation IDs, duration, safe
structural counts, and generic `error_type`-only automatic failures. It never generically
serializes arguments or results. `pretty=True` adds bounded, privacy-safe progress lines
on stderr after events are persisted; JSONL remains the durable source of truth. Rich is
optional and lazily imported, with a plain-text fallback.

For zero-code activation, set both environment variables:

```bash
PREFSCOPE_EVENTS_PATH=events.jsonl \
PREFSCOPE_EVENTS_PRETTY=1 \
python experiment.py
```

For production, use an existing resolved private directory; on macOS, use an existing
directory under `/private/tmp`, not `/tmp`.

`pretty=None` (the default) consults `PREFSCOPE_EVENTS_PRETTY`; `pretty=False` explicitly
disables the terminal view. The pretty variable alone does not activate recording. The
environment settings are sampled together when the lazy process-local run is created.
With neither a context nor `PREFSCOPE_EVENTS_PATH`, PrefScope writes no observability
files. `RunEvent`, `JsonlRecorder`, `RecorderLoggingHandler`, and `capture_warnings`
remain available for custom manual use. See
[Run observability](observability.md) for the supported boundary, privacy rules,
process/warning limits, and secure-path requirements.

These are Phase-1 foundations: no report compiler or renderer ships yet. See
[Report bundles](report-bundle.md) and [Run observability](observability.md).

Registration is import-driven. Import a trusted module directly, call
`load_plugins(["my_package.prefscope_plugin"])`, or list it under `plugins` in a `prefscope run` config. PrefScope does not scan installed packages automatically.

See [Add a representation source](../extending/add-a-representation-source.md).

---

## The lens object — `prefscope.Lens`

Wraps a built lens directory (SAE projector + embedder + optional
`feature_names.csv`) as a reusable inference artifact. Convention: `y_a` = "self"
(model under study), `y_b` = "other"; pair codes are signed self-minus-other;
`pref` = P(self preferred).

### Loading

```text
Lens.load(lens_dir, *, device="cpu", annotations=None) -> Lens
Lens.from_pretrained(repo_id, *, revision=None, subfolder=None, device="cpu") -> Lens
Lens.from_saelens(release, sae_id, *, input_rep="individual", device="cpu") -> Lens
Lens.from_config(path_or_mapping, *, device=None) -> Lens
Lens.from_backend(backend) -> Lens
lens.featurize(dataset, *, views=None, feature_ids=None, batch_size=None) -> FeatureBatch
lens.project_saelens_tokens(*, row_ids, token_activations, token_row_ids,
                            representation_contract, feature_ids=None) -> FeatureBatch
load_lens("hf://owner/repository[/subfolder]", *, revision=None, device="cpu") -> Lens
```

Builds the real torch `SAEProjector` + `Embedder` (embedder model id taken from
the manifest's `embed_model_id`) and records the backing directory as
`lens.lens_dir`. The loader merges bundled name, fidelity, calibration, context,
and cluster tables by `feature_id`; `annotations=` can attach an external
interpretation directory without first copying it. `from_pretrained` downloads
an ordinary lens directory through the Hugging Face cache and then uses the same
local loader. Mutable refs (including omitted `revision`) resolve to a 40-hex commit
before download. The loaded object exposes `requested_revision` and `resolved_revision`;
these runtime fields never mutate the published manifest or retain an access token.
Offline loading of Hub artifacts requires an explicit commit SHA.

`Lens.from_saelens(...)` is a separate experimental loader for a pretrained
internal-activation SAE. It needs the optional `saelens` extra. `lens.featurize` loads
one reader lazily and produces prompt, response-A, response-B, and derived A-minus-B
views. The exact-activation method remains an advanced escape hatch. Both paths encode
exact-hook tokens before max pooling and can retain selected features. They do not make
the external release a pinned PrefScope lens artifact.
See [Use a pretrained SAE through SAELens](../how-to/use-saelens.md).

One repository may contain several lenses:

```python
completion = Lens.from_pretrained(
    "owner/repository", subfolder="completion", revision="v1")
prompt = Lens.from_pretrained(
    "owner/repository", subfolder="prompt", revision="v1")
```

For the common one-prompt/one-response path, the convenience API loads both lenses,
shares one embedder instance, and returns concept rows with presence provenance:

```python
from prefscope import extract_text_concepts

result = extract_text_concepts(
    "Explain why the sky is blue.",
    "Shorter wavelengths scatter more strongly.",
    repo_id="owner/repository",
    device="cuda",
    presence_policy="calibrated",
)
print(result["prompt"])
print(result["completion"])
```

Pass `prompt_lens=` and `completion_lens=` for local directories or `hf://` sources.
The default verified-only filter can be relaxed with `fidelity_only=False`; doing so
makes the resulting names exploratory rather than verified.

### `encode` / `encode_one` — per-response codes

```text
lens.encode(prompts, completions=None) -> np.ndarray   # (N, M)
lens.encode_one(prompt, completion=None) -> np.ndarray # (M,)
```

Concept codes for individual responses. For an **individual** lens, both prompt
and completion are embedded; for a **prompt** lens, the prompt alone is embedded
(completions ignored). A single `str` is accepted for either argument (wrapped to
length 1; `encode` still returns a 2-D array, `encode_one` returns 1-D). A
**difference** lens is contrast-only and raises `ValueError` — use `encode_pairs`
instead.

```python
lens = load_lens("lenses/indiv_8b")
codes = lens.encode(["Write a poem", "Explain gravity"],
                    ["Roses are red…", "Mass curves spacetime…"])   # (2, M)
one = lens.encode_one("Write a poem", "Roses are red…")             # (M,)
```

### `concept_names` / `top_concepts` — naming

```text
lens.concept_names                          # pd.Series feature_id -> name, or None
lens.top_concepts(codes, k=5) -> list[list[tuple[str, float]]]
```

`concept_names` is `None` unless the lens carries a named `feature_names.csv`.
`top_concepts` returns, per row, up to `k` active **named** features with the largest
`|code|` as `(concept, signed_value)` pairs. Unnamed/zero axes are skipped. For a signed
lens it defaults to the positive pole because the name describes that pole; pass
`matching_pole_only=False` only for deliberate axis-level inspection. The input must
have exactly the lens's `m_total` feature columns; width mismatches are rejected before
names are attached. Prefer `concept_activations` when you need explicit pole metadata.

```python
codes = lens.encode(prompts, completions)
for row in lens.top_concepts(codes, k=3):
    print(row)   # e.g. [('verbosity', 2.1), ('code blocks', 0.9)]
```

### `feature_catalog` / `feature_table` / `concept_activations`

```text
lens.feature_catalog     # proposed display labels + feature-space identity
lens.feature_table       # compatibility view with bundled scientific annotations
lens.concept_activations(
    codes,
    row_ids=None,
    active_only=True,
    pole="any",                       # any | positive | negative
    min_abs_activation=0.0,
    top_k=None,                       # None = every active feature
    fidelity_only=False,
    semantic_presence_only=False,
) -> pd.DataFrame
```

`concept_activations` accepts either the historical full-width ndarray or a
selected/reordered `FeatureMatrix`. It returns one row for each retained item-feature pair
and joins by feature ID. It keeps the feature ID, concept name, activation, rank, pole,
semantic-presence status, and bundled scientific annotations. A negative value on a
signed lens points away from the stored positive-pole
name, so the table marks it as not matching that name. Zero features are omitted by
default because including them can create `N × M` rows.

For calibrated concepts from a prompt lens:

```python
lens = Lens.from_dir("prompt-lens")
if lens.input_rep != "prompt":
    raise ValueError("This task needs a prompt lens")
codes = lens.encode(prompts)
concepts = lens.concept_activations(
    codes, row_ids=row_ids, semantic_presence_only=True)
```

See [Extract concepts from every prompt](../how-to/extract-prompt-concepts.md) for a
complete example.

### `project_representations` — source-agnostic projection

```text
lens.project_representations(batch: RepresentationBatch) -> FeatureBatch
```

Requires `prompt` for a prompt lens or `response_a`/optional `response_b` for a
response lens. It validates source width against the lens and returns named views with
explicit orientation. A difference lens cannot project a single response.

### `encode_pairs` — paired contrast codes

```text
lens.encode_pairs(dataset, *, return_meta=True) -> (codes (N, M), meta)  # alias: project
```

Iterates the dataset (`PairItem`-like objects with `.x`, `.y_a`, `.y_b`, `.id`,
`.pref`, `.model_a`, `.model_b`), embeds both responses, forms the lens's contrast
(per `input_rep`), and projects through the SAE to signed self-minus-other codes.
`meta` carries `id`, `pref`, `model_a`, `model_b`. `return_meta=False` returns just
the codes. It raises on single-response items or a token-granularity lens.
`lens.project` is kept as an alias and still returns the
`(codes, meta)` tuple.

### `encode_items` — paired or single-response datasets

```text
lens.encode_items(dataset, *, return_meta=True) -> (codes, meta)
```

Accepts a homogeneous iterable: paired items delegate to `encode_pairs`; items with
`y_b=None` produce absolute per-response codes through an **individual** lens. Mixed
paired/single input and single input on a difference lens raise clearly. Preference
analyses below require the paired contrast form.

### Analyses on pair codes

```text
lens.diagnose(codes, meta, *, fidelity_only=False) -> pd.DataFrame
lens.feature_preference_relevance(codes, meta) -> pd.DataFrame
lens.evaluate_preference(codes, meta, **kwargs) -> dict
```

`diagnose` gives per-feature over/under-expression + outcome association (sorted by
`net_direction`, names attached, optionally restricted to fidelity-passing
features). `feature_preference_relevance` gives per-feature univariate preference
relevance. `evaluate_preference` returns a cross-validated logistic readout dict
(`n`, `accuracy`, `auc`, `baseline_accuracy`, `n_features`, `top_features`); kwargs
forward to `analysis.evaluate_preference` (`n_splits=5`, `seed=0`). All three
delegate to `prefscope.analysis`.

```python
codes, meta = lens.encode_pairs(my_dataset)
diag  = lens.diagnose(codes, meta, fidelity_only=True)
rel   = lens.feature_preference_relevance(codes, meta)
score = lens.evaluate_preference(codes, meta)   # {'accuracy': ..., 'auc': ..., ...}
```

### `save`

```text
lens.save(dest, *, overwrite=False, annotations=None, inference_only=False) -> Path
```

Atomically copies the backing lens directory to `dest`; it refuses a non-empty
destination unless `overwrite=True` and never merges with stale files. Pass an
interpretation directory or a list of canonically named CSVs through `annotations=` to assemble one
self-contained shareable lens:

```python
lens = Lens.load("lenses/completion_m2048", annotations="interpret/completion_m2048")
lens.save(
    "release/completion_m2048",
    annotations="interpret/completion_m2048",
    inference_only=True,                 # omit corpus-aligned z_*.npy and text
)
```

The resulting directory can be uploaded as a Hugging Face model repository:

```bash
hf auth login
hf upload owner/repository ./release/completion_m2048 .
```

The installed `prefscope package-lens` command is the recommended public workflow
because it also validates the migrated manifest and can attach a model card. See
[Publish a lens](../how-to/publish-a-lens.md).

---

## Synthetic quickstart data

```python
from prefscope import create_demo, make_demo_corpus

frame = make_demo_corpus()           # deterministic 60-row DataFrame
paths = create_demo("demo")          # corpus + matching quickstart.yaml
```

This is smoke-test data only; its concepts and statistical results are not research
evidence.

---

## Training a lens — `Lens.train`

```text
Lens.train(data, config=TrainConfig(), *, out, columns=None) -> Lens
```

Trains and saves a fresh lens, then loads it. `data` is normalized to a battles
DataFrame by `pairs_to_battles` (below), embedded, and passed to `build_lens`; the
result directory at `out` is loaded into a `Lens`.

### Configuration — `SAEConfig` / `TrainConfig`

```python
@dataclass
class SAEConfig:            # architecture — defines the frozen lens
    m: int = 128
    k: int = 16
    input_rep: str = "individual"     # "individual" | "difference" | "prompt"
    sae_type: str = "auto"             # individual/prompt -> batchtopk-relu
    matryoshka_prefix: tuple = ()       # opt in to nested-width training
    sparsity_coef: float = 1e-3         # JumpReLU
    bandwidth: float = 1e-3             # JumpReLU STE
    sparsity_warmup_steps: int = 0

@dataclass
class TrainConfig:          # run-time
    sae: SAEConfig = SAEConfig()
    embed_model_id: str | None = None # None -> embedder/config default
    embed_model_revision: str | None = None # pin tag or commit for reproducibility
    val_frac: float = 0.1
    device: str = "cpu"
    max_train_rows: int | None = None
    train_kwargs: dict = {}           # n_epochs/lr/etc. -> build_lens **train_kwargs
```

`SAEConfig` is the part that defines what the lens *is* (width, sparsity,
architecture, polarity, and input representation). `auto` resolves to signed
BatchTopK for direct differences and non-negative BatchTopK for individual responses
or prompts. `TrainConfig` adds run-time knobs; entries in `train_kwargs` forward to
`build_lens` (for example `n_epochs` and `lr`).

### `pairs_to_battles` — data normalization

```text
pairs_to_battles(data, columns=None) -> pd.DataFrame
```

Pure (no torch, no embedding). Accepts:

- a `Dataset` / iterable of `PairItem` — mapped
  `x→prompt`, `y_a→completion_a`, `y_b→completion_b`, `id→instruction_id`, plus
  `pref→human_pref`, `model_a`, `model_b` when present;
- a `pd.DataFrame` — the `columns` rename map is applied first, then the three
  required columns (`prompt`, `completion_a`, `instruction_id`) are validated;
  `completion_b` is optional for homogeneous single-response data;
- a `str` / `Path` — read as parquet, then treated as a DataFrame.

Raises `ValueError` if any required column is missing.

---

## Two flows

### 1. Use a trained lens on (prompt, completion) lists

```python
from prefscope import load_lens

lens = load_lens("lenses/indiv_8b", device="cpu")
codes = lens.encode(prompts, completions)          # (N, M)
for row in lens.top_concepts(codes, k=5):
    print(row)
```

### 2. Train from a custom `Dataset`

```python
from prefscope import Lens, Dataset, PairItem, TrainConfig, SAEConfig

class MyData(Dataset):
    def __iter__(self):
        yield PairItem(id="1", x="Write a haiku", y_a="…", y_b="…",
                       pref=1.0, model_a="A", model_b="B")
        # …

cfg = TrainConfig(sae=SAEConfig(m=128, k=16, input_rep="individual"),
                  device="cuda", train_kwargs={"n_epochs": 20})
lens = Lens.train(MyData(), cfg, out="lenses/my_lens")
codes, meta = lens.encode_pairs(MyData())
score = lens.evaluate_preference(codes, meta)
lens.save("releases/my_lens_v1")
```

For instruction-tuning rows with one output each, omit `y_b`, keep
`input_rep="individual"`, then call `lens.encode_items(data)`. The saved artifact has
`dataset_mode: single`, `z_a.npy`, and enough text in `battles.parquet` to run
`name → verify → cluster` without a separate corpus file.

`Lens.train` also accepts a DataFrame or a parquet path directly:

```python
lens = Lens.train("battles.parquet", TrainConfig(), out="lenses/from_parquet")
lens = Lens.train(df, TrainConfig(), out="lenses/from_df",
                  columns={"q": "prompt", "a": "completion_a",
                           "b": "completion_b", "iid": "instruction_id"})
```

---

## Format-agnostic analyses — `prefscope.analysis`

Operate on `(codes, meta)` or raw `z` matrices, independent of how the codes were
produced. Paired preference helpers use signed self-minus-other codes and require
`pref = P(self preferred)`. Presence, region, and generic outcome helpers can instead use
`z_a`, `z_prompt`, or other aligned feature matrices and do not require preference labels.
`diagnose`, `evaluate_preference`, and `feature_preference_relevance` are re-exported at
the top level (`prefscope`).

Outcome normalization and association are also reusable directly:

```python
outcomes = normalize_outcomes(
    ratings[["helpfulness", "correctness"]],
    kind="multi_continuous", normalization="auto")
result = associate_outcomes(codes, outcomes, group_ids=prompt_ids)
long_table = result.table
```

`normalize_outcomes` validates binary `[0,1]`, bounded probability/preference, continuous,
and multi-continuous contracts while preserving per-attribute missingness. `auto` keeps
bounded scales natural and z-scores continuous scales. `associate_outcomes` returns
Pearson descriptive associations. It reports p/q values only when both range-midpoint arms
of the feature and outcome have at least five independent units; thin cells remain
explicitly descriptive. Repeated groups are first reduced to per-group means and
continuous normalization is recomputed across those means, giving every independent
prompt equal weight; `associate_outcomes_by_group` is an explicit convenience wrapper for the same
equal-group-weight analysis.

Re-exported from `prefscope.analysis` (`__init__.py`):

| function | returns | summary |
|----------|---------|---------|
| `normalize_outcomes(values, *, kind, names=None, normalization="auto")` | `NormalizedOutcomes` | validated 2-D values plus scale/missingness provenance |
| `associate_outcomes(codes, outcomes, *, feature_ids=None, group_ids=None, min_units=3)` | `OutcomeAssociationResult` | long-form feature × outcome associations with explicit row/group estimand |
| `associate_outcomes_by_group(codes, outcomes, group_ids, *, feature_ids=None, min_groups=3)` | `OutcomeAssociationResult` | explicit convenience wrapper for equal-group-weight associations |
| `diagnose(codes, meta, *, names=None, fidelity_only=False)` | DataFrame | per-feature over/under-expression + outcome assoc, sorted by `net_direction` |
| `feature_preference_relevance(codes, meta, *, names=None, group_ids=None, group_col=None)` | DataFrame | per-feature univariate preference relevance |
| `evaluate_preference(codes, meta, *, n_splits=5, seed=0, names=None, group_col=None)` | dict | CV logistic readout (accuracy/auc/top_features) |
| `inside_outside_contrast(inside, outside)` | dict | Welch two-sample contrast (mean/delta/welch_t/welch_p/cohens_d) |
| `dataset_reward(z)` | ndarray | per-feature reward summary over a dataset |
| `split_half_stable(z, effect_fn, *, seed=0)` | DataFrame | split-half stability of a feature effect |
| `spurious_share(z, undesirable, *, eps=1e-9)` | ndarray | share of activity attributable to an undesirable surrogate |
| `label_inconsistency(z, reward, undesirable)` | ndarray | per-feature label-inconsistency signal |
| `diagnose_dataset(z, undesirable, *, ids=None, names=None, seed=0)` | — | dataset-level diagnosis bundle |
| `symmetric_activity(z_a, z_b)` | ndarray | per-feature A/B activity symmetry |
| `region_behavior_contrast(z, cluster_ids, *, seed=0)` | DataFrame | per-cluster region/behavior contrast |
| `feature_confound_correlation(z, surrogate)` | DataFrame | per-feature correlation with a confound surrogate |
| `screen_length_confound(z_diff, human_pref, length_difference, *, annotations=None, confound_threshold=0.3, collapse_fraction=0.5, permutations=0, seed=0, group_ids=None)` | `(DataFrame, dict)` | sensitivity screen for features whose preference association overlaps with response length; descriptive, not causal |
| `auto_undesirable(z, surrogate, *, threshold=0.3)` | list | feature ids auto-flagged undesirable by surrogate correlation |

```python
import numpy as np, pandas as pd
from prefscope import diagnose, evaluate_preference

codes = np.load("lenses/indiv_8b/z_diff.npy")    # (N, M)
meta = pd.DataFrame({"pref": ...})               # P(self preferred) per row
diag = diagnose(codes, meta)                     # per-feature tendencies
score = evaluate_preference(codes, meta)         # CV readout dict
```

---

## Plug-in registry — `prefscope.registry`

Strategies (interpreters, verifiers, clusterers) register under a `kind`. List
the names registered for a kind, then build one by name:

```python
import prefscope.interpret.strategy  # activates built-in naming strategies
from prefscope import registry

registry.available("interpreter")            # -> list of registered names
obj = registry.make("interpreter", name, ...)  # construct by name (kwargs forwarded)
```

---

## Config pipeline — `prefscope.pipeline.run`

### `PipelineConfig`
Typed, validated view of a pipeline config (see `config-schema.md`).

```text
PipelineConfig.from_dict(d: dict) -> PipelineConfig   # validate an in-memory mapping
PipelineConfig.load(path) -> PipelineConfig           # load .yaml/.yml/.json then from_dict
```

Fields: `lens_dir`, `out_dir`, `stages`, `corpus`, `annotations`, `lens_kind`,
`llm` (`LLMConfig`), `interpreter`/`verifier`/`clusterer` (`StageConfig`),
`win_relevance` (dict). Returns the dataclass; raises `ValueError` on a bad/unknown
key.

### `run_pipeline`
```text
run_pipeline(cfg: PipelineConfig, *, client=None, verbose: bool = True) -> dict
```
Executes `cfg.stages` in canonical order, threading artifacts under `out_dir`.
Returns `{stage_name: output_Path}`. `client` overrides the LLM client (tests
inject a fake); otherwise the config's `llm` block builds one lazily on the first
LLM stage. `preflight(cfg)` (also public) fails fast on a missing lens/corpus.

```python
from prefscope.pipeline.run import PipelineConfig, run_pipeline

cfg = PipelineConfig.load("examples/workflows/pipeline.yaml")
outputs = run_pipeline(cfg)            # runs name -> verify -> cluster -> win-relevance
print(outputs["win-relevance"])        # Path to win_relevance.csv under out_dir
```

---

## Raw SAE projection — `prefscope.encode.sae.SAEProjector`

Frozen SAE projector. Accepts a path to `sae_model.pt` or the lens dir
containing it; auto-applies the lens's `whiten.npz` if present.

```text
SAEProjector(model_path, device: str = "cpu")
  .project(x: np.ndarray (N, D)) -> np.ndarray (N, M)     # sparse codes
  .activation_polarity                                  # signed | nonnegative
  .code_semantics                                       # axis | presence
  .reconstruct(z: np.ndarray (N, M)) -> np.ndarray (N, D) # back to embedding space
  .residual_norm(x: np.ndarray (N, D)) -> np.ndarray (N,) # ||x - recon|| per row
```
Attributes: `m_total` (M), `input_dim` (D), `config`, `device`. `project` raises
`ValueError` if the input dim != `input_dim` (embedder/lens mismatch).

```python
import numpy as np
from prefscope.encode.sae import SAEProjector

proj = SAEProjector("lenses/indiv_8b", device="cpu")
e = np.load("emb/e_a.npy").astype(np.float32)   # (N, D) embeddings
z = proj.project(e)                              # (N, M) sparse codes
resid = proj.residual_norm(e)                    # off-dictionary signal per row
```
## Paired response comparison

`compare_encoded_responses(...)` consumes an individual-lens encoded A/B bundle and
returns a `ResponseComparison` containing overall shifts, prompt-conditioned shifts,
scope classifications, and paired examples. `ResponseComparison.save(path)` writes the
stable artifact contract used by the viewer.

Lower-level, array-oriented analysis is available through:

```python
from prefscope import (
    concept_presence,
    paired_concept_shift,
    paired_concept_shift_by_region,
    summarize_response_scope,
)
```

`Lens.presence(codes, policy="calibrated")` applies the annotations bundled with a loaded
lens and returns feature-aligned boolean values plus threshold provenance.
