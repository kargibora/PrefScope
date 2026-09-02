# Changelog

Notable user-visible changes are recorded here. PrefScope follows semantic versioning;
the `0.2` series is an alpha API and may still evolve with explicit release notes.

## 0.2.0 — 2026-09-01

- Prepared `PrefScope` as the canonical public repository: updated package,
  citation, clone, documentation, and issue URLs; added the project logo; expanded
  local credential ignores; and shipped the upstream WIMHF MIT notice with adapted
  prompt templates.

- Added the experimental Phase-1 reporting foundation. Contracted task-centered summary
  results now publish under per-table and aggregate budgets, load as detached
  `LoadedAnalysisResult` tables, and reattach only to an exactly matching dataset.
  Schema-2 `FeatureBundleReader` provides validated live read-only memory maps, bounded
  Parquet/NPY preflight, chunking, and explicit row/view selection. Schema-1 bundles use
  the explicit eager `load_feature_batch` compatibility and migration path, with fixed
  budgets over every declared array.
- Added observability event schema v1, secure bounded JSONL recording, and opt-in
  automatic events for supported Lens and durable-artifact operations. `observe_run(...)`
  activates context-local spans; `PREFSCOPE_EVENTS_PATH` activates a process-local
  recorder lazily on first instrumented use. `observe_run(..., pretty=True)` and the
  zero-code `PREFSCOPE_EVENTS_PRETTY=1` setting add bounded privacy-safe progress lines on
  stderr after successful persistence, with colored interactive Rich rendering and a
  plain-text fallback. The normal `examples/advanced/presentations/compare_completions.py` flow compares
  two completions in colored raw-activation tables without manual event calls.
  `pretty=None` consults the environment, while `pretty=False` overrides it;
  the pretty setting alone does not activate logging. With neither recording opt-in,
  observability performs no file writes. Automatic events carry correlation/timing and
  safe structural fields; omit raw text, paths, caller IDs, payloads, and exception
  messages; and bridge
  PrefScope logs and emitted warnings within their documented limits. Also added strict
  report bundle v3
  I/O rooted at manifest-last `bundle_manifest.json`. V3 requires
  exact dataset/source/compiler/spec/sampling-frame lineage and artifact source references,
  enforces section/artifact status and evidence roll-up, and distinguishes raw
  `json_payload` sanitation from already-sanitized object/table payloads. Canonical
  JSON-table v1 and recursive typed local/shareable privacy roles fail closed on unknown
  shareable fields, direct email/phone literals, secrets, and small cells. Phase-1
  shareable artifacts are JSON-only. Publication uses persistent advisory lock files,
  rejects untrusted parents and non-owned/multi-link lock inodes before mutation, imports
  Unix locking lazily, uses no-replace new installs on Darwin/Linux, and retains
  recoverable staging; overwrite is not a
  linearizable atomic directory exchange. Viewer bundle v2 is unchanged. A report
  compiler and renderer remain Phase-2 work; browsers do not recompute scientific results.

- Added a Torch-free `FeatureCatalog` proposed-label artifact, exact/declared
  feature-space identity, selected/reordered-ID-safe `feature_activation_table(...)`, an
  explicit provenance-bearing `NeuronpediaProvider`, and bounded lazy-Rich/plain
  `FeatureTableRenderer`. `Lens.feature_catalog` exposes native names without copying
  annotations into numerical `FeatureBatch` objects. Existing `feature_table`,
  `concept_names`, and ndarray `concept_activations(...)` contracts remain; the latter now
  also accepts `FeatureMatrix` and joins annotations by feature ID. Live `ReportDataset`
  accepts typed catalogs while retaining DataFrame compatibility.

- Reorganized source-checkout examples into a self-contained capability gallery for
  inference, training, analysis, workflows, assets, and advanced demonstrations. Basic
  cards use editable constants instead of argument-heavy mini-CLIs, stay under 80 lines,
  wrap public operations in pretty observability, and print compact results. The
  single-response SAELens card prints top codes with matching Neuronpedia descriptions.
  The Hub dataset revision is optional, with exact commits recommended for reproducibility.
  Feature-batch semantics now fail closed, and selected-view provenance is pruned so every
  successfully published schema-2 bundle has matching eager/lazy reader contracts.
  The completion-comparison example also keeps event logging opt-in, bounds displayed rows,
  and suppresses inactive zero features. A separate local multi-row inspection card joins
  selected codes to proposed descriptions without changing the numerical bundle example;
  the feature-table renderer adds bounded `row_id` and `rank` columns when multiple rows
  are visible.
- Added the backend-neutral `Lens.featurize(...) -> FeatureBatch` contract with declared
  `LensCapabilities`, a public `LensBackend` extension point, strict lens YAML loading,
  direct typed analysis input, transactional `save_feature_batch(...)`, and grouped
  `preference_relevance(...)`. Native
  representation lenses and pretrained SAELens checkpoints now share this dataset API;
  historical ndarray methods keep their signatures and delegate where supported. Table
  adapters now preserve first-class group IDs and caller metadata. Per-view provenance
  correctly marks derived A-minus-B activity as signed even when per-side SAE activity
  is nonnegative. The SAELens reader honors proxy-model tokenization and checkpoint
  BOS, sequence-position, and special-token exclusion metadata.

- Added an optional experimental SAELens 6.50+ backend. `SAELensProjector`,
  `Lens.from_saelens(...)`, and `project_saelens_tokens(...)` reuse registered
  pretrained flat SAEs without training a PrefScope SAE. The token path applies the SAE
  before max pooling, supports selected features and bounded dense chunks, and fails
  closed on coordinate, direct-difference, structured-hook, temporal-SAE, and implicit
  pre-SAE-pooling mismatches. Raw outputs remain numerical activity. Mutable external
  release identifiers are recorded as unpinned rather than published as PrefScope lens
  artifacts, and SAELens/PyTorch remain outside base imports.

- Clarified the supported-alpha API layers and made `prefscope.api` the canonical typed
  import surface while preserving all top-level and legacy facades. Split the analysis,
  lens, and analyze-config implementations into focused modules without changing public
  method signatures. Built-in analysis artifacts now carry versioned `TableContract`
  schemas with required columns, logical pandas types, unique keys, direction, and units.
  The `prefscope run` config now accepts an explicit ordered `plugins` module list;
  imports are deterministic, trusted-code only, and occur before
  registry validation. Shared viewer-export and workflow default profiles now come from
  named constants rather than copied literals.

- Added a Torch-free, source-agnostic Python boundary: `RepresentationSource` produces
  validated aligned `RepresentationBatch` vectors, and `Lens.project_representations`
  returns typed `FeatureBatch` views with role, A/B orientation, metadata, and provenance.
  The built-in text source is duck-typed, while custom embedding or pooled-residual
  sources can be passed directly without changing lens or analysis code.
- Added the task-centered `analyze_dataset` API with `FeatureMatrix`, `OutcomeSpec`,
  `AnalysisPlan`, typed artifacts/results, and composable `AnalysisComponent` plug-ins.
  A registered `PrecomputedRepresentationSource` now serves aligned static or pooled
  residual arrays with exact requested-ID checks; callable and lens item-encoding paths
  also reject reordered source output. Lens projection now compares declared
  representation model/revision/preprocessing coordinates instead of trusting width;
  an explicit unsafe override is recorded in feature provenance. Typed item encoding now requires non-empty unique
  `PairItem.id` values; this is a deliberate fail-closed tightening of the legacy path.
  Empty `pairs_to_battles([])` still returns the canonical empty schema.
  `load_feature_batch` validates reusable encoded directories, and the paired concept
  component compares only explicitly calibrated semantic-presence matrices. General
  paired-outcome components now report raw-scale B-minus-A checkpoint/response-set
  changes and actual prompt-concept heterogeneity contrasts, with equal-group weighting,
  outcome-specific missingness, side labels, bounded inference/support gates, and
  separate multiplicity families. A task-centered preference-length-confound component
  now enforces A-minus-B feature/length and P(A preferred) orientation before running the
  descriptive sensitivity screen. Preference/outcome tables now state missingness and
  tie policies explicitly: generic/paired preference values retain `0.5` as neutral,
  while the binary logistic preference estimator declares that it drops ties. A
  feature-artifact diagnostic component reports
  density/L0/dead-feature/value-range health without relabeling numerical activity as
  semantic presence. Typed results now emit JSON-safe manifests with row-ID hashes and
  portable artifact metadata while keeping result tables as DataFrames. The built-in
  outcome component records grouping, normalization, estimand, inference,
  and BH families rather than returning an unqualified table.
- Centralized type-stable prompt-group factorization and made grouped length-confound
  screening use equal-group feature, outcome, and length summaries throughout. Elicitation
  alignment now rejects duplicate IDs and code/ID row-count mismatches.

- Added prompt-group-aware preference, elicitation, prompt-conditioned delta, cluster,
  and confound inference. Repeated prompts now receive equal total weight; artifacts name
  their estimand/test and report independent-group support. Unique-group inputs preserve
  the historical row-level estimands.
- Semantic-presence calibration now selects thresholds and confirms them on deterministic
  disjoint prompt-group splits. Only uniformly sampled confirmation groups above the fixed
  cutoff, a precision Wilson LCB, and a silent-leakage Wilson UCB can set
  `presence_pass=True`; phase-specific status, coverage, support, and audit evidence are
  persisted.
- Added reusable typed outcome analysis for binary, probability/preference, continuous,
  and multi-attribute continuous outcomes, with explicit normalization/missingness and
  row- or equal-group-weighted estimands. Thin independent support keeps descriptive
  effects but suppresses p/q claims. It is available from Python, `associate-outcomes`,
  and the config-driven `analyze` workflow.
- Completion/prompt lens builds, applied encoding bundles, and applied viewer-lens materialization now
  validate and publish clean whole directories transactionally. Manifests bind ordered retained metadata and source embeddings through
  `dataset_hash`; stale undeclared files are removed and failed rebuilds restore the prior
  lens.
- Hugging Face datasets and lenses now resolve mutable refs to immutable commit SHAs before
  loading. Prepared-data sidecars bind ordered canonical table content, loaded Hub lenses
  expose requested/resolved revisions without retaining tokens, and `analyze` resume state
  fingerprints the resolved commits.
- Added a manual opt-in real tiny-model smoke, a push/PR viewer integration smoke, and
  offline documentation-link and command-inventory checks in CI.

- Hardened artifact and alignment contracts: lens manifests now reject non-2-D,
  width-mismatched, row-misaligned, or metadata-misaligned code arrays; lens builds and
  applied-dataset encoding validate finite aligned embeddings and codes; embedding
  provenance mismatches fail closed; stale whitening transforms cannot leak into a
  rebuilt lens; and low-level embedding rejects unequal prompt/completion lists.
- Fixed token activation extraction duplicating the assistant generation header. Token
  activation caches now refuse non-empty destinations instead of truncating an existing
  cache, and token train/validation splits reject invalid or empty partitions.
- Hardened `prefscope analyze` resume and replacement behavior. Resume signatures now
  include content fingerprints for local datasets, lenses, and annotations. `--fresh`
  removes only a recognized PrefScope analysis directory and refuses unsafe,
  unrecognized, or input-containing paths. State writes are atomic.
- Winner-oriented `conditional-delta` and prompt-map exports now require an individual
  completion lens. They reject nonlinear direct-difference codes rather than incorrectly
  treating `-f(e_a-e_b)` as `f(e_b-e_a)`.
- Repeated-prompt paired comparisons now test the reported equal-group-weight mean with a
  conservative Hoeffding bound instead of attaching a group-sign/median test to that mean.
  Scope gates use `n_nonzero_groups` rather than raw repeated-row discordance.
- Documented `feature-relations`, the published-lens `analyze` workflow, prompt naming
  token/reasoning controls, separate positive/negative signed-prompt naming, and the
  current calibration/repeated-prompt inference limits without population-level claims.
- Added config-driven published-lens analysis through `prefscope analyze` and
  `run_analysis(...)`, including strict config validation and resumable stage outputs.
- Signed prompt axes can now be named one pole at a time. Prompt naming exposes its
  output-token budget, and OpenRouter reasoning can be disabled with
  `--reasoning-effort none`.
- Viewer example shards now include activation percentiles and deterministic samples
  from the strongest, random active, and near-cutoff regions. New CLI options control
  each response and prompt sample count while keeping per-language/source examples.
- Viewer exports with a prompt lens now include a complete prompt-feature atlas,
  prompt–prompt co-activation neighbors, and top-activating prompt examples for every
  axis. Partial prompt verification no longer drops named but unverified axes, and
  matched prompt/response evidence now supports single-response SFT corpora.
- Added `prefscope-export-viewer` as the unambiguous static-bundle export command;
  the existing `prefscope-viewer` command remains as a compatibility alias.
- Added `concept_distribution.json` and `coactivation.json` to the viewer bundle:
  per-concept prevalence with per-row concept counts and optional per-group fire
  rates, and concept pairs that co-fire above independence with example row indices.
  Both are computed in row chunks and sized by feature count, not corpus size.
- Viewer export now recognises single-response lenses and skips the model and
  preference artifacts (`diagnosis.json`, examples-by-model, report battles,
  head-to-head) instead of failing or writing empty files. Map exports and
  `conditional-delta` refuse contrast-only work with a message naming the missing
  codes rather than raising `FileNotFoundError`.
- Removed the unreachable `encode/shards.py`, `encode/build.py`, `core/sources.py`,
  the old unused representation implementation, and the adapter re-export shims. The
  later `core/representation.py` is a new public source contract. The old
  `representation` and `source` registry kinds no longer exist; the new registry kind is
  `representation_source`.
- `win-relevance` now reports a clear error on single-response data instead of raising
  `KeyError` on the missing second response.

- Added `select-lens`, which picks a width and sparsity from a `sae-metrics` sweep.
  Reconstruction improves monotonically with capacity and cannot select a
  configuration alone, so configurations are first screened for dead features,
  duplicated decoder directions, realised sparsity, and rows-per-feature — the last
  bounds width for document-embedding SAEs, which train on far fewer vectors than
  token-level ones. Reports the expansion ratio against the representation dimension.
- Fixed `inspect` failing on single-response corpora, which have no model columns;
  it now reports model counts only when those columns exist and states whether the
  data is paired.
- `interpret classify-role` now accepts single-response lenses. Evidence is scored on
  `z_a` alone, the paired-response block and its system-prompt sentence are omitted,
  and the CLI no longer requires `z_b`.
- Fixed `_select_evidence` returning one random example when `n_random=0`.
- Fixed `embed-corpus` failing on single-response corpora, which have no
  `completion_b` column.
- Added `NpyCache.get_many`, which reads each block file once. Bulk cache reads run
  in corpus order while blocks are written per worker, so key-at-a-time access
  reloaded the same blocks repeatedly.
- Grouped embedding-cache vectors into block files instead of one `.npy` per text,
  reducing a 110k-row corpus from ~220,000 cache files to ~860 at the default 256
  vectors per block. Existing per-text caches are still read, and concurrent writers
  sharing a cache directory need no locking. Callers that embed must now call
  `NpyCache.flush()` when finished; the bundled embedder does this already.
- Fixed the documented `prepare-dataset` → `build-lens` path, which could not load a
  prepared table at all: corpus loading required the full arena battle schema, so
  single-response SFT data (no `model_a`/`model_b`/`completion_b`) and even prepared
  paired data were rejected. Loading now requires only `prompt` and `completion_a`,
  synthesizing a content-hashed `battle_id` and defaulting `source`/`language`.
  Positional `item_id`/`row_id` are deliberately not reused as ids, because separate
  prepared tables both start at zero and would collide when concatenated.
- Fixed the installed `prefscope` command hanging after completing its work. Optional
  native dependencies leave non-daemon threads alive and CPython >= 3.13 parks those
  during finalization, so commands wrote every output file and then never exited —
  under a scheduler this consumed the entire walltime allocation. The console entry
  point is now `console_main`, which flushes and exits without finalizing; the
  importable `main` is unchanged.
- Split CLI argument registration into domain modules without changing commands or
  defaults, and documented one canonical dataset-analysis path plus the shorter
  published-lens application paths.


- Split the monolithic CLI into domain modules while retaining `python -m prefscope`
  and the installed `prefscope` entry point.
- Added `extract-concepts` / `extract_text_concepts` for direct prompt-response
  inference through local or Hugging Face lenses, with explicit semantic-presence
  provenance and clearer unavailable-accelerator errors.
- Added `init-demo`, making the documented smoke test reproducible from a wheel without
  repository-relative example files.
- Added `package-lens`, canonical inference-manifest migration, model-card support, and
  a documented Hub publishing workflow. Repackaging preserves original array provenance
  without declaring omitted corpus-aligned arrays.
- Removed project-specific scheduler, filesystem, and synchronization wrappers from the
  framework repository; deployment recipes now live outside the reusable library.
- Removed the premature `py.typed` marker until the full public surface is type-check clean;
  annotations remain available as documentation without overpromising PEP 561 support.

- Promoted the length-confound screen and static viewer exporter from loose scripts to
  installed commands (`prefscope screen-confounds` and `prefscope-viewer`), removed
  obsolete one-off scripts, and fixed strict
  parsing of persisted fidelity/significance flags so missing values or `"False"` are
  never treated as passing.
- Made shared SAE checkpoint loading portable to Apple MPS by validating on CPU before
  accelerator transfer; Hugging Face embedding inference now uses fp16 on MPS, and a
  runnable Hub-lens concept-extraction example shares one embedder across both lenses.
- Added calibrated, label-free `compare-responses` analysis for arbitrary paired response
  sets, with exact paired tests, repeated-prompt grouping, overlapping prompt contexts,
  response-scope classification, paired evidence, Python APIs, and viewer export.
- Added reusable local/Hugging Face dataset preparation with explicit column,
  structured-message, and preference-label mappings; added the registered
  `HuggingFaceDataset` Python adapter and direct win-relevance analysis of
  `encode-dataset` bundles.
- Added Hugging Face `Lens.from_pretrained` / `hf://` loading, self-contained
  annotation bundles, and streaming `prefscope concepts` export with raw
  activations and optional fidelity/calibration filters.
- Extended prompt→response elicitation to single-response `(prompt, completion)`
  artifacts and replaced dominant-prompt `argmax` conditioning with overlapping
  prompt-concept/cluster membership.
- Added non-negative `batchtopk-relu` SAE codes for presence-style individual and
  prompt lenses; `--sae-type auto` now keeps signed `batchtopk` for direct differences.
  Existing `batchtopk` checkpoints retain their historical signed behavior.
- Corrected JumpReLU to gate ReLU pre-activations, track firing statistics, expose
  sparsity warmup, and reject unsupported Matryoshka combinations.
- Switched SAE training to Adam without weight decay, corrected explained variance,
  calibrated BatchTopK deployment thresholds to target L0, and recorded deployment
  fit/sparsity/dead-feature metrics in schema-v2 manifests.
- Made Matryoshka training opt-in and added an explicit `--pole positive` safeguard
  for one-pole interpretation of legacy signed individual or prompt lenses.
- Added `cofire-leiden` clustering for interpreted feature poles, with positive
  co-presence affinities, mutual-kNN graphs, seed-stability and coherence diagnostics,
  safe singleton preservation, optional hierarchical superclusters, and mixed/abstaining
  LLM labels. Legacy `mi-leiden` remains available for artifact reproduction.

## 0.1.0 — 2026-07-16

Initial alpha release.

- Added the reusable `Lens` API for training, loading, saving, and encoding paired or
  homogeneous single-response post-training data.
- Added registered SAE, interpretation, verification, clustering, and lens-
  representation components with YAML/JSON pipeline configuration.
- Added held-out concept-name verification, multi-candidate naming, stratified judgment
  sampling, exact verification budgets, and independent naming/verifier LLM settings.
- Added preference relevance, model diagnosis, reporting, prompt/response maps, and
  viewer-bundle export.
- Added versioned lens manifests, torch-free base imports, optional accelerator and
  clustering dependencies, and restricted wheel/source-distribution contents.
- Added tutorials, extension guides, API/config/artifact references, and smoke/research
  example configurations.
