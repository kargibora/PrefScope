# Components reference

This page lists the built-in components by kind. Select a component by name in a config
or construct it through the Python registry. See [The registry](../extending/the-registry.md)
to add one. Registry lists include only modules imported in the current Python process;
the CLI imports the built-ins it needs.


## representation_source — produce aligned vectors

Base: `RepresentationSource` (`prefscope/core/representation.py`). Sources return a
validated `RepresentationBatch` with named numerical arrays, unique row IDs, aligned
scalar metadata, and portable provenance without credentials or absolute local paths.

| name | class | output |
|---|---|---|
| `text-embedding` | `EmbeddingRepresentationSource` | `prompt`, `response_a`, optional `response_b` |
| `precomputed` | `PrecomputedRepresentationSource` | exact-ID-aligned static embeddings or already pooled residual arrays |

Custom residual or embedding sources are programmatic and import-driven today. See
[Add a representation source](../extending/add-a-representation-source.md).

A pretrained SAELens checkpoint can replace PrefScope's trained projector through
`Lens.from_saelens(...)`. This experimental path consumes exact-hook token activations,
applies the SAE before max pooling, and does not treat them as ordinary text embeddings;
it is documented in
[Use a pretrained SAE through SAELens](../how-to/use-saelens.md).

## analysis_component — reusable typed analysis tasks

Base: `AnalysisComponent` (`prefscope/api/analysis.py`). Components consume an
`AnalysisDataset` and return one `AnalysisArtifact` that names its estimand and
metadata.

| name | class | output |
|---|---|---|
| `outcome-associations` | `OutcomeAssociations` | descriptive feature/outcome table with explicit analysis unit, normalization, test, and BH family |
| `paired-concept-shift` | `PairedConceptShift` | B-minus-A calibrated semantic-presence shift for aligned response sets |
| `paired-outcome-shifts` | `PairedOutcomeShifts` | raw-scale B-minus-A outcome changes over aligned rows or equal-weight independent groups |
| `prompt-conditioned-outcome-shifts` | `PromptConditionedOutcomeShifts` | actual present-minus-absent heterogeneity in paired B-minus-A changes, gated on calibrated prompt presence |
| `preference-length-confounds` | `PreferenceLengthConfounds` | sensitivity screen that requires aligned A-minus-B features, length differences, and P(A preferred); not a bias classifier |
| `feature-artifact-diagnostics` | `FeatureArtifactDiagnostics` | deterministic density/L0/dead-feature/value-range summary; numerical activity is not semantic presence |

Pass component instances through `AnalysisPlan`, or resolve registered names with
`AnalysisPlan.from_names(...)` after importing their module. Each built-in result carries
a versioned `TableContract`. It validates required columns, logical types, unique keys,
direction, and units without changing the pandas DataFrame. Custom components can attach
their own contract or remain uncontracted.

## interpreter — name each feature
Base: `NameStrategy` (`prefscope/interpret/strategy.py`). Shared `__init__` params:
`features, n_active, n_zero, verify_frac, seed, abbreviate, concurrency, debug_dir,
negatives, n_candidates, candidate_pool_factor, pole`.
Select: `--name-mode` / `interpreter: {name: …}`.
The CLI additionally injects an internal `on_result` callback for per-feature checkpoints.

| name | class | uses | notes |
|------|-------|------|-------|
| `pairwise` | `PairwiseNameStrategy` | `z_diff` | default for difference lenses |
| `individual` | `IndividualNameStrategy` | `z_a`, optional `z_b` | default for individual lenses (paired or single) |
| `single-text` | `SingleTextNameStrategy` | `z_prompt`, `prompts` | prompt lenses |

`individual` ranks responses by feature activation but displays at most one A/B completion
per instruction. A second LLM call reviews and may refine each proposal on the same naming
evidence. Its support vectors feed a cheap triage screen (strict majority among activators
and enrichment over controls); they are diagnostics, not held-out proof. Multi-candidate
synthesis receives a final union review. The disjoint verifier determines fidelity.

## verifier — check a name is real
Base: `VerifyStrategy`. Shared `__init__` params (`_VOPT`): `n_per_bucket,
verify_frac, seed, fidelity_threshold, concurrency, negatives, embeddings,
min_success_rate, min_bucket, sampling, n_examples, pole`.
Select: `--verify-mode` / `verifier: {name: …}`. Output must include `feature_id,
concept, correlation, fidelity_pass`.
The resumable CLI additionally injects unfinished `features` and an internal `on_result`
callback.

| name | class | for |
|------|-------|-----|
| `pairwise` | `PairwiseVerifyStrategy` | difference lenses |
| `individual` | `IndividualVerifyStrategy` | individual lenses |
| `prompt` | `PromptVerifyStrategy` | prompt lenses |

The individual verifier likewise samples at most one response per instruction, so paired
A/B answers to one prompt cannot inflate its effective sample size or p-value.

## clusterer — group co-firing features
Base: `Clusterer` (`prefscope/pipeline/cluster.py`); no shared `__init__` (each
absorbs extras via `**_`). Output: `[feature_id, cluster_id]`. Select: `--method` /
`clusterer: {name: …}`. Reserved control keys (not constructor params):
`cluster_on, fidelity_only, name_clusters, concurrency`.

| name | params | method |
|------|--------|--------|
| `cofire-leiden` | `resolution, knn, affinity_metric, pole, min_cooccur, knn_mode, min_cluster_size, small_community_policy, stability_runs, super_resolution, super_knn, seed` | positive-pole co-firing graph + Leiden; preferred for interpreted behavior features |
| `mi-leiden` | `resolution, knn, min_cluster_size, seed` | MI co-firing graph + Leiden; count emerges |
| `spherical-kmeans` | `n_clusters, seed` | cosine k-means on activation columns |
| `agglomerative` | `n_clusters, seed` | average-linkage on `1−|corr|` |

## lens_rep — how A/B form the SAE input + codes
Base: `LensRep` (`prefscope/pipeline/lens_rep.py`). Select: `--input-rep` (CLI
`choices` are `difference`/`individual`) → recorded in the manifest.

| name | training input | saved codes |
|------|----------------|-------------|
| `difference` | `e_a − e_b` | `z_diff` |
| `individual` | pooled `[e_a; e_b]`, or `e_a` for single data | paired: `z_a`, `z_b`, `z_diff`; single: `z_a` |
| `prompt` | prompt embeddings | `z_prompt` (built via `build-prompt-lens`) |

## sae — the autoencoder architecture
Base: `BatchTopKSAE` (`prefscope/sae/model.py`), a `torch.nn.Module`. Select:
`--sae-type` (recorded in the manifest). See [add an SAE](../extending/add-an-sae.md).

| name | params | notes |
|------|--------|-------|
| `batchtopk` | `--m-total, --k, --matryoshka-prefix` | legacy-compatible signed BatchTopK; auto default for `difference` |
| `signed-batchtopk` | same | public alias of `batchtopk` |
| `batchtopk-relu` | same | non-negative presence codes; auto default for `individual` and prompt lenses |
| `jumprelu` | `--sparsity-coef, --bandwidth, --sparsity-warmup-steps` | learned per-feature thresholds and L0 penalty; non-negative; Matryoshka unsupported |
| `simple-topk` | `--m-total, --k` | training-time ablation; deployable as a frozen lens (selects top-`K` per example at inference) |

The public `--sae-type auto` is resolved from the representation and the resolved type,
activation polarity (`signed`/`nonnegative`), and code semantics (`axis`/`presence`) are
recorded in the checkpoint and manifest. Naming a legacy signed individual/prompt lens
requires `pole: positive` (CLI: `--pole positive`) so a one-pole label is not presented
as a complete axis interpretation.

## dataset — adapt your data into `PairItem`s
Base: `Dataset` (`prefscope/core/dataset.py`). Adapters are usable from Python;
the same mapping contract is exposed by `prefscope prepare-dataset` — see
[bring your own dataset](../how-to/bring-your-own-dataset.md).

| name | class | source |
|------|-------|--------|
| `table` | `TableDataset` (`CsvDataset` alias) | DataFrame / CSV / parquet / JSON with column mapping |
| `huggingface` | `HuggingFaceDataset` | Hub dataset configuration/split, optional bounded streaming |
| `openjury` | `OpenJuryDataset` | OpenJury annotation JSON |

## negative_sampler — pick "silent" items for fidelity
Used inside verification. Select: `--negatives`.

| name | picks |
|------|-------|
| `random` | random non-activating items |
| `close` / `similar` | nearest non-activating items to the active centroid (needs embeddings/codes) |

## Lens representation boundary

`lens_rep` has a registry implementation seam, but the current manifest and CLI
support only the three built-in policies. It is not a general artifact plug-in
contract. `RepresentationSource` is the supported public seam for changing the
dense vector producer.
