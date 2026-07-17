# Components reference

Every registered component, by kind. Select these by name in a config or CLI flag;
add your own with [the registry](../extending/the-registry.md). Introspect at runtime
with `registry.available(kind)` after `import prefscope.adapters`.

## interpreter — name each feature
Base: `NameStrategy` (`prefscope/interpret/strategy.py`). Shared `__init__` params:
`features, n_active, n_zero, verify_frac, seed, abbreviate, concurrency, debug_dir,
negatives, n_candidates, candidate_pool_factor`.
Select: `--name-mode` / `interpreter: {name: …}`.

| name | class | uses | notes |
|------|-------|------|-------|
| `pairwise` | `PairwiseNameStrategy` | `z_diff` | default for difference lenses |
| `individual` | `IndividualNameStrategy` | `z_a`, optional `z_b` | default for individual lenses (paired or single) |
| `single-text` | `SingleTextNameStrategy` | `z_prompt`, `prompts` | prompt lenses |

## verifier — check a name is real
Base: `VerifyStrategy`. Shared `__init__` params (`_VOPT`): `n_per_bucket,
verify_frac, seed, fidelity_threshold, concurrency, negatives, embeddings,
min_success_rate, min_bucket, sampling, n_examples`.
Select: `--verify-mode` / `verifier: {name: …}`. Output must include `feature_id,
concept, correlation, fidelity_pass`.

| name | class | for |
|------|-------|-----|
| `pairwise` | `PairwiseVerifyStrategy` | difference lenses |
| `individual` | `IndividualVerifyStrategy` | individual lenses |
| `prompt` | `PromptVerifyStrategy` | prompt lenses |

## clusterer — group co-firing features
Base: `Clusterer` (`prefscope/pipeline/cluster.py`); no shared `__init__` (each
absorbs extras via `**_`). Output: `[feature_id, cluster_id]`. Select: `--method` /
`clusterer: {name: …}`. Reserved control keys (not constructor params):
`cluster_on, fidelity_only, name_clusters, concurrency`.

| name | params | method |
|------|--------|--------|
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
| `batchtopk` | `--m-total, --k, --matryoshka-prefix` | default; signed codes, any `--input-rep` |
| `jumprelu` | `--sparsity-coef, --bandwidth` | learned per-feature thresholds, L0 penalty; one-sided codes — use `--input-rep individual` |
| `simple-topk` | `--m-total, --k` | training-time ablation; deployable as a frozen lens (selects top-`K` per example at inference) |

## dataset — adapt your data into `PairItem`s
Base: `Dataset` (`prefscope/core/dataset.py`). Used **programmatically** (not
name-selected by the build CLI) — see [bring your own dataset](../how-to/bring-your-own-dataset.md).

| name | class | source |
|------|-------|--------|
| `table` | `CsvDataset` | DataFrame / CSV / parquet with column mapping |
| `openjury` | `OpenJuryDataset` | OpenJury annotation JSON |

## negative_sampler — pick "silent" items for fidelity
Used inside verification. Select: `--negatives`.

| name | picks |
|------|-------|
| `random` | random non-activating items |
| `close` / `similar` | nearest non-activating items to the active centroid (needs embeddings/codes) |

## Legacy (registered, not wired)
`representation` (`identity, diff, concat, both`) and `source` were consumed by a
removed build facade. They remain registered but the live pipeline does not use
them; use `lens_rep` for the representation seam.
