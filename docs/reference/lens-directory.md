# Lens Directory Reference

A **lens** is a frozen SAE directory. Build-time contents depend on `input_rep`;
analysis stages add more files later. Completion and prompt builds publish transactionally:
they validate a clean sibling staging directory and then replace the destination as one
unit. A failed build or publish restores the prior directory, and undeclared stale files
cannot survive a rebuild.

A shareable/pretrained lens should bundle the checkpoint, manifest, and the
interpretation tables consumers need. `Lens.save(dest, annotations=DIR,
inference_only=True)` assembles a compact directory by staged whole-directory replacement, excluding
corpus-aligned codes and text; `Lens.from_pretrained("owner/repository")`
downloads it from the Hugging Face Hub and invokes the same local loader.

Sources of truth:
- `prefscope/pipeline/build_lens.py` (`_train_and_save`, `build_prompt_lens`)
- `prefscope/artifacts.py` (filename constants)
- `prefscope/pipeline/lens_rep.py` (`output_arrays` per rep)
- Analysis writers: `prefscope/cli/`, `prefscope/pipeline/run.py`,
  `prefscope/interpret/{name,verify}.py`, `prefscope/pipeline/{cluster,winrelevance}.py`

Canonical filenames (`artifacts.py`):
`manifest.json`, `sae_model.pt`, `battles.parquet`, `z_diff.npy`, `z_a.npy`,
`z_b.npy`, `z_prompt.npy`, `feature_names.csv`, `feature_fidelity.csv`,
`feature_roles.csv`, `feature_calibration.csv`, `feature_context.csv`,
`model_feature_context.parquet`, `feature_clusters.csv`, `win_relevance.csv`, and the
`prompt_feature_*` variants.

## Build-time contents

These are written by the lens build (per `input_rep`):

| file | written by | `difference` | `individual` | `prompt` | contents / shape |
|------|-----------|:---:|:---:|:---:|------------------|
| `sae_model.pt` | build | ✔ | ✔ | ✔ | torch checkpoint `{state_dict, config}` |
| `manifest.json` | build | ✔ | ✔ | ✔ | lens metadata (schema below) |
| `battles.parquet` | build | ✔ | ✔ | ✔ | per-row meta, aligned to the `z_*` arrays |
| `sae_training_log.csv` | build | ✔ | ✔ | ✔ | one row per epoch (training log) |
| `z_diff.npy` | build (`output_arrays`) | ✔ | paired only | — | `(N, M)` contrast codes |
| `z_a.npy` | build (`output_arrays`) | — | ✔ | — | `(N, M)` `f(e_a)` codes |
| `z_b.npy` | build (`output_arrays`) | — | paired only | — | `(N, M)` `f(e_b)` codes |
| `z_prompt.npy` | build (`output_arrays`) | — | — | ✔ | `(N, M)` prompt codes |
| `whiten.npz` | build (only if `--whiten != none`) | optional | optional | — | saved whitening transform, re-applied at projection |

`output_arrays` per rep (`lens_rep.py`):
- `difference` → `{z_diff = project(e_a − e_b)}`
- `individual`, paired → `{z_a = f(e_a), z_b = f(e_b), z_diff = z_a − z_b}` (note
  `f(e_a) − f(e_b) ≠ f(e_a − e_b)`); single-response → `{z_a = f(e_a)}`
- `prompt` → `{z_prompt}` (written directly by `build_prompt_lens`, which does
  not route through `LensRep`)

N = number of battles (or prompts); M = `m_total`. Loading validates this as a
strict alignment contract: every declared code array must be exactly two-dimensional,
have width `M`, share the same `N`, match any recorded `array_shapes` and `n_battles`,
and have the same row count as `battles.parquet` when it is bundled. A malformed or
partially overwritten lens is rejected rather than loaded.

### `battles.parquet` columns
Subset of the source columns that exist. For paired completion lenses
(`_META_COLS`): `instruction_id`, `group_id`, `model_a`, `model_b`, `y_judge`,
`lang`, `source`, `language`. For the prompt lens: `battle_id`, `instruction_id`,
`group_id`, `model_a`, `model_b`, `source`, `language`, `human_pref`. (Only columns present in
the input are kept.) A single-response lens also retains `prompt` and `completion_a`
so it can be interpreted without a pair-corpus schema. Row order is aligned to the
`z_*` arrays.

## Analysis-stage outputs (added later)

Written into the lens dir (or any `--out` path) by the named stage. Prompt-lens
runs produce the `prompt_feature_*`-prefixed variants with the same columns.

| file | written by | key columns |
|------|-----------|-------------|
| `feature_names.csv` | `interpret name` / `run` (name) | `feature_id`, `concept`, `concept_abbrev`, `n_active`, `n_zero`, `n_candidates`, `candidate_concepts`; individual names also record `reviewed_concept`, naming support/control counts, `naming_screen_pass`, `naming_review_action`, and compatibility `naming_audit_*` diagnostics. |
| `feature_fidelity.csv` | `interpret verify` / `run` (verify) | `feature_id`, `concept`, `n`, `agreement`, `precision`, `recall`, `f1`, `correlation`, `sign`, `p_value`, `p_bonferroni`, `fidelity_pass` (single-text verifier additionally has `fp_rate`) |
| `feature_calibration.csv` | `interpret calibrate-presence` | disjoint `selection_status` / `confirmation_status`, fixed `semantic_threshold`, phase-specific precision/LCB and coverage, independent group/sample counts, silent-leakage rate and Wilson UCB, audit samples, and confirmatory-only `presence_pass` |
| `feature_context.csv` | `context-profile` | semantic prevalence, prompt dependence/breadth, paired choice ratio, `behavior_category` |
| `model_feature_context.parquet` | `context-profile` | per model-feature discordant choice effect, exact-test q-value, context consistency, stability, category |
| `feature_relations.csv` | `feature-relations` | conditional feature pairs with support, effect/lift, uncertainty, adjusted significance, and relation label |
| `feature_clusters.csv` | `cluster-features` / `run` (cluster) | one row per feature: `feature_id`, canonical fine `cluster_id`, optional `supercluster_id`, `concept`, and `behavior` |
| `feature_clusters_summary.csv` | same (sibling, `<out>_summary.csv`) | per-community size, neutral/LLM label, central/diverse representatives, full members, and within/external coherence diagnostics |
| `feature_clusters_diagnostics.csv` | same (sibling, `<out>_diagnostics.csv`) | run parameters, cluster-size distribution, and cross-seed adjusted-Rand stability |
| `win_relevance.csv` | `win-relevance` / `run` (win-relevance) | row support and descriptive rates plus `estimand`, test names, `n_independent_groups` / firing-group support, group-valid adjusted significance, and separately named length-controlled `delta_win_*` estimand/test/support fields |
| `<win_relevance>_clusters.csv` | `win-relevance --clusters` | cluster-level effect plus estimand, test, independent-group support, and adjusted significance |
| `<outcome>.csv` + `<outcome-stem>_outcomes.json` | `associate-outcomes` | long-form feature × outcome descriptive association, missingness/unit counts, estimand/test/q-value, and normalization provenance |
| `prompt_feature_names.csv` | `interpret name --lens-kind prompt` / `name-prompts` alias / prompt `run` (name) | `feature_id`, `concept`, `status`, `confidence`, `n_active`, `n_candidates`, `candidate_concepts`, `fire_rate` |
| `prompt_feature_fidelity.csv` | prompt `verify` | same as `feature_fidelity.csv` |
| `prompt_feature_clusters.csv` | prompt `cluster` | `feature_id`, `cluster_id` (+ `behavior`) |
| `llm_usage.json` | config-driven `run` | current-run requests, retries, tokens, cached/reasoning tokens, exact reported cost, and stage/model breakdowns |
| `llm_usage.jsonl` | config-driven `run` | append-only per-response/error accounting that survives interrupted runs; never contains prompt or response text |

Standalone LLM commands put equivalent `<output-stem>.usage.json` and
`<output-stem>.usage.jsonl` files beside their requested output. Resumable
`interpret name` / `interpret verify` / `interpret calibrate-presence` runs also write `<output-stem>.resume.json`,
which guards the per-feature checkpoint against incompatible settings.

`fidelity_pass` = `correlation >= fidelity_threshold` AND `p_bonferroni < 0.05`.
`significant` (win-relevance) = `p_bonferroni < 0.05`.

For a release artifact, include at least:

```text
manifest.json
sae_model.pt
feature_names.csv
feature_fidelity.csv
feature_calibration.csv       # when semantic-presence filtering is advertised
feature_context.csv           # when behavior categories are advertised
feature_clusters.csv          # when cluster labels are advertised
whiten.npz                    # only when declared by the manifest
```

Prompt lenses use the `prompt_feature_names.csv`,
`prompt_feature_fidelity.csv`, and `prompt_feature_clusters.csv` variants.
Training arrays and the original corpus are not needed for frozen inference.

## `manifest.json` schema

### Completion lens (`difference` / `individual`)
Written by `build_lens._train_and_save`:

| key | type | meaning |
|-----|------|---------|
| `schema_version` | int | manifest schema; new artifacts use v2 |
| `lens_kind` | `difference` \| `individual` | serialized lens type; `completion` is only the pipeline routing label |
| `array_shapes` | mapping | exact serialized shape for every declared output array |
| `n_battles` | int | total battles |
| `dataset_hash` | str | SHA-256 of ordered retained canonical metadata rows plus canonical little-endian float32 source embeddings; row-order-sensitive |
| `n_items` | int | total input rows (same value, clearer for single data) |
| `dataset_mode` | `paired` \| `single` | whether a B response exists |
| `n_train_battles` | int | training split size |
| `n_train_rows_used` | int | rows used after an optional training-row cap |
| `n_val_battles` | int | validation split size |
| `m_total` | int | SAE feature count M |
| `k` | int | top-k active per row |
| `sae_type` | str | resolved architecture (`batchtopk`, `batchtopk-relu`, `jumprelu`, …) |
| `activation_polarity` | `signed` \| `nonnegative` | numerical code polarity |
| `code_semantics` | `axis` \| `presence` | how an activation should be interpreted |
| `selection_rule` | str | frozen gating/selection rule |
| `input_dim` | int | embedding dim D the lens expects |
| `embed_model_id` / `embed_model_revision` | str \| null | embedder and pinned revision |
| `max_tokens`, `embed_instruction`, `pooling`, `normalization`, `dtype`, `backend` | preprocessing provenance | exact contract used to create embeddings; legacy nulls trigger a warning |
| `best_val_norm_mse` | float \| null | best validation normalized MSE (null if never finite) |
| `best_val_select_norm_mse` | float \| null | model-selection NMSE before final deployment calibration |
| `best_val_explained_variance` | float \| null | `1 - best_val_norm_mse` |
| `deployment_val_norm_mse` | float \| null | validation NMSE through the final frozen inference gate |
| `deployment_val_explained_variance` | float \| null | deployment `1 - NMSE` |
| `deployment_val_active` | float \| null | deployed mean L0 |
| `threshold_calibration_rows` | int \| null | rows used to calibrate the frozen deployment threshold |
| `deployment_dead_neurons` / `deployment_rare_neurons` | int \| null | validation firing health |
| `target_l0` / `calibration_l0` | number \| null | requested and achieved BatchTopK deployment sparsity |
| `optimizer` / `weight_decay` / `seed` | training provenance | optimizer configuration and RNG seed |
| `matryoshka_prefix_lengths` | list[int] | Matryoshka prefixes; `[]` means explicitly disabled |
| `n_epochs_trained` | int | epochs actually trained |
| `input_rep` | `difference` \| `individual` | SAE input representation |
| `whiten` | `none` \| `standardize` \| `pca` | input whitening method |
| `output_arrays` | list[str] | which `z_*` arrays were saved (e.g. `["z_diff"]` or `["z_a","z_b","z_diff"]`) |

### Prompt lens (`prompt`)
Written by `build_prompt_lens`:

| key | type | meaning |
|-----|------|---------|
| `schema_version` | int | manifest schema; new artifacts use v2 |
| `lens_kind` | `prompt` | explicit lens role |
| `array_shapes` | mapping | exact `z_prompt.npy` shape |
| `n_prompts` | int | total prompts |
| `dataset_hash` | str | SHA-256 of ordered prompt metadata plus canonical little-endian float32 prompt embeddings |
| `n_train` | int | training split size |
| `n_val` | int | validation split size |
| `m_total` | int | feature count M |
| `k` | int | top-k active |
| `sae_type`, `activation_polarity`, `code_semantics`, `selection_rule` | str | same explicit SAE semantics as completion lenses |
| `input_dim` | int | embedding dim D |
| `embed_model_id` / `embed_model_revision` | str \| null | embedder and pinned revision used for inference |
| `max_tokens`, `embed_instruction`, `pooling`, `normalization`, `dtype`, `backend` | preprocessing provenance | exact prompt embedding contract |
| `best_val_norm_mse`, `best_val_select_norm_mse`, `best_val_explained_variance` | float \| null | model-selection validation metrics |
| `deployment_val_norm_mse`, `deployment_val_explained_variance`, `deployment_val_active` | float \| null | metrics for the final frozen inference rule |
| `threshold_calibration_rows`, `deployment_dead_neurons`, `deployment_rare_neurons` | int \| null | calibration support and feature health |
| `target_l0`, `calibration_l0` | number \| null | requested and achieved deployment sparsity |
| `optimizer`, `weight_decay`, `seed` | training provenance | optimizer settings and random seed |
| `n_train_rows_used` | int | rows used after an optional training-row cap |
| `matryoshka_prefix_lengths` | list[int] | Matryoshka prefixes; `[]` means disabled |
| `n_epochs_trained` | int | epochs trained |
| `input_rep` | `"prompt"` | fixed |
| `output_arrays` | `["z_prompt"]` | fixed |

(The prompt-lens manifest records `whiten: null`; prompt whitening is not currently
exposed.)

## Applied encoded feature bundle

`prefscope encode-dataset` writes a separate schema-v1 directory containing
`manifest.json`, `meta.parquet`, `battles.parquet`, and one or more declared `z_*.npy`
arrays. This is not a trained lens directory. Its manifest records row/feature counts,
array shapes, dataset hash, source-lens manifest digest, lens input representation,
activation polarity, code semantics, selection rule, and portable embedding provenance.

Load it without Torch:

```python
from prefscope import load_feature_batch

features = load_feature_batch("analysis/codes")
z_diff = features.matrix("z_diff")
```

The loader rejects duplicate IDs, row/shape drift, missing arrays, non-canonical
float32/non-finite codes, undeclared views, metadata-twin drift, extra files, and a
dataset-hash mismatch across ordered metadata and all arrays. Returned matrices explicitly distinguish `absolute_a`, `absolute_b`,
`a_minus_b`, and prompt orientation.

### `sae_model.pt`
`torch.save({"state_dict": ..., "config": ...})`. The state dict carries
`encoder.weight` `(M, D)`, `input_bias` `(D,)`, `neuron_bias` `(M,)`,
`threshold` (scalar BatchTopK gate; JumpReLU additionally has per-feature
`log_threshold`), `decoder.weight` `(D, M)` — the parameters
`SAEProjector` reads (`prefscope/encode/sae.py`).
