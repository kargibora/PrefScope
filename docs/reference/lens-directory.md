# Lens Directory Reference

A **lens** is a frozen SAE directory. Build-time contents depend on `input_rep`;
analysis stages add more files later.

Sources of truth:
- `prefscope/pipeline/build_lens.py` (`_train_and_save`, `build_prompt_lens`)
- `prefscope/artifacts.py` (filename constants)
- `prefscope/pipeline/lens_rep.py` (`output_arrays` per rep)
- Analysis writers: `prefscope/__main__.py`, `prefscope/pipeline/run.py`,
  `prefscope/interpret/{name,verify}.py`, `prefscope/pipeline/{cluster,winrelevance}.py`

Canonical filenames (`artifacts.py`):
`manifest.json`, `sae_model.pt`, `battles.parquet`, `z_diff.npy`, `z_a.npy`,
`z_b.npy`, `z_prompt.npy`, `feature_names.csv`, `feature_fidelity.csv`,
`feature_clusters.csv`, `win_relevance.csv`, and the `prompt_feature_*` variants.

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

N = number of battles (or prompts); M = `m_total`.

### `battles.parquet` columns
Subset of the source columns that exist. For paired completion lenses
(`_META_COLS`): `instruction_id`, `model_a`, `model_b`, `y_judge`, `lang`,
`source`, `language`. For the prompt lens: `battle_id`, `instruction_id`,
`model_a`, `model_b`, `source`, `language`, `human_pref`. (Only columns present in
the input are kept.) A single-response lens also retains `prompt` and `completion_a`
so it can be interpreted without a pair-corpus schema. Row order is aligned to the
`z_*` arrays.

## Analysis-stage outputs (added later)

Written into the lens dir (or any `--out` path) by the named stage. Prompt-lens
runs produce the `prompt_feature_*`-prefixed variants with the same columns.

| file | written by | key columns |
|------|-----------|-------------|
| `feature_names.csv` | `interpret name` / `run` (name) | `feature_id`, `concept`, `concept_abbrev`, `n_active`, `n_zero`, `n_candidates`, `candidate_concepts` (pairwise/individual). Prompt names use the same candidate provenance fields. |
| `feature_fidelity.csv` | `interpret verify` / `run` (verify) | `feature_id`, `concept`, `n`, `agreement`, `precision`, `recall`, `f1`, `correlation`, `sign`, `p_value`, `p_bonferroni`, `fidelity_pass` (single-text verifier additionally has `fp_rate`) |
| `feature_clusters.csv` | `cluster-features` / `run` (cluster) | `feature_id`, `cluster_id`, plus `concept` (if names given) and `behavior` |
| `feature_clusters_summary.csv` | same (sibling, `<out>_summary.csv`) | `cluster_id`, `n_features`, `n_verified`, `behavior`, `feature_ids`, `member_concepts` |
| `win_relevance.csv` | `win-relevance` / `run` (win-relevance) | `feature_id` (+ `concept` if names), `n_fire`, `fire_rate`, `win_rate_a_more`, `win_rate_a_less`, `win_assoc`, `correlation`, `p_value`, `p_bonferroni`, `sign`, `significant`, plus length-controlled `beta`, `delta_win_rate`, `lr_p`, `delta_win_p_bonferroni`, `delta_win_significant` |
| `<win_relevance>_clusters.csv` | `win-relevance --clusters` | `cluster_id`, `behavior` (if present), `n_features`, `beta`, `delta_win_rate`, `lr_p`, `delta_win_p_bonferroni`, `delta_win_significant` |
| `prompt_feature_names.csv` | `name-prompts` / prompt `run` (name) | `feature_id`, `concept` |
| `prompt_feature_fidelity.csv` | prompt `verify` | same as `feature_fidelity.csv` |
| `prompt_feature_clusters.csv` | prompt `cluster` | `feature_id`, `cluster_id` (+ `behavior`) |

`fidelity_pass` = `correlation >= fidelity_threshold` AND `p_bonferroni < 0.05`.
`significant` (win-relevance) = `p_bonferroni < 0.05`.

## `manifest.json` schema

### Completion lens (`difference` / `individual`)
Written by `build_lens._train_and_save`:

| key | type | meaning |
|-----|------|---------|
| `n_battles` | int | total battles |
| `n_items` | int | total input rows (same value, clearer for single data) |
| `dataset_mode` | `paired` \| `single` | whether a B response exists |
| `n_train_battles` | int | training split size |
| `n_val_battles` | int | validation split size |
| `m_total` | int | SAE feature count M |
| `k` | int | top-k active per row |
| `input_dim` | int | embedding dim D the lens expects |
| `embed_model_id` | str \| null | embedder this lens was built with |
| `best_val_norm_mse` | float \| null | best validation normalized MSE (null if never finite) |
| `matryoshka_prefix_lengths` | list[int] | Matryoshka prefix lengths used |
| `n_epochs_trained` | int | epochs actually trained |
| `input_rep` | `difference` \| `individual` | SAE input representation |
| `whiten` | `none` \| `standardize` \| `pca` | input whitening method |
| `output_arrays` | list[str] | which `z_*` arrays were saved (e.g. `["z_diff"]` or `["z_a","z_b","z_diff"]`) |

### Prompt lens (`prompt`)
Written by `build_prompt_lens`:

| key | type | meaning |
|-----|------|---------|
| `n_prompts` | int | total prompts |
| `n_train` | int | training split size |
| `n_val` | int | validation split size |
| `m_total` | int | feature count M |
| `k` | int | top-k active |
| `input_dim` | int | embedding dim D |
| `embed_model_id` | str \| null | label only (recorded, not enforced) |
| `best_val_norm_mse` | float \| null | best validation normalized MSE |
| `matryoshka_prefix_lengths` | list[int] | Matryoshka prefix lengths |
| `n_epochs_trained` | int | epochs trained |
| `input_rep` | `"prompt"` | fixed |
| `output_arrays` | `["z_prompt"]` | fixed |

(The prompt-lens manifest has no `whiten` key.)

### `sae_model.pt`
`torch.save({"state_dict": ..., "config": ...})`. The state dict carries
`encoder.weight` `(M, D)`, `input_bias` `(D,)`, `neuron_bias` `(M,)`,
`threshold` (scalar), `decoder.weight` `(D, M)` — the parameters
`SAEProjector` reads (`prefscope/encode/sae.py`).
