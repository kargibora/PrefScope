# Viewer Bundle Reference

The **viewer bundle** is the directory of static JSON that the `prefscope.viewer_export`
CLI writes (to `viewer-data/` by default) for static visualization clients. Browsers
cannot read `.npy`/`.parquet`, so every artifact is flattened JSON. The export format
is self-contained and does not depend on a particular viewer repository or directory.

Sources of truth:
- `prefscope/viewer_export/` — the export implementation
  (`sanitize` / `features` / `diagnosis` / `examples` / `tables` / `maps` / `cli`)
- `bundle_manifest.json` — schema version, completed files, and processing errors

## bundle_manifest.json

Written **last**, so it only ever describes a completed run. The viewer loads it
first and treats any file **not listed** as absent — a stale artifact from an
older export cannot masquerade as current data. A missing manifest (legacy
bundle) or a `schema_version` mismatch surfaces as a banner in the viewer.

```json
{
  "schema_version": 2,
  "generated_at": "2026-07-02T09:00:00+00:00",
  "lens": "indiv_m2048_k64",
  "files": ["features.json", "meta.json", "examples/", "..."],
  "errors": [{"stage": "report_battles", "error": "..."}]
}
```

- `files` — artifact names written this run; the shard directory is recorded as
  the single entry `"examples/"`.
- `errors` — stages that failed during export (the corresponding panel shows
  partial or no data). An entry here distinguishes "processing failed" from
  "input legitimately absent".
- `schema_version` is emitted from `BUNDLE_SCHEMA_VERSION` in
  `prefscope.viewer_export.cli`.

## Artifacts

| file | producer (`viewer_export/`) | contents | absent when |
|------|----------------------------|----------|-------------|
| `meta.json` | `features.export_meta` | headline numbers: EV, n_verified/n_named, `r2`/`is_loo`/`loo_r2`, `has_preference`, M/K/dim, n_battles | never (always written) |
| `features.json` | `features.export_features` | per-feature concept, fidelity verdict (F1/precision/recall/fp_rate/corr/agreement/n), win-relevance (`win_assoc`, length-controlled `delta_win_rate` + significance), fire rate (`generality`), `n_prompt_types`, cluster | never |
| `validation.json` | `cli.main` (from `diagnosis_validation.csv`) | per-model predicted vs actual win rate (`predicted_score_loo` when LOO ran) | no validation CSV (e.g. label-free dataset) |
| `diagnosis.json` | `diagnosis.export_diagnosis` | per-model `net_direction`, `delta_vs_pool`, `fire_rate`, raw counts `fire_pos`/`fire_neg` + pool totals `tot_pos`/`tot_neg`/`n_total` (for the client-side vs-pool z-test), `prompt_types`, `relations` | no oriented bank (an honest `{"error": "no_bank"}` stub is written instead) |
| `examples/<fid>.json` | `examples.export_examples` | top-activating battles for feature `fid` — **sharded per named feature** (verified or not) so the viewer lazy-loads only the feature being viewed; stale shards are cleared each run | no `--corpus` |
| `examples_by_model.json` | `examples.export_examples_by_model` | per (model × feature): that model's OWN answers exhibiting the feature, with outcome | flag off, no corpus, or no per-side codes |
| `report_battles.json` | `examples.export_report_battles` | per (model × prompt-concept) sample battles for the report-card drill-in | flag off, no labels, or no prompt lens |
| `head_to_head.json` | `diagnosis.export_head_to_head` | per model pair: discordant counts `bpos`/`cpos` per feature (viewer runs McNemar + BH) | flag off, difference lens, or misaligned codes (refused) |
| `bias_screen.json` | `tables.export_bias_screen` | per-feature length-confound screen (win assoc, length covariance, residual) | no bias-screen CSV |
| `conditional.json` | `tables.export_conditional` | `{raw, clustered}` δ_{f,k} cells: Δwin within prompt type, significance, `n` (type size) and `nf` (battles where the feature fires — the honest per-cell support) | no conditional CSV |
| `elicitation.json` | `tables.export_elicitation` | prompt→response co-activation edges (lift, log2-lift, support, Bonferroni significance); payload capped per-concept by \|log2 lift\| (symmetric — suppression edges kept) plus all significant | no elicitation CSV |
| `delta.json` | `tables.export_delta` | `{raw, clustered}` winner-contrast table (legacy; no longer fetched at startup) | no delta CSV |
| `prompt_features.json` | `tables.export_prompt_features` | prompt-lens concepts + fidelity + clusters | no prompt interpret dir |
| `map.json` | `maps.export_map` | battle-level UMAP scatter (z_diff) | `--map` off |
| `response_map.json` | `maps.export_response_map` | single-response UMAP (`z_a`, plus `z_b` when paired) | flag off or no `z_a` |
| `prompt_map.json` | `maps.export_prompt_map` | prompt-space UMAP with per-point prompt/completion features; needs labels for winner orientation | flag off, missing lenses, or no labels |

## Conventions

- **JSON validity**: every write goes through `sanitize._dumps` — `NaN`/`Inf`
  and numpy scalars become `null`/plain numbers (bare `NaN` breaks
  `JSON.parse` silently in the browser).
- **Label-optionality**: `meta.has_preference` is false when the dataset had no
  usable preference column; the viewer then hides every preference-derived
  surface. Preference-dependent artifacts are simply absent.
- **Honest fit reporting**: `meta.r2` is a true R² (1 − SS_res/SS_tot on
  linearly rescaled predictions); `is_loo` says whether predictions are
  leave-one-model-out; `loo_r2` is null unless they genuinely are.
- **Lazy loading**: `examples/` shards, `examples_by_model.json`, and the three
  maps are fetched on demand, never at startup.
