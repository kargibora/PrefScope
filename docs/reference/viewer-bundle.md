# Viewer Bundle Reference

The **viewer bundle** is the directory of static JSON that `prefscope-export-viewer` (or
`python -m prefscope.viewer_export`) writes to `viewer-data/` by default for static
visualization clients. Browsers
cannot read `.npy`/`.parquet`, so every artifact is flattened JSON. The export format
is self-contained and does not depend on a particular viewer repository or directory.

Sources of truth:
- `prefscope/viewer_export/` — the export implementation
  (`sanitize` / `features` / `diagnosis` / `examples` / `tables` / `maps` / `cli`)
- `bundle_manifest.json` — schema version, completed files, and processing errors

For example:

```bash
prefscope-export-viewer --lens-dir lenses/response --analysis-dir results/response \
    --corpus battles.parquet --feature-map \
    --prompt-lens lenses/prompt --prompt-interpret-dir results/prompt \
    --out viewer-data
```

By default, each response and prompt example shard contains the strongest activators,
four random active examples, four examples near the activation cutoff, and a small
strong set for each available language/source. Use `--examples-random`,
`--examples-boundary`, `--prompt-examples-random`, and
`--prompt-examples-boundary` to change those counts.

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
| `meta.json` | `features.export_meta` | headline numbers: EV, n_verified/n_named, `dataset_mode`, `r2`/`is_loo`/`loo_r2`, `has_preference`, M/K/dim, n_battles | never (always written) |
| `concept_distribution.json` | per-concept prevalence and observed maximum activation, per-row concept counts, per-group fire rates and group totals | always |
| `prompt_concept_distribution.json` | per-prompt-concept prevalence and observed maximum activation, concepts activated per prompt, and available per-group fire rates/totals; computed strictly from `z_prompt.npy` | no prompt lens or prompt interpretations |
| `coactivation.json` | concept pairs that co-fire above independence, with example row indices, prompt/response text, and both sparse activation magnitudes; includes every axis and both response sides when a paired individual lens has `z_a` and `z_b` | always |
| `features.json` | `features.export_features` | every SAE axis, including unnamed/unverified axes, plus available concept, fidelity, semantic role/family, semantic calibration/threshold, prompt dependence/category, win-relevance, semantic fire rate (`generality` when calibrated), `n_prompt_types`, cluster | never |
| `feature_clusters.json` | `clusters.export_feature_clusters` | complete response-feature community inventory: every member ID and annotation, representative labels, coherence statistics, and clustering stability diagnostics; communities are co-firing groups, not merged concepts | no `feature_clusters.csv` in the analysis dir |
| `validation.json` | `cli.main` (from `diagnosis_validation.csv`) | per-model predicted vs actual win rate (`predicted_score_loo` when LOO ran) | no validation CSV (e.g. label-free dataset) |
| `diagnosis.json` | `diagnosis.export_diagnosis` | per-model direction/prevalence, semantic presence basis, model-specific behavior category and context stability, raw counts + pool totals, `prompt_types`, `relations` | no oriented bank (an honest `{"error": "no_bank"}` stub is written instead) |
| `paired_comparison.json` | `comparison.export_paired_comparison` | label-free A/B concept shifts, response scope, prompt-conditioned cells, paired examples | no `--comparison-dir` |
| `examples/<fid>.json` | `examples.export_examples` | strongest, random active, near-cutoff, and per-language/source examples for feature `fid`; each row includes its selection type and within-feature activation percentile | no `--corpus` |
| `examples_by_model.json` | `examples.export_examples_by_model` | per (model × feature): that model's OWN answers exhibiting the feature, with outcome | flag off, no corpus, or no per-side codes |
| `report_battles.json` | `examples.export_report_battles` | per (model × prompt-concept) sample battles for the report-card drill-in | flag off, no labels, or no prompt lens |
| `joint_examples/<prompt-feature>.json` | `examples.export_joint_examples` | top examples where a selectable prompt concept and response concept are both positively active; ranked by balanced joint activation | `--joint-examples` off, no prompt lens, or no individual response codes (`z_a`) |
| `head_to_head.json` | `diagnosis.export_head_to_head` | per model pair: discordant counts `bpos`/`cpos` per feature (viewer runs McNemar + BH) | flag off, difference lens, or misaligned codes (refused) |
| `bias_screen.json` | `tables.export_bias_screen` | per-feature length-confound screen (win assoc, length covariance, residual) | no bias-screen CSV |
| `conditional.json` | `tables.export_conditional` | `{raw, clustered}` δ_{f,k} cells: Δwin within prompt type, significance, `n` (type size) and `nf` (battles where the feature fires — the honest per-cell support) | no conditional CSV |
| `elicitation.json` | `tables.export_elicitation` | prompt→response co-activation edges (lift, log2-lift, support, Bonferroni significance); payload capped per-concept by \|log2 lift\| (symmetric — suppression edges kept) plus all significant | no elicitation CSV |
| `delta.json` | `tables.export_delta` | `{raw, clustered}` winner-contrast table (legacy; no longer fetched at startup) | no delta CSV |
| `prompt_features.json` | `tables.export_prompt_features` | complete prompt-lens name inventory plus available fidelity and clusters; partial verification never removes unverified names | no prompt interpret dir |
| `prompt_feature_clusters.json` | `clusters.export_feature_clusters` | complete prompt-feature community inventory with the same member-preserving contract as `feature_clusters.json` | no `prompt_feature_clusters.csv` in the prompt interpret dir |
| `prompt_examples/<fid>.json` | `examples.export_prompt_examples` | strongest, random active, near-boundary, and per-language/source prompts for every prompt feature; each row includes its selection type and within-feature activation percentile | no corpus or prompt lens |
| `prompt_coactivation.json` | `overview.export_prompt_coactivation` | prompt-axis pairs that co-fire above independence, retaining every prompt axis; when a corpus is supplied, examples include prompt text and both sparse activation magnitudes | no prompt lens |
| `feature_map.json` | `maps.export_feature_map` | one point for every SAE decoder direction; UMAP with cosine distance when available, deterministic SVD fallback otherwise; labels do not determine coordinates | `--feature-map` off or no `sae_model.pt` |
| `prompt_feature_map.json` | `maps.export_feature_map` on the prompt lens | one point for every prompt SAE decoder direction; produced by `--prompt-feature-map`, or by `--feature-map` when prompt-lens inputs are present | no prompt lens/interpretation or map flag |
| `map.json` | `maps.export_map` | battle-level UMAP scatter (z_diff) | `--map` off |
| `response_map.json` | `maps.export_response_map` | single-response UMAP (`z_a`, plus `z_b` when paired) | flag off or no `z_a` |
| `prompt_map.json` | `maps.export_prompt_map` | prompt-space UMAP with per-point prompt/completion features; needs labels and an individual completion lens for valid winner orientation | flag off, missing lenses, no labels, or a nonlinear direct-difference lens |

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
- **Lazy loading**: `examples/` and `prompt_examples/` shards,
  `examples_by_model.json`, and map artifacts are fetched on demand, never at startup.
- **Fail-closed behavior labels**: the exporter uses explicit semantic-role or context
  artifacts when present. Without `feature_roles.csv`, `feature_calibration.csv`, or
  `feature_context.csv`, response axes remain `unclassified`; it never infers
  “behavioral” from a median fire-rate cutoff.
