# Status — what's production vs experimental

A single, honest maturity map of what ships today.

## Supported alpha — works today and is tested

| Area | What |
|------|------|
| Data | `prepare-dataset` for local or Hub tables; `build-corpus` for registered arena sources; `inspect` for table checks |
| Lens build | `build-lens` (`difference`/`individual`), `Lens.train` on paired or homogeneous single-response data, `build-prompt-lens`, sharded `embed-corpus`/`embed-prompts` |
| Analysis chain | `interpret name` / `interpret verify` / disjoint `interpret calibrate-presence`, `cluster-features`, `feature-relations`, prompt-group-aware `win-relevance` / `elicit` / `conditional-delta`, generic `associate-outcomes`, and label-free paired `compare-responses`; the config runner `prefscope run` and `run_pipeline(...)` |
| Published-lens application | `prefscope analyze` / `run_analysis(...)` for strict config-driven preparation, frozen-lens encoding, concept export, applicable relationships/comparison/preference analyses, and viewer export |
| Diagnosis | `diagnose`, `build-bank`, `validate-diagnosis`, `win-relevance` |
| Python: load and encode | `Lens.load`, `Lens.from_pretrained`, `Lens.from_config`, backend-neutral `featurize`, historical `encode`/`encode_items`/`encode_pairs`, and `project_representations` |
| Python: analyze | `analyze_dataset`, paired outcome components, prompt-conditioned change components, `normalize_outcomes`, `associate_outcomes`, and `prefscope.analysis` functions |
| Python: data types | `RepresentationBatch`, `FeatureBatch`, `FeatureMatrix`, `OutcomeSpec`, versioned `TableContract` schemas, and aligned result artifacts |
| Extending | public `RepresentationSource`, `LensBackend`, and `AnalysisComponent`; explicit trusted plug-ins; import-driven components; public table/Hub dataset adapters |
| Viewer | versioned static-bundle export plus the built-in Streamlit viewer; compatible external frontends can consume the JSON bundle |

“Supported” means the software path is tested. It does not mean every result is ready
for a paper. You must still explain why prompt groups are independent, report missing
data and analysis choices, and avoid causal or universal quality claims. Semantic
presence chooses a threshold on one set of prompt groups and checks it on another.
Group-aware tests report how many independent groups support the result.

## Experimental / partial

| Area | State |
|------|-------|
| Token-level SAE | `extract-activations`, `train-token-sae`, `summarize-activations` — present, less exercised |
| Pretrained SAELens backend | `Lens.from_saelens(...)` and `featurize(...)` lazily load one declared reader and produce prompt/A/B/A-minus-B views from independent documents; `project_saelens_tokens(...)` remains the exact-activation escape hatch. Prompt-conditioned/chat rendering, structured/temporal hooks, and pinned external-checkpoint publication are not built |
| Alternate SAE (`simple-topk`) | trainable as an ablation; deployable as a frozen lens — it selects top-`K` per example at inference (`_threshold_select` → per-example top-`K`) |
| `interpret classify-role` | LLM-assigned semantic roles; its own help calls it experimental |
| `select-lens` | screens a `sae-metrics` sweep; thresholds are heuristics, not a validated selector |
| Custom `lens_rep` artifacts | the implementation seam is registry-backed, but schema/CLI/downstream capabilities support only difference/individual/prompt; use `RepresentationSource` to replace the vector producer |
| Residual representation artifacts | pooled fixed-width residuals can use `PrecomputedRepresentationSource` or a custom in-memory `RepresentationSource`; versioned manifest reconstruction and token/ragged activation contracts are not built yet |
| Third-party plug-in discovery | `prefscope run` loads an explicit trusted `plugins` module list; automatic installed-package/entry-point discovery is not built |

## Not built (roadmap)

Mentioned for scope; not shipped behavior:

- A `diagnose-dataset` command for per-row spurious-preference detection. The low-level
  `prefscope.analysis.diagnose_dataset` function exists, but no full command or saved
  output contract ships yet.
- Feature-Conditioned Prompting — a candidate research direction.

## Building a lens: CLI or Python

Build from the CLI with `build-lens`, or in Python with
`Lens.train(data, config=..., out=<dir>)`, which trains, saves, and returns a loaded
`Lens` in one call. Load an existing lens directory with `Lens.load(path)` (alias
`load_lens`) or `Lens.from_dir` for Python-side inference/diagnosis.
