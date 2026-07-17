# Status — what's production vs experimental

A single, honest maturity map of what ships today.

## Production — works today, tested

| Area | What |
|------|------|
| Corpus | `build-corpus` (arena loaders + custom parquet), `inspect` |
| Lens build | `build-lens` (`difference`/`individual`), `Lens.train` on paired or homogeneous single-response data, `build-prompt-lens`, sharded `embed-corpus`/`embed-prompts` |
| Analysis chain | `interpret name` / `interpret verify`, `cluster-features`, `win-relevance`; the config runner `prefscope run` and `run_pipeline(...)` (completion + prompt lenses) |
| Diagnosis | `diagnose`, `build-bank`, `validate-diagnosis`, `win-relevance` |
| Python (inference) | `Lens.load/encode/encode_items/encode_pairs/diagnose/evaluate_preference`; the `prefscope.analysis` functions; `SAEProjector` |
| Extensibility | the registry + `interpreter`/`verifier`/`clusterer`/`lens_rep`/`negative_sampler` components; `Dataset` adapters (`CsvDataset`, `OpenJuryDataset`) |
| Viewer | the Streamlit app |

## Experimental / partial

| Area | State |
|------|-------|
| Token-level SAE | `extract-activations`, `train-token-sae`, `summarize-activations` — present, less exercised |
| Alternate SAE (`simple-topk`) | trainable as an ablation; deployable as a frozen lens — it selects top-`K` per example at inference (`_threshold_select` → per-example top-`K`) |
| Custom `lens_rep` from the CLI | registry-resolvable, but `--input-rep` `choices` are hardcoded — needs a one-line edit to select a new name |

## Not built (roadmap)

Mentioned for scope; not shipped behavior:

- `diagnose-dataset` — per-sample spurious-preference detection.
- Feature-Conditioned Prompting — a candidate research direction.

## Removed

- The earlier, unused build/interpret/verify/cluster **facade** (`api/lens.py`) and the
  per-feature ABCs (`core.Interpreter`/`Verifier`, `FeatureEvidence`/`FeatureLabel`/
  `VerifyResult`) were deleted. It is distinct from the current public `Lens` object
  in `api/loaded_lens.py`, which owns train/load/encode/analyze. The live extension
  contracts are the strategy classes in `prefscope/interpret/strategy.py` and
  `prefscope/pipeline/cluster.py` — see [the registry](../extending/the-registry.md).

## Building a lens: CLI or Python

Build from the CLI with `build-lens`, or in Python with
`Lens.train(data, config=..., out=<dir>)`, which trains, saves, and returns a loaded
`Lens` in one call. Load an existing lens directory with `Lens.load(path)` (alias
`load_lens`) or `Lens.from_dir` for Python-side inference/diagnosis.
