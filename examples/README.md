# PrefScope examples

This gallery uses small, self-contained scripts. Edit the constants at the top of a
script, then run it directly. Each basic script demonstrates one capability and stays
under 80 lines.

## Setup

From the repository root:

```bash
uv sync --extra cpu --extra saelens
```

The inference examples default to the small public SAELens configuration in
`inference/saelens.yaml`. It downloads GPT-2 Small and its SAE on first use. To use a
native PrefScope lens instead, first run the training example and change `LENS_CONFIG`
to `native_trained.yaml`.

Generated lenses, bundles, and event logs go under `example-output/`. Feature bundles
retain input text and labels; keep them private. Observability JSONL files are logs, not
scientific results.

## Capability map

| folder | example | status | shows |
|---|---|---|---|
| `analysis/` | `outcome_association.py` | supported alpha | typed, group-aware descriptive association |
| `analysis/` | `inspect_local_features.py` | supported API; SAELens experimental | multi-row codes joined to proposed descriptions |
| `training/` | `train_completion_lens.py` | supported alpha | `Lens.train` on toy paired data |
| `inference/` | `single_item.py` | supported API; SAELens experimental | top codes and proposed descriptions for one response |
| `inference/` | `local_dataset.py` | supported API; SAELens experimental | local table → private `FeatureBatch` bundle |
| `inference/` | `huggingface_dataset.py` | supported API; SAELens experimental | bounded streaming Hub split → bundle |
| `analysis/` | `preference_relevance.py` | supported analysis; SAELens experimental | descriptive P(A preferred) association |
| `workflows/` | YAML recipes | supported alpha | config-driven interpretation/analysis chains |
| `advanced/` | longer or artifact-heavy examples | mixed; read each header | extensions, presentation, calibrated comparisons |

## Run one example at a time

No model download:

```bash
.venv/bin/python examples/analysis/outcome_association.py
```

Train a small native lens. This downloads `Qwen/Qwen3-Embedding-0.6B`; a GPU is faster.
The synthetic data only demonstrates the API.

```bash
.venv/bin/python examples/training/train_completion_lens.py
```

Inspect one response and print its strongest feature codes and proposed descriptions:

```bash
.venv/bin/python examples/inference/single_item.py
```

The SAELens checkpoint supplies its `neuronpedia_id`; the provider reads that coordinate
and returns a provenance-bound `FeatureCatalog`. Native PrefScope lenses use their
bundled feature names instead.

Featurize the bundled local table:

```bash
.venv/bin/python examples/inference/local_dataset.py
```

Inspect the strongest response-A codes across several bundled rows:

```bash
.venv/bin/python examples/analysis/inspect_local_features.py
```

This separate card keeps bundle creation independent from mutable annotations. It joins
codes and proposed descriptions by `feature_id` and includes bounded row identity and rank
columns.

Stream ten rows from a public Hugging Face dataset. `REVISION = None` is valid; set an
exact commit SHA in the script for a reproducible run.

```bash
.venv/bin/python examples/inference/huggingface_dataset.py
```

Run a small descriptive preference association:

```bash
.venv/bin/python examples/analysis/preference_relevance.py
```

Every basic runnable card wraps the main PrefScope calls with:

```python
with observe_run(EVENTS, pretty=True):
    result = operation()
```

This prints compact progress and writes a privacy-safe structural event log. The script's
normal stdout prints the actual example result.

## Configuration files

- `inference/saelens.yaml`: small public SAELens checkpoint, used by default.
- `inference/native_trained.yaml`: native lens produced by the training card.
- `workflows/quickstart.yaml`: small name/verify/cluster/preference workflow; requires an
  existing lens and `OPENROUTER_API_KEY`.
- `workflows/pipeline.yaml` and `workflows/research.yaml`: templates for real runs.
- `workflows/analyze-published-lenses.yaml`: published-lens analysis template.

## Interpretation boundaries

- Raw SAE activity is numerical activity, not semantic presence, reward, or quality.
- A feature name is a proposed label unless held-out evidence establishes fidelity.
- Only a calibrated presence threshold supports a semantic-presence claim.
- Preference and outcome associations are descriptive, dataset/judge-specific, and
  noncausal.
- Direct `f(e_A - e_B)` is not post-encoding `f(e_A) - f(e_B)`.

The advanced colored comparison remains at
`advanced/presentations/compare_completions.py`. It is a presentation demo, not the basic
API path.
