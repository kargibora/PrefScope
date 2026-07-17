# PrefScope

PrefScope is a framework for analyzing post-training preference data by concept. It
trains a sparse autoencoder — a *lens* — over embeddings of prompts and model
responses so that each feature is a concept direction, names those directions with an
LLM, and verifies each name on held-out data. Given a set of battles (a prompt, two
responses, and — when available — human or LLM-judge preferences), it produces named
concept tables describing what concepts appear in prompts and responses, how prompt
concepts relate to response concepts, which concepts are preferred (an association with
the evaluator, not an objective good/bad label), and a per-model profile: the response
concepts a model expresses often (absolute prevalence, from an individual lens), how it
compares to opponents on the same prompts, and which preferred concepts it under-expresses.

## Installation

### Install the released package

The core package can inspect data, call remote services, and consume existing
artifacts without installing PyTorch:

```bash
python -m pip install prefscope
```

Install an extra only for the capability you need:

```bash
python -m pip install "prefscope[cpu]"       # lens building on CPU or Apple MPS
python -m pip install "prefscope[cluster]"   # mi-leiden clustering
python -m pip install "prefscope[arena]"     # HuggingFace arena loaders
python -m pip install "prefscope[viewer]"    # Streamlit viewer
```

For a source checkout, install `uv`, clone the repository, and sync the environment:

```bash
git clone https://github.com/kargibora/PrefScope.git prefscope
cd prefscope
uv sync --extra cpu --extra cluster
source .venv/bin/activate       # macOS/Linux
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. The commands below
assume the environment is active; after activation, `prefscope` is the installed CLI.

For GPU embedding and SAE training from a source checkout, choose the torch build for
your hardware: `uv sync --extra cu121` (NVIDIA), `--extra rocm` (AMD), or
`--extra cpu`. Requires Python ≥ 3.10.
The naming and verification steps call an LLM through any OpenAI-compatible endpoint —
a hosted one via `OPENROUTER_API_KEY`, or a local vLLM server via `--api-base`.

The repository uses the conventional Python layout: this README, `pyproject.toml`,
`docs/`, `examples/`, and `tests/` live at the repository root; only importable library
code lives under `prefscope/`. The local checkout directory can have any name.

## Quickstart

A 60-battle sample corpus ships in [`examples/`](examples/), so you can exercise the
whole pipeline without preparing data. Lens building still downloads the configured
embedding model, and naming/verification still need an LLM endpoint:

```bash
# 1. look at a battle table (no model needed)
prefscope inspect --corpus examples/sample_corpus.parquet

# 2. build a lens with the smaller 0.6B embedder — embed responses, train the SAE
prefscope build-lens --corpus examples/sample_corpus.parquet \
    --embed-model-id Qwen/Qwen3-Embedding-0.6B --device cpu \
    --input-rep individual --out lenses/demo --m-total 16 --k 4

# 3. name → verify → cluster → score, all from a config matching lenses/demo
OPENROUTER_API_KEY=... prefscope run --config examples/quickstart.yaml
```

Step 3 writes the four concept tables under the config's `out_dir`. For the guided
version, see [Your first lens](docs/tutorials/your-first-lens.md). The tiny counts in
`quickstart.yaml` are only a smoke test; [`examples/research.yaml`](examples/research.yaml)
shows the higher-cost multi-candidate, 300-judgment verification profile.

## Python API

The same lens artifact is reusable from Python. Any iterable of `PairItem` objects can
be encoded; DataFrames and parquet files are also accepted by `Lens.train`:

```python
from prefscope import Lens, PairItem

lens = Lens.load("lenses/demo", device="cpu")
items = [
    PairItem(id="row-1", x="Explain gravity", y_a="Response A", y_b="Response B",
             pref=1.0, model_a="candidate", model_b="baseline"),
]
codes, metadata = lens.encode_pairs(items)
diagnosis = lens.diagnose(codes, metadata)
```

See the [Python API reference](docs/reference/python-api.md) for training,
single-response data, preference analysis, and custom components.

## What you can do

Each row is something PrefScope produces from a built lens; use the ones your
analysis needs.

| Goal | Output |
|------|--------|
| See the concepts your data contains (prompts and responses) | `feature_names.csv` |
| Check the names aren't LLM guesses (held-out verification) | `feature_fidelity.csv` |
| Group concepts into higher-level behaviors | `feature_clusters.csv` |
| Find which concepts humans or judges reward | `win_relevance.csv` |
| Relate prompt concepts to the response concepts they elicit | a prompt lens + a response lens |
| Analyze the prompts instead of the responses | a prompt lens |
| Report what one model over- or under-does vs its peers, by concept | a per-model report card |
| Browse it all interactively | the Streamlit viewer |

For example, the per-model report card summarizes one model's behavior by concept:

```
# Model X — concept report card
123 battles · win rate 47%

## Frequently distinguishes from opponents
- refuses — differs from opponent in 34% of battles
- very descriptive — differs from opponent in 28% of battles

## Rarely distinguishes from opponents
- gives worked examples — differs from opponent in 3% of battles

## Rewarded gaps
- worked examples — under-expressed, +0.12 Δwin (length-controlled)
```

→ Full guide: [Report a model](docs/how-to/report-a-model.md).

## How it works

Three artifacts, in order — each reusable, so changing a late step never re-runs an
early one:

```
raw datasets ──build-corpus──▶ corpus.parquet
                                   │
                          embed + train SAE   (build-lens)
                                   ▼
                               a LENS dir         ◀── the durable artifact
                                   │   (SAE encoder + sparse codes + manifest)
                         interpret by concept     (prefscope run)
                                   ▼
            concept tables: names · fidelity · clusters · win-relevance
```

- **Corpus** — normalized battles (`prompt`, two completions, optional `human_pref`).
- **Lens** — a frozen SAE over the corpus's embeddings. Its encoder maps a response to
  a sparse code; each entry is one concept's activation, and its sign says which side
  expresses the concept more.
- **Concept tables** — what each concept is, which survive verification, how they
  group, and which correlate with being preferred.

## Pluggable by design

The SAE, namer, verifier, clustering algorithm, and lens representation are registered
components selected by name in configuration. Dataset adapters use the same registry
but are currently passed programmatically to `Lens.train` / `encode_items`; the corpus
CLI accepts the normalized parquet schema. A typo lists the valid options. See
[Extending PrefScope](docs/extending/the-registry.md).

## Documentation

Full docs live in [`docs/`](docs/index.md):

- **[Tutorials](docs/tutorials/getting-started.md)** — install, then build your first lens.
- **[How-to guides](docs/how-to/build-and-analyze-a-lens.md)** — build, analyze, diagnose, bring your own data.
- **[Explanation](docs/explanation/architecture.md)** — the architecture and the math.
- **[Reference](docs/reference/cli.md)** — CLI, config schema, Python API, components.
- **[Extending](docs/extending/the-registry.md)** — add your own verifier, SAE, clusterer, dataset.

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Release changes are
recorded in [CHANGELOG.md](CHANGELOG.md).
