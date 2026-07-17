# Getting started

This tutorial installs PrefScope and runs a smoke test. By the end you will have a
working environment and will have inspected a battle table — no GPU, no API key, no
model download required.

PrefScope is a pluggable diagnostic framework for pairwise preference data. The
later tutorial ([Your first lens](your-first-lens.md)) trains a real lens; this one
just gets the tooling in place.

## Requirements

- Python ≥ 3.10
- [`uv`](https://docs.astral.sh/uv/) for dependency management

The framework's abstraction layer is pure Python (numpy / pandas). Embedding, SAE
training, and local-model interpretation pull in torch / transformers — but only
when you actually use them, so the install below stays light.

## Install the core

For a released package in a virtual environment:

```bash
python -m pip install prefscope
prefscope inspect --help
```

For a source checkout, clone the repository and let `uv` create `.venv`:

```bash
git clone https://github.com/kargibora/PrefScope.git prefscope
cd prefscope
uv sync                 # lightweight core; intentionally does not install torch
source .venv/bin/activate   # macOS/Linux
```

On Windows PowerShell, use `.venv\Scripts\Activate.ps1`. Once activated, invoke the
installed CLI directly as `prefscope`.

That is enough to run `prefscope inspect` and to drive the analysis chain against a
remote LLM. Two optional feature sets layer on top:

```bash
uv sync --extra arena    # + HuggingFace arena loaders (the build-corpus command)
uv sync --extra cluster  # + igraph + leidenalg for the default mi-leiden clusterer
uv sync --extra viewer   # + the Streamlit viewer
```

## Choosing a PyTorch build

GPU work needs a torch build matched to your accelerator. CUDA and ROCm are the same
platform from torch's point of view, so PrefScope cannot auto-detect which you want —
you pick:

```bash
uv sync --extra cpu     # CPU-only / CI
uv sync --extra cu121   # NVIDIA (CUDA 12.1)
uv sync --extra rocm    # AMD ROCm 6.3 (e.g. MI250X)
```

Pick exactly one. On macOS, `--extra cpu` resolves the regular PyPI torch wheel and
therefore includes MPS support. A plain `uv sync` remains torch-free on every platform.

### Keep the selected environment active

Extras are **not** automatically carried into a later `uv run`. If you
`uv sync --extra cu121` and then use a bare `uv run`, `uv` may re-sync to a different
torch build. Activate `.venv` after syncing and call `prefscope` directly. For scripts
or tests, call `python` from the active environment. Alternatively, set `UV_NO_SYNC=1`
when deliberately using `uv run`.

For the test suite, `.venv/bin/python -m pytest` is the unambiguous non-activated form.

## Smoke test

`prefscope inspect` reads a battle table and prints a summary — no model load, no
network. The repo ships a tiny sample corpus, so this runs immediately with no model
or API key:

```bash
prefscope inspect --corpus examples/sample_corpus.parquet
# or point it at your own corpus parquet / annotations JSON:
prefscope inspect --annotations /path/to/annotations.json
```

You should see battle and model counts, the languages present, and — if the table
carries labels — the preference distribution. If you do not have a table yet, this is
your cue to build one (see the next tutorial's tiny example, or `build-corpus` with
`--extra arena`).

## What you have now

- A PrefScope install you can invoke as `prefscope …` from the active environment.
- A working `inspect` command — your fast, model-free sanity check on any battle
  table before committing to the (slower) embed + train steps.

You do **not** yet have a lens. A lens is the framework's durable artifact: a frozen
encoder that turns responses into sparse concept codes. Building one is the next
tutorial.

→ [Your first lens](your-first-lens.md) — train the smallest possible end-to-end lens
and read its concept tables.

For the conceptual picture of how the stages fit together, see
[the architecture explanation](../explanation/architecture.md).
