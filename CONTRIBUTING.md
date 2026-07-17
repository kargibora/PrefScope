# Contributing to PrefScope

Thank you for helping improve PrefScope. Bug fixes, documentation improvements,
dataset adapters, analysis components, and reproducibility checks are welcome.

## Development setup

Requires Python 3.10 or newer and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/kargibora/PrefScope.git prefscope
cd prefscope
uv sync --extra cpu --extra cluster --group dev
.venv/bin/python -m pytest -m "not slow" -q
```

Use one accelerator extra at a time: `cpu`, `cu121`, or `rocm`. The tests use the CPU
extra. Tests marked `slow` require a large local model or a live service and are not
part of the default contribution loop.

## Making a change

1. Add or update a focused test for behavior changes.
2. Preserve the public contracts documented in `docs/reference/`, or update the
   documentation and changelog when the contract intentionally changes.
3. Keep generated corpora, embeddings, lenses, result directories, credentials, and
   viewer bundles out of commits.
4. Run the test suite and `git diff --check` before opening a pull request.

Registered components should have a stable kebab-case name, validate unknown options,
and document their required inputs. Start with
[`docs/extending/the-registry.md`](docs/extending/the-registry.md).

## Pull requests

Describe the user-visible problem, the chosen behavior, and how it was verified. For
statistical changes, include a deterministic synthetic test and state the estimand or
sampling assumption. Do not include private model outputs or licensed datasets.

The project is released under the MIT License; by contributing, you agree that your
contribution is distributed under that license.
