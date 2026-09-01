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

The default test loop uses the `cpu` extra. GPU setup depends on the platform; install
the appropriate PyTorch build from the official PyTorch index before installing the
`torch` extra. Tests marked `slow` require a large local model or a live service and are
not part of the default contribution loop.

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

## Documentation style

Write for a reader who is new to the project.

- Start with what the user can do. Explain internal design afterward.
- Use short sentences and common words.
- Define a technical term the first time it appears. Keep exact API names unchanged.
- Put one main action in each numbered step.
- Separate a plain explanation from exact statistical or schema details.
- Do not call raw feature activity semantic presence.
- Test every command example and use links instead of vague file paths.

## Pull requests

Describe the user-visible problem, the chosen behavior, and how it was verified. For
statistical changes, include a deterministic synthetic test and state the estimand or
sampling assumption. Do not include private model outputs or licensed datasets.

The project is released under the MIT License; by contributing, you agree that your
contribution is distributed under that license.
