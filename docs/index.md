# PrefScope documentation

PrefScope trains a sparse-autoencoder **lens** over preference data, names the
concept directions, verifies them, clusters them, and scores which concepts humans
reward. This is the documentation map. New here? Start with
[the README](../README.md) for what the framework does, then
[Tutorials](#tutorials).

The docs are organized by what you're trying to do:

| | |
|---|---|
| **[Tutorials](#tutorials)** — learn by doing | **[How-to guides](#how-to-guides)** — accomplish a task |
| **[Explanation](#explanation)** — understand how it works | **[Reference](#reference)** — look up a detail |

Plus **[Extending](#extending)** — add your own components.

## Tutorials
Hand-held, guaranteed-to-work introductions.
- [Getting started](tutorials/getting-started.md) — install and smoke-test.
- [Your first lens](tutorials/your-first-lens.md) — a tiny dataset to the four concept tables, end to end.

## How-to guides
Task recipes; assume you know the basics.
- [Build and analyze a lens](how-to/build-and-analyze-a-lens.md)
- [Report a model's concept profile](how-to/report-a-model.md)
- [Diagnose a model](how-to/diagnose-a-model.md)
- [Bring your own dataset](how-to/bring-your-own-dataset.md)

## Explanation
The ideas and the math, contract-first (the default SAE is one choice, not the framework).
- [Architecture](explanation/architecture.md) — the pipeline as swappable stages.
- [The lens](explanation/the-lens.md) — frozen encoder → signed sparse codes.
- [Representations](explanation/representations.md) — difference vs individual.
- [Naming and fidelity](explanation/naming-and-fidelity.md) — how names are made and checked.
- [Diagnosis math](explanation/diagnosis-math.md) — net_direction, pool contrast, validation.

## Reference
Dry, exhaustive lookups.
- [CLI](reference/cli.md) — every `prefscope` subcommand and flag.
- [Config schema](reference/config-schema.md) — the `pipeline.yaml` keys.
- [Python API](reference/python-api.md) — `run_pipeline`, `LoadedLens`, `prefscope.analysis`.
- [Components](reference/components.md) — every registered component (kind, name, params).
- [Lens directory](reference/lens-directory.md) — files in a lens dir + manifest schema.
- [Glossary](reference/glossary.md) — battle, lens, code, fidelity, net_direction, …
- [Status](reference/status.md) — what's production vs experimental.

## Extending
How to add your own swappable components.
- [The registry](extending/the-registry.md) — **start here** — the extension mechanism.
- [Add a verifier](extending/add-a-verifier.md) · [an interpreter](extending/add-an-interpreter.md) · [a clusterer](extending/add-a-clusterer.md)
- [Add a dataset](extending/add-a-dataset.md) · [a representation](extending/add-a-representation.md) · [an SAE](extending/add-an-sae.md)
