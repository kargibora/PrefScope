# PrefScope documentation

PrefScope finds recurring concepts in prompts and model responses. It can compare
concepts across datasets, response sets, models, ratings, and preferences. This page
helps you find the right guide. New here? Read the [README](../README.md), then start
with a tutorial.

Choose a section based on what you want to do:

| | |
|---|---|
| **[Tutorials](#tutorials)** — learn by doing | **[How-to guides](#how-to-guides)** — accomplish a task |
| **[Explanation](#explanation)** — understand how it works | **[Reference](#reference)** — look up a detail |

Plus **[Extending](#extending)** — add your own components.

## Tutorials
Tested, step-by-step introductions.
- [Getting started](tutorials/getting-started.md) — install and smoke-test.
- [Your first lens](tutorials/your-first-lens.md) — a tiny dataset to the four concept tables, end to end.

## How-to guides
Short instructions for a specific task.
- [Build and analyze a lens](how-to/build-and-analyze-a-lens.md)
- [Analyze a supervised fine-tuning (SFT) dataset](how-to/analyze-an-sft-dataset.md)
  — single response, no preference labels.
- [Extract concepts from every prompt](how-to/extract-prompt-concepts.md)
- [Report a model's concept profile](how-to/report-a-model.md)
- [Diagnose a model](how-to/diagnose-a-model.md)
- [Bring your own dataset](how-to/bring-your-own-dataset.md)
- [Publish a lens](how-to/publish-a-lens.md)
- [Use a pretrained SAE through SAELens](how-to/use-saelens.md)
- [Compare two response sets](how-to/compare-response-sets.md)

## Explanation
Plain explanations of the design and math.
- [Architecture](explanation/architecture.md) — the pipeline as swappable stages.
- [The lens](explanation/the-lens.md) — how a saved lens turns text vectors into sparse feature values.
- [The SAE](explanation/sae.md) — the built-in sparse autoencoders and BatchTopK.
- [Representations](explanation/representations.md) — difference vs individual.
- [Naming and fidelity](explanation/naming-and-fidelity.md) — how names are made and checked.
- [Semantic presence and context](explanation/presence-and-context.md) — thresholds,
  prompt dependence, and model-tendency categories.
- [Diagnosis math](explanation/diagnosis-math.md) — net_direction, pool contrast, validation.

## Reference
Detailed lookup pages.
- [CLI](reference/cli.md) — the `prefscope` subcommands and their flags.
- [Run config schema](reference/config-schema.md) — the `prefscope run` pipeline keys.
- [Analyze config schema](reference/analyze-config-schema.md) — frozen-lens dataset analysis keys.
- [Lens config schema](reference/lens-config-schema.md) — native, SAELens, and custom backend loading.
- [Python API](reference/python-api.md) — `Lens`, typed analysis contracts, and `prefscope.analysis`.
- [API stability](reference/api-stability.md) — stable, advanced, experimental, and internal surfaces.
- [Components](reference/components.md) — every registered component (kind, name, params).
- [Lens directory](reference/lens-directory.md) — files in a lens dir + manifest schema.
- [Viewer bundle](reference/viewer-bundle.md) — the exported JSON artifacts.
- [Glossary](reference/glossary.md) — battle, lens, code, fidelity, net_direction, …
- [Status](reference/status.md) — what's production vs experimental.

## Extending
How to add your own swappable components.
- [The registry](extending/the-registry.md) — **start here** — the extension mechanism.
- [Add a verifier](extending/add-a-verifier.md) ·
  [an interpreter](extending/add-an-interpreter.md) ·
  [a clusterer](extending/add-a-clusterer.md)
- [Add a dataset](extending/add-a-dataset.md) ·
  [add a representation source](extending/add-a-representation-source.md) ·
  [add a lens backend](extending/add-a-lens-backend.md) ·
  [add an SAE](extending/add-an-sae.md)
- [Lens representation policies](extending/add-a-representation.md)
