# PrefScope documentation

PrefScope has one main data flow:

```text
PairItem -> Lens.featurize(...) -> FeatureBatch / FeatureMatrix
```

## Start here

- [Getting started](tutorials/getting-started.md)
- [Build and use a lens](how-to/build-and-analyze-a-lens.md)
- [Bring your own dataset](how-to/bring-your-own-dataset.md)
- [Use a SAELens checkpoint](how-to/use-saelens.md)

## Concepts

- [Architecture](explanation/architecture.md)
- [The lens](explanation/the-lens.md)
- [Representations](explanation/representations.md)
- [Presence and context](explanation/presence-and-context.md)
- [Naming and fidelity](explanation/naming-and-fidelity.md)

## Reference

- [Python API](reference/python-api.md)
- [CLI](reference/cli.md)
- [API stability](reference/api-stability.md)
- [Current status](reference/status.md)
- [Lens directory](reference/lens-directory.md)
- [Lens config](reference/lens-config-schema.md)
- [Reports](reference/report-bundle.md)
- [Viewer data bridge](reference/viewer-bundle.md)
- [Glossary](reference/glossary.md)

## Extend

- [Add a lens backend](extending/add-a-lens-backend.md)
- [Add a representation source](extending/add-a-representation-source.md)
- [Add a dataset adapter](extending/add-a-dataset.md)

[Specialized analyses](recipes/specialized-analysis.md) live in `prefscope.recipes`. They are usable Python source, not a
stable public API or an automatic analysis system.
