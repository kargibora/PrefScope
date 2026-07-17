# The registry — how PrefScope is extended

Every swappable part of PrefScope is a **component** registered under a string name.
A namer, a verifier, a clustering algorithm, a lens representation — each is a class
you select by name from a config file or a CLI flag. Adding your own is: write a
class, decorate it, make sure its module is imported. This page explains the
mechanism; the `add-a-*` guides apply it to each kind.

## The mechanism

`prefscope/core/registry.py` is a map of `(kind, name) → class`:

```python
from prefscope.core import registry

registry.register(kind, name)      # class decorator — registers a component
registry.available(kind)           # -> sorted list of registered names
registry.make(kind, name, **params)  # construct one by name; the resolver every stage uses
```

`make` is what the analysis runner and the CLI call. If `name` isn't registered it
raises `ValueError` listing the available names — so a typo in a config tells you
the valid options instead of failing obscurely.

```python
@registry.register("verifier", "my-verifier")
class MyVerifier(VerifyStrategy):
    ...
```

## Import to activate

A decorator only runs when its module is imported. The built-in components register
when you `import prefscope.adapters` (which imports the strategy/cluster/adapter
modules). **For your own component to exist, its module must be imported too** —
either:

- add an import to `prefscope/adapters/__init__.py` (so it registers with the
  built-ins), or
- import your module yourself before calling `registry.make` / running the pipeline.

If you forget this, `registry.make("verifier", "my-verifier")` will report the name
as unknown — the class exists but was never registered.

## Configuration maps to constructor arguments

A component's tunables are its `__init__` keyword arguments. A config block maps
straight onto them:

```yaml
verifier: {name: my-verifier, n_per_bucket: 20, fidelity_threshold: 0.5}
```

is `registry.make("verifier", "my-verifier", n_per_bucket=20, fidelity_threshold=0.5)`.
Params are validated against your component's `__init__` signature, so an unknown key
raises a clear error and any keyword you declare is config-settable — including new
tunables your subclass adds. (Caveat: params reached only through `**kwargs` are
invisible to the validator. See each `add-a-*` guide.)

## The component kinds

| kind | what it does | built-in names | selected by |
|------|--------------|----------------|-------------|
| `interpreter` | name each feature with a concept | `pairwise`, `individual`, `single-text` | `--name-mode` / `interpreter:` |
| `verifier` | check a concept name is real | `pairwise`, `individual`, `prompt` | `--verify-mode` / `verifier:` |
| `clusterer` | group co-firing features into behaviors | `mi-leiden`, `spherical-kmeans`, `agglomerative` | `--method` / `clusterer:` |
| `lens_rep` | how the SAE input + codes are formed | `difference`, `individual`, `prompt` | `--input-rep` / lens manifest |
| `sae` | the autoencoder architecture | `batchtopk`, `jumprelu`, `simple-topk` | `--sae-type` / lens manifest |
| `negative_sampler` | pick "silent" items for fidelity checks | `random`, `similar` | `--negatives` |
| `dataset` | adapt your data into `PairItem`s | `table`, `openjury` | *programmatic* (see below) |

Each `add-a-*` guide gives the exact interface, the data your method receives, and a
runnable example:

- [Add a verifier](add-a-verifier.md) · [Add an interpreter](add-an-interpreter.md) · [Add a clusterer](add-a-clusterer.md)
- [Add a dataset](add-a-dataset.md) · [Add a representation](add-a-representation.md) · [Add an SAE](add-an-sae.md)

### Two notes

- **`dataset` is programmatic today.** The live `build-lens` path reads a corpus
  parquet or annotation JSON directly; it does not name-select a `dataset`. You use
  a custom `Dataset` by instantiating it and passing it to `LoadedLens.project(...)`
  (any iterable of `PairItem`). See [bring your own dataset](../how-to/bring-your-own-dataset.md).
- **The SAE is a `torch.nn.Module`,** not a lightweight strategy — it is used at both
  training and inference. Adding one means subclassing `BatchTopKSAE` rather than a
  plain class; see [add an SAE](add-an-sae.md). The `representation` and `source`
  kinds are also registered but are currently unwired (legacy of a removed build
  facade) — prefer `lens_rep` for the contrast/representation seam.

## Why a registry

It keeps the pipeline declarative: a run is fully described by names + params in a
config, the same component can be selected from the CLI or constructed in Python,
and a new strategy drops in without touching the orchestration. The analysis runner
(`prefscope run`) is just a loop that `make`s each stage's component and calls it.
