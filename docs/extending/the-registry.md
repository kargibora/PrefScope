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

A decorator runs only when Python imports its module. In Python, import the module before
calling `registry.make`, or load a declared list explicitly:

```python
from prefscope import load_plugins

load_plugins(["my_package.prefscope_plugin"])
```

The `prefscope run` config accepts the same module list:

```yaml
plugins:
  - my_package.prefscope_plugin
```

The runners import these modules in the given order before registry resolution. They do
not search the environment or automatically import installed packages. Plug-in imports
execute Python code, so use only trusted packages and lock their versions.

`import prefscope.adapters` remains the compatibility aggregator for every built-in
adapter, including heavy SAE implementations. Ordinary runners import only the built-ins
they need.

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
| `clusterer` | group co-firing features into communities | `cofire-leiden`, `mi-leiden`, `spherical-kmeans`, `agglomerative` | `--method` / `clusterer:` |
| `representation_source` | produce aligned fixed-width vectors | `text-embedding`, `precomputed` | programmatic |
| `analysis_component` | run one reusable analysis | `outcome-associations`, `paired-concept-shift`, `paired-outcome-shifts`, `prompt-conditioned-outcome-shifts`, `preference-length-confounds`, `feature-artifact-diagnostics` | `AnalysisPlan` |
| `lens_rep` | built-in SAE input + code policy | `difference`, `individual`, `prompt` | `--input-rep` / lens manifest |
| `sae` | the autoencoder architecture | `batchtopk`, `signed-batchtopk`, `batchtopk-relu`, `jumprelu`, `simple-topk` | `--sae-type` / lens manifest |
| `negative_sampler` | pick "silent" items for fidelity checks | `random`, `close` (`similar` is an alias) | `--negatives` |
| `dataset` | adapt your data into `PairItem`s | `table`, `openjury`, `huggingface` | *programmatic* (see below) |

Each `add-a-*` guide gives the exact interface, the data your method receives, and a
runnable example:

- [Add a verifier](add-a-verifier.md) · [Add an interpreter](add-an-interpreter.md) ·
  [Add a clusterer](add-a-clusterer.md)
- [Add a dataset](add-a-dataset.md) ·
  [Add a representation source](add-a-representation-source.md) ·
  [Lens representation policies](add-a-representation.md) · [Add an SAE](add-an-sae.md)

### Two notes

- **`dataset` is programmatic today.** The live `build-lens` path reads a corpus
  parquet or annotation JSON directly; it does not name-select a `dataset`. You use
  a custom `Dataset` by instantiating it and passing it to `Lens.project(...)`
  (any iterable of `PairItem`). See [bring your own dataset](../how-to/bring-your-own-dataset.md).
- **The SAE is a `torch.nn.Module`,** not a lightweight strategy — it is used at both
  training and inference. Adding one means subclassing `BatchTopKSAE` rather than a
  plain class; see [add an SAE](add-an-sae.md).
- **`lens_rep` is closed at the artifact boundary today.** The registry resolves the
  three built-ins, but manifests and downstream capabilities do not safely round-trip
  arbitrary third-party policies. Replace the vector producer through
  `RepresentationSource`; do not advertise a custom `lens_rep` as reloadable.
- **Third-party registration is import-driven.** Import the custom module before
  programmatic use. PrefScope does not yet discover installed plug-ins in a fresh CLI
  process.

## Why a registry

It keeps the pipeline declarative: a run is fully described by names + params in a
config, the same component can be selected from the CLI or constructed in Python,
and a new strategy drops in without touching the orchestration. The analysis runner
(`prefscope run`) is just a loop that `make`s each stage's component and calls it.
