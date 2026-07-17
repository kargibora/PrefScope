# Add an SAE

The SAE that turns embeddings into sparse codes is a registry component (kind
`sae`). The three built-ins are `batchtopk`, `jumprelu`, and `simple-topk`; this
guide shows how to add your own and select it with `build-lens --sae-type <name>`.

Unlike the analysis components, an SAE is a `torch.nn.Module` used at both training
and inference. The simplest path is to **subclass `BatchTopKSAE`** and override the
parts that differ — most often just the inference selection rule. Read
[the registry](the-registry.md) first.

## The contract

`BatchTopKSAE` (`prefscope/sae/model.py`) provides the pieces the pipeline calls:

```python
class BatchTopKSAE(nn.Module):
    def encode_pre(self, x) -> Tensor: ...                 # x (N,D) -> pre-activations (N,M)
    def _threshold_select(self, acts) -> Tensor: ...       # inference selection -> sparse codes
    def encode(self, x) -> Tensor: ...                     # = _threshold_select(encode_pre(x))
    def forward(self, x) -> tuple[Tensor, dict]: ...       # (reconstruction, info) for training
    def compute_loss(self, x, recon, info, *, aux_coef=0.0): ...
```

To add an SAE you provide a subclass that:

- **trains** — `forward` returns `(recon, info)` and `compute_loss` returns the loss;
  inherit them, or override for a new objective.
- **selects at inference** — `_threshold_select(acts)` turns pre-activations into the
  sparse code. `encode` (inherited) wraps `encode_pre` + `_threshold_select`, and the
  frozen lens (`SAEProjector`) calls `encode`, so overriding `_threshold_select` is
  enough to change inference behavior.
- **persists** — your weights must be in the `state_dict` under the names the projector
  needs: `encoder.weight`, `decoder.weight`, `input_bias`, `neuron_bias` (the base
  registers these). The projector rebuilds your class from the saved `config` and
  `load_state_dict`s, then calls `encode`.

## Minimal example

A toy SAE that keeps every feature with a positive pre-activation (ReLU gating):

```python
import torch
from prefscope.core import registry
from prefscope.sae.model import BatchTopKSAE


@registry.register("sae", "relu")
class ReLUSAE(BatchTopKSAE):
    def __init__(self, **kw):
        super().__init__(**kw)        # **kw absorbs the param union train_sae passes

    def _threshold_select(self, acts):
        return torch.where(acts > 0, acts, torch.zeros_like(acts))
```

End your `__init__` with `**kw` (or `**_`) so the class constructs from the union of
parameters `train_sae` passes to every SAE — the same convention the clusterers use.

## Register and select

The `@registry.register("sae", "relu")` decorator registers it, but the decorator only
runs if the module is imported. Add an import to `prefscope/adapters/__init__.py`, or
import your module before running `build-lens`.

```bash
prefscope build-lens --corpus corpus.parquet --input-rep individual \
    --sae-type relu --out lenses/relu
```

`--sae-type` accepts any registered name; an unknown one errors and lists the
available SAEs. The chosen name is written to the manifest, so the frozen lens loads
and runs your class automatically.

## Test it

```python
def test_relu_sae_round_trips(tmp_path):
    import numpy as np
    from prefscope.core import registry
    from prefscope.sae.train import train_sae
    from prefscope.encode.sae import SAEProjector
    import torch, prefscope.adapters  # noqa: F401

    assert "relu" in registry.available("sae")
    X = np.random.randn(200, 16).astype("float32")
    model, config, _ = train_sae(X[:160], X[160:], m_total=8, k=4,
                                 sae_type="relu", n_epochs=1, device="cpu")
    torch.save({"state_dict": model.state_dict(), "config": config},
               tmp_path / "sae_model.pt")
    z = SAEProjector(tmp_path / "sae_model.pt").project(X)
    assert z.shape == (200, 8)
```

For a real architecture (e.g. a different sparsity mechanism) follow the `jumprelu`
class in `model.py`: it subclasses `BatchTopKSAE`, adds its own parameters, and
overrides `forward`/`compute_loss`/`_threshold_select`.
