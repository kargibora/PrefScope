# Add a representation (lens_rep)

A **lens representation** decides how a battle's two embeddings `e_a, e_b` become
(1) the rows the SAE trains on and (2) the signed codes saved per battle. The
built-ins are `difference` (`e_a − e_b`), `individual` (pooled `[e_a; e_b]`), and
`prompt`. This is a deeper seam than the analysis components — read
[the registry](the-registry.md) and [explanation/representations.md](../explanation/representations.md)
first.

> **Note — two "representation" things.** The live seam is `lens_rep`
> (`prefscope/pipeline/lens_rep.py`), documented here. There is also a legacy
> `representation` kind (`identity`/`diff`/`concat`/`both`) left from a removed build
> facade; it is registered but not wired into the current pipeline — ignore it.

## The contract

```python
class LensRep(ABC):
    contrastive: bool = True
    @abstractmethod
    def training_matrix(self, e_a, e_b) -> np.ndarray: ...        # (rows, D) SAE training input
    @abstractmethod
    def contrast_codes(self, projector, e_a, e_b) -> np.ndarray: ...  # (N, M) signed per-battle codes
    @abstractmethod
    def oriented_codes(self, projector, e_a, e_b): ...           # -> (z_a_self, z_b_self)
    @abstractmethod
    def output_arrays(self, projector, e_a, e_b) -> dict: ...     # {filename_stem: ndarray} to persist
```

- `e_a, e_b` are `(N, D)` float embedding matrices (the framework applies train/val
  masks before calling, so your methods stay pure `(e_a, e_b) → …`).
- `projector` is duck-typed: anything with `.project(ndarray) -> ndarray` (the real
  one is `SAEProjector`). For a non-linear SAE, `oriented_codes` must do **two real
  forward passes** — `f(e_a) − f(e_b) ≠ f(e_a − e_b)`.
- `output_arrays` returns a dict whose entries are saved as `<stem>.npy` and recorded
  in the manifest. Downstream code expects the stems `z_diff`, and (for per-side
  reps) `z_a` / `z_b`.

## Minimal example

```python
import numpy as np
from prefscope.core import registry
from prefscope.pipeline.lens_rep import LensRep


@registry.register("lens_rep", "sum")          # toy: SAE sees e_a + e_b
class SumRep(LensRep):
    contrastive = True
    def training_matrix(self, e_a, e_b):
        return e_a + e_b
    def contrast_codes(self, projector, e_a, e_b):
        return projector.project(e_a) - projector.project(e_b)
    def oriented_codes(self, projector, e_a, e_b):
        return projector.project(e_a), projector.project(e_b)
    def output_arrays(self, projector, e_a, e_b):
        za, zb = self.oriented_codes(projector, e_a, e_b)
        return {"z_a": za, "z_b": zb, "z_diff": za - zb}
```

## Register and select

Register with `@registry.register("lens_rep", "sum")` and import the module
(add it to `prefscope/adapters/__init__.py`).

**Caveat — the CLI choice list is hardcoded.** `build-lens --input-rep` currently
has `choices=["difference", "individual"]` in `prefscope/__main__.py`. A new
`lens_rep` name is registry-resolvable, but to select it from the CLI you must add it
to that `choices` list (and `build_lens` rejects non-`contrastive` reps up front).
Once built, the lens's `manifest.json` records `input_rep`, and every downstream
stage reads it via `get_lens_rep(input_rep)` — so naming and analysis pick up your
representation automatically.

See [explanation/representations.md](../explanation/representations.md) for why
`difference` and `individual` exist and when each is the right base.
