# Add a verifier

A **verifier** decides which concept names are *real* — it checks, on held-out
data, that a feature labelled "uses code blocks" actually fires on responses with
code blocks and not on others. PrefScope ships three (`pairwise`, `individual`,
`prompt`); this guide shows how to add your own.

Read [the registry](the-registry.md) first if you haven't — it explains how
components are registered and selected. This guide assumes you know that part.

## The contract

A verifier subclasses `VerifyStrategy` (`prefscope/interpret/strategy.py`) and
implements one method:

```python
class VerifyStrategy(ABC):
    def __init__(self, *, n_per_bucket=10, verify_frac=0.2, seed=0,
                 fidelity_threshold=0.3, concurrency=1,
                 negatives="random", embeddings=None): ...

    @abstractmethod
    def verify(self, codes: VerifyCodes, names: pd.DataFrame, client) -> pd.DataFrame:
        ...
```

- **`codes: VerifyCodes`** — the lens's codes and text (fields below). You never
  touch `.npy` files; the framework loads them for you.
- **`names: pd.DataFrame`** — the `feature_names.csv` produced by the namer; has
  columns `feature_id` and `concept`.
- **`client`** — an LLM client. Call `client.raw(messages, ...)` for a completion
  (see the built-in verifiers for the message format); ignore it if your check
  isn't LLM-based.
- **returns** — a `DataFrame` with **at least** `feature_id`, `concept`,
  `correlation`, and `fidelity_pass` (bool). Downstream stages filter on
  `fidelity_pass`, so it is required.

### What `VerifyCodes` gives you

| field | type | meaning |
|-------|------|---------|
| `lens_kind` | `str` | `"completion"` or `"prompt"` |
| `input_rep` | `str` | `"difference"`, `"individual"`, or `"prompt"` |
| `battles` | `pd.DataFrame` | the battle table — has `completion_a`, `completion_b`, `instruction_id` |
| `instruction_ids` | `list[str]` | per-row battle ids (for held-out splitting) |
| `z_diff` | `np.ndarray \| None` | `(N, M)` contrast codes (completion lenses) |
| `z_a`, `z_b` | `np.ndarray \| None` | `(N, M)` per-side codes (individual lenses) |
| `z_prompt` | `np.ndarray \| None` | `(N, M)` prompt codes (prompt lenses) |
| `prompts` | `list[str] \| None` | prompt texts (prompt lenses) |

`N` = number of battles, `M` = number of SAE features. Column `f` of a `z_*` array
is feature `f`'s signed activation across battles.

### Tunables — configured from the config file

A verifier's constructor keyword arguments *are* its config knobs: a block like
`verifier: {name: my-verifier, n_per_bucket: 20}` becomes
`registry.make("verifier", "my-verifier", n_per_bucket=20)`. The framework validates
the keys against your class's `__init__` signature, so a typo raises a clear error
listing the valid ones.

If you **reuse the base `VerifyStrategy.__init__`** (like the minimal example above),
you inherit its seven common knobs, all settable from config: `n_per_bucket`,
`verify_frac`, `seed`, `fidelity_threshold`, `concurrency`, `negatives`,
`embeddings`. The base bundles them into `self.opts` for you to read.

**To add your own tunable, just declare it in your subclass's `__init__`** — you do
*not* edit any framework whitelist. Store it on `self` and read it in `verify`:

```python
@registry.register("verifier", "thresholded")
class ThresholdedVerifier(VerifyStrategy):
    def __init__(self, *, my_threshold=0.5, n_per_bucket=10, seed=0, **kw):
        super().__init__(n_per_bucket=n_per_bucket, seed=seed, **kw)
        self.my_threshold = my_threshold        # your new knob

    def verify(self, codes, names, client):
        ...                                      # use self.my_threshold
```
```yaml
verifier: {name: thresholded, my_threshold: 0.9, n_per_bucket: 20}   # both accepted
```

**Gotcha:** validation ignores `**kwargs`. List explicitly any *base* knob you want
to stay config-settable (above, `n_per_bucket` and `seed` are listed). A base param
left to flow only through `**kw` will be rejected as "unknown" in a config — it's
visible to Python but not to the config validator.

## A minimal verifier

This toy verifier passes a feature if its activation variance clears a threshold —
no LLM. It shows the contract; the built-ins (`prefscope/interpret/verify.py`) show
the real LLM-annotation pattern.

```python
import numpy as np
import pandas as pd
from prefscope.core import registry
from prefscope.interpret.strategy import VerifyStrategy


@registry.register("verifier", "variance")
class VarianceVerifier(VerifyStrategy):
    def verify(self, codes, names, client):
        z = codes.z_diff if codes.z_diff is not None else codes.z_prompt
        rows = []
        for _, r in names.iterrows():
            f = int(r["feature_id"])
            score = float(np.std(z[:, f]))            # your fidelity signal
            rows.append({
                "feature_id": f,
                "concept": r["concept"],
                "correlation": score,                 # required column
                "fidelity_pass": score >= self.opts["fidelity_threshold"],
            })
        return pd.DataFrame(rows)
```

## Register and select it

The `@registry.register("verifier", "variance")` decorator registers it — but the
decorator only runs if the module is **imported**. Add it to
`prefscope/adapters/__init__.py`, or import it before you call the pipeline.

Then select it:

```bash
# CLI
prefscope interpret verify --lens-dir lenses/mylens --verify-mode variance ...
```
```yaml
# config (pipeline.yaml)
verifier: {name: variance, fidelity_threshold: 0.5}
```

`--verify-mode auto` (the default) picks `pairwise`/`individual` from the lens's
`input_rep`, or `prompt` for a prompt lens — name your verifier explicitly to
override that. Config params are validated against your `__init__` keywords, so a
typo raises a clear error listing the valid ones.

## Test it

```python
def test_variance_verifier():
    import numpy as np, pandas as pd
    from prefscope.core import registry
    from prefscope.interpret.strategy import VerifyCodes
    codes = VerifyCodes(lens_kind="completion", input_rep="difference",
                        battles=pd.DataFrame(), instruction_ids=[],
                        z_diff=np.random.randn(50, 3).astype("float32"))
    names = pd.DataFrame({"feature_id": [0, 1, 2], "concept": ["a", "b", "c"]})
    out = registry.make("verifier", "variance").verify(codes, names, client=None)
    assert {"feature_id", "concept", "correlation", "fidelity_pass"} <= set(out.columns)
```

See [`add-an-interpreter.md`](add-an-interpreter.md) and
[`add-a-clusterer.md`](add-a-clusterer.md) for the sibling components — they follow
the same pattern with a different `kind` and method.
