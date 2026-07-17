# Add an interpreter

An **interpreter** (a *namer*) labels each SAE feature with a short concept — it
looks at the responses (or prompts) where feature `f` fires hardest and asks an LLM
"what do these share?", writing one `feature_id → concept` row per feature.
PrefScope ships three (`pairwise`, `individual`, `single-text`); this guide shows
how to add your own.

Read [the registry](the-registry.md) first if you haven't — it explains how
components are registered and selected. This guide assumes you know that part.

## The contract

An interpreter subclasses `NameStrategy` (`prefscope/interpret/strategy.py`) and
implements one method:

```python
class NameStrategy(ABC):
    def __init__(self, *, features=None, n_active=12, n_zero=8, verify_frac=0.2,
                 seed=0, abbreviate=False, concurrency=1, debug_dir=None): ...

    @abstractmethod
    def name(self, codes: LensCodes, client) -> pd.DataFrame:
        ...
```

- **`codes: LensCodes`** — the lens's codes and text (fields below). You never touch
  `.npy` files; the framework loads them for you.
- **`client`** — an LLM client. Call `client.raw(messages, ...)` for a completion
  (see the built-ins for the message format); ignore it if your naming isn't
  LLM-based.
- **returns** — a `DataFrame` with **at least** `feature_id` (int) and `concept`
  (str). Downstream stages (verify, cluster) join on `feature_id`, so both columns
  are required.

### What `LensCodes` gives you

| field | type | meaning |
|-------|------|---------|
| `lens_dir` | `Path` | the lens directory on disk |
| `input_rep` | `str` | `"difference"`, `"individual"`, or `"prompt"` |
| `battles` | `pd.DataFrame` | the battle table — has `completion_a`, `completion_b`, `instruction_id` |
| `z_diff` | `np.ndarray \| None` | `(N, M)` contrast codes (completion lenses) |
| `z_a`, `z_b` | `np.ndarray \| None` | `(N, M)` per-side codes (individual lenses) |
| `lens_kind` | `str` | `"completion"` or `"prompt"` |
| `z_prompt` | `np.ndarray \| None` | `(N, M)` prompt codes (prompt lenses) |
| `prompts` | `list \| None` | prompt texts (prompt lenses) |
| `instruction_ids` | `list \| None` | per-row battle ids (for held-out splitting) |

`N` = number of battles, `M` = number of SAE features. Column `f` of a `z_*` array
is feature `f`'s signed activation across battles.

### Tunables — configured from the config file

A namer's constructor keyword arguments are its config knobs:
`interpreter: {name: my-namer, n_active: 20}` becomes
`registry.make("interpreter", "my-namer", n_active=20)`, validated against your
class's `__init__` signature.

If you **reuse the base `NameStrategy.__init__`**, you inherit its common knobs, all
settable from config: `features`, `n_active`, `n_zero`, `verify_frac`, `seed`,
`abbreviate`, `concurrency`, `debug_dir` (bundled into `self.opts`).

**To add your own tunable, declare it in your subclass's `__init__`** — don't edit
any framework whitelist. List explicitly any base knob you also want to stay
config-settable (parameters reached only through `**kwargs` are invisible to the
config validator and rejected as "unknown"):

```python
@registry.register("interpreter", "my-namer")
class MyNamer(NameStrategy):
    def __init__(self, *, top_phrases=3, n_active=12, **kw):
        super().__init__(n_active=n_active, **kw)
        self.top_phrases = top_phrases
    def name(self, codes, client): ...
```

See [add a verifier](add-a-verifier.md#tunables--configured-from-the-config-file)
for the same pattern explained in full.

## A minimal interpreter

This toy namer labels each feature by the longest top-activating response — no LLM.
It shows the contract; the built-ins (`prefscope/interpret/name.py`) show the real
LLM-annotation pattern.

```python
import numpy as np
import pandas as pd
from prefscope.core import registry
from prefscope.interpret.strategy import NameStrategy


@registry.register("interpreter", "longest-response")
class LongestResponseNamer(NameStrategy):
    def name(self, codes, client):
        z = codes.z_diff if codes.z_diff is not None else codes.z_prompt
        texts = (codes.battles["completion_a"].tolist()
                 if codes.z_prompt is None else codes.prompts)
        rows = []
        for f in range(z.shape[1]):
            top = int(np.argmax(np.abs(z[:, f])))      # peak-firing item
            concept = texts[top][:40].strip() or f"feature {f}"
            rows.append({"feature_id": f, "concept": concept})
        return pd.DataFrame(rows)
```

## Register and select it

The `@registry.register("interpreter", "longest-response")` decorator registers it —
but the decorator only runs if the module is **imported**. Add it to
`prefscope/adapters/__init__.py`, or import it before you call the pipeline.

Then select it:

```bash
# CLI
prefscope interpret name --lens-dir lenses/mylens --name-mode longest-response ...
```
```yaml
# config (pipeline.yaml)
interpreter: {name: longest-response, n_active: 20}
```

`--name-mode auto` (the default) picks `individual` when the lens's `input_rep` is
`"individual"`, else `pairwise`; a prompt lens always resolves to `single-text`.
Name your interpreter explicitly to override that. Config params are validated
against your `__init__` keywords, so a typo raises a clear error listing the valid
ones.

## Test it

```python
def test_longest_response_namer():
    import numpy as np, pandas as pd
    from prefscope.core import registry
    from prefscope.interpret.strategy import LensCodes
    codes = LensCodes(
        lens_dir=".", input_rep="difference",
        battles=pd.DataFrame({"completion_a": ["short", "a longer answer"],
                              "completion_b": ["x", "y"]}),
        z_diff=np.random.randn(2, 3).astype("float32"), z_a=None, z_b=None)
    out = registry.make("interpreter", "longest-response").name(codes, client=None)
    assert {"feature_id", "concept"} <= set(out.columns)
    assert len(out) == 3
```

See [`add-a-verifier.md`](add-a-verifier.md) and [`add-a-clusterer.md`](add-a-clusterer.md)
for the sibling components — they follow the same pattern with a different `kind`
and method.
