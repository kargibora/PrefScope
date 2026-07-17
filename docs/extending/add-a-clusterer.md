# Add a clusterer

A **clusterer** groups co-firing features into higher-level behaviors — a large
dictionary tends to split one behavior across several near-duplicate features, and
clustering merges the features that fire together so you analyze a handful of
behaviors instead of hundreds of axes. PrefScope ships three (`mi-leiden`,
`spherical-kmeans`, `agglomerative`); this guide shows how to add your own.

Read [the registry](the-registry.md) first if you haven't — it explains how
components are registered and selected. This guide assumes you know that part.

## The contract

A clusterer subclasses `Clusterer` (`prefscope/pipeline/cluster.py`) and implements
one method:

```python
class Clusterer(ABC):
    @abstractmethod
    def cluster(self, z: np.ndarray, *, features=None) -> pd.DataFrame:
        ...
```

- **`z: np.ndarray`** — the `(N, M)` co-firing matrix. The framework picks *which*
  matrix by lens kind (`z_diff` for a difference lens, stacked `[z_a; z_b]` for an
  individual lens, `z_prompt` for a prompt lens), so you just cluster the columns
  you're handed.
- **`features`** — an optional list of feature ids to restrict to (e.g. only the
  fidelity-verified ones); `None` means all `M` columns.
- **returns** — a `DataFrame` with `feature_id` and `cluster_id` (both int), one row
  per clustered feature.

### No shared `__init__`

Unlike the interpreter and verifier, `Clusterer` has **no base `__init__`** — each
clusterer declares its own keyword tunables. End your signature with `**_` so it
absorbs the union of params the CLI/config runner may pass to *any* clusterer (e.g.
`resolution` is meaningful to `mi-leiden` but not k-means); without `**_` an
unrelated param would raise `TypeError`.

**Reserved control keys** — the runner pops these from the config block before
constructing your clusterer, so do **not** name your params any of them:
`cluster_on`, `fidelity_only`, `name_clusters`, `concurrency`. They steer the runner
(which code matrix to load, whether to restrict to verified features, whether to
LLM-name the resulting clusters), not the algorithm.

## A minimal clusterer

This toy clusterer assigns features to `n_clusters` buckets by the sign and rank of
their mean activation — no sklearn. It shows the contract; the built-ins wrap
`cluster_features` with their `method` fixed.

```python
import numpy as np
import pandas as pd
from prefscope.core import registry
from prefscope.pipeline.cluster import Clusterer


@registry.register("clusterer", "quantile-bucket")
class QuantileBucketClusterer(Clusterer):
    def __init__(self, *, n_clusters: int = 10, seed: int = 0, **_) -> None:
        self.n_clusters, self.seed = n_clusters, seed

    def cluster(self, z, *, features=None):
        feats = list(range(z.shape[1])) if features is None else [int(f) for f in features]
        strength = np.abs(z[:, feats]).mean(axis=0)        # per-feature activity
        order = np.argsort(strength)                       # rank features
        k = max(1, min(self.n_clusters, len(feats)))
        labels = (order.argsort() * k // len(feats)).astype(int)
        return pd.DataFrame({"feature_id": feats, "cluster_id": labels})
```

## Register and select it

The `@registry.register("clusterer", "quantile-bucket")` decorator registers it —
but the decorator only runs if the module is **imported**. Add it to
`prefscope/adapters/__init__.py`, or import it before you call the pipeline.

Then select it:

```bash
# CLI
prefscope cluster-features --lens-dir lenses/mylens --method quantile-bucket ...
```
```yaml
# config (pipeline.yaml)
clusterer: {name: quantile-bucket, n_clusters: 8}
```

The default `mi-leiden` clusterer needs `igraph` + `leidenalg`, installed via the
`cluster` extra: `uv sync --extra cluster`.

The built-ins for reference: `mi-leiden` (params `resolution`, `knn`,
`min_cluster_size`, `seed`), `spherical-kmeans` (`n_clusters`, `seed`),
`agglomerative` (`n_clusters`, `seed`). Config params are validated against your
`__init__` keywords (plus the reserved control keys), so a typo raises a clear error
listing the valid ones.

## Test it

```python
def test_quantile_bucket_clusterer():
    import numpy as np
    from prefscope.core import registry
    z = np.random.randn(50, 12).astype("float32")
    out = registry.make("clusterer", "quantile-bucket", n_clusters=4).cluster(z)
    assert {"feature_id", "cluster_id"} <= set(out.columns)
    assert len(out) == 12
    assert out["cluster_id"].nunique() <= 4
```

See [`add-an-interpreter.md`](add-an-interpreter.md) and [`add-a-verifier.md`](add-a-verifier.md)
for the sibling components — they follow the same pattern with a different `kind`
and method.
