# Your first lens

Use `prefscope build-lens --help` to select a dataset, embedding model, and output path.
Building a native lens requires the relevant optional dependencies.

Once built, the lens has the same interface as a published, precomputed, custom, or
SAELens-backed lens:

```python
from prefscope import Lens

lens = Lens.load("artifacts/my-lens")
features = lens.featurize(items)
```

Choose views explicitly when downstream code requires a particular coordinate:

```python
features = lens.featurize(items, views=("response_difference",))
z_diff = features.matrix("z_diff")
```

Paired differences use A-minus-B orientation. Their exact derivation is declared by the
lens representation policy. Save features with
`save_feature_batch(...)` or pass a `FeatureMatrix` directly to a numerical helper.
