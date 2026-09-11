# Build and use a lens

Start with the local demo and inspect command-specific options:

```bash
prefscope init-demo --out demo
prefscope build-lens --help
```

After building or downloading a lens, feature extraction uses the Python API:

```python
from prefscope import Lens, PairItem, activation_summary

lens = Lens.load("artifacts/lens")
items = [PairItem(id="1", x="prompt", y_a="A", y_b="B")]
batch = lens.featurize(items, views=("response_a", "response_b"))
activity_a = activation_summary(batch.matrix("z_a"))
```

Use `save_feature_batch(...)` when the computed views need to be shared with another local
step. Join proposed annotations through `FeatureCatalog` and explicit `feature_id` values.

There is no automatic analysis command. Call the small numerical function needed by the
study, or explicitly adapt a module from `prefscope.recipes`.
