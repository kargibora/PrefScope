# Add a representation source

A `RepresentationSource` converts ordered `PairItem` values into fixed-width numerical
representations. A lens backend then projects them into feature coordinates.

```python
from prefscope import (
    Lens,
    PairItem,
    RepresentationBatch,
    RepresentationSource,
)

class MySource(RepresentationSource):
    def encode(self, items):
        items = list(items)
        return RepresentationBatch(
            row_ids=tuple(item.id for item in items),
            arrays={"response_a": compute_vectors(items)},
            provenance={"source_type": "my-source", "revision": "pinned"},
        )

source = MySource()
lens = Lens(projector, representation_source=source)
features = lens.featurize(
    [PairItem("1", "prompt", "answer")],
    views=("response_a",),
)
```

The returned row IDs must exactly match input item order. Arrays must be finite,
two-dimensional, and fixed width. Provenance must be portable and must not contain
credentials or local secrets.

For already-computed representations, use `PrecomputedRepresentationSource`. For a small
function adapter, use `CallableRepresentationSource`. Both are ordinary objects; no
registry or plug-in loader is needed.

The public extraction result is always `FeatureBatch`. Do not add ndarray-returning Lens
methods for a custom source. Token-level SAELens input uses its explicit token projection
path because encoding must occur before item pooling.
