# Add a representation source

A **representation source** turns `PairItem` rows into fixed-width vectors in the same
row order. Use one to replace the default text embedder with a hosted service,
precomputed vectors, or pooled model activations. The lens and analysis code do not need
to change.

Three contracts are deliberately separate:

1. `RepresentationSource` produces dense vectors such as `prompt`, `response_a`,
   and `response_b`.
2. `LensRep` is the built-in policy that defines how A/B vectors train and pass
   through a difference, individual, or prompt lens.
3. `FeatureBatch` contains projected `z_prompt`, `z_a`, `z_b`, or `z_diff` views
   with explicit role, orientation, row IDs, metadata, and provenance.

## Built-in text source

```python
from prefscope import EmbeddingRepresentationSource, Lens
from prefscope.encode.embed import Embedder

source = EmbeddingRepresentationSource(Embedder(cache=None, device="cuda"))
vectors = source.encode(dataset)             # iterable of PairItem
lens = Lens.load("lenses/completion")
features = lens.project_representations(vectors)
z_diff = features.matrix("z_diff")
```

`Lens.encode_pairs(dataset)` remains available and uses this path internally.
Its historical `(ndarray, DataFrame)` return value is unchanged.

## Precomputed vectors

```python
from prefscope import PrecomputedRepresentationSource, RepresentationBatch

batch = RepresentationBatch(
    row_ids=row_ids,
    arrays={"response_a": pooled_a, "response_b": pooled_b},
    metadata={"group_id": group_ids},
    provenance={
        "representation_family": "pooled_residual",
        "model_revision": "immutable-revision",
        "layer": 20,
        "pooling": "last-token",
    },
)
source = PrecomputedRepresentationSource(batch, source_name="layer-20")
```

This is the direct path for static cached embeddings and already pooled residuals. The
source requires the requested `PairItem` IDs to exactly match batch order. A loaded lens
also compares its declared embedding/model coordinate contract against the batch contract;
same width is not enough. A missing or incompatible description raises an error. The low-level
`Lens.project_representations(..., allow_representation_mismatch=True)` override exists
only for an explicitly audited unsafe projection and records that override in output
provenance. Keep
`representation_family`, model/revision, layer, and pooling explicit: shared static
embedding coordinates and model-specific residual coordinates use the same plumbing but
not the same scientific interpretation.

## Custom source class

```python
import numpy as np
from prefscope import PairItem, RepresentationBatch, RepresentationSource

class MyResidualSource(RepresentationSource):
    def __init__(self, model, *, layer):
        self.model = model
        self.layer = int(layer)

    def encode(self, items):
        items = list(items)
        # Your extractor must preserve this order and return fixed-width,
        # item-level vectors. This example is for a lens trained on pooled residuals.
        a, b = extract_pooled_residuals(self.model, items, layer=self.layer)
        return RepresentationBatch(
            row_ids=tuple(item.id for item in items),
            arrays={
                "response_a": np.asarray(a),
                "response_b": np.asarray(b),
            },
            metadata={
                "prompt": tuple(item.x for item in items),
                "group_id": tuple(item.meta["group_id"] for item in items),
            },
            provenance={
                "source_type": "residual",
                "model_revision": "immutable-revision",
                "layer": self.layer,
                "pooling": "last-token",
            },
        )
```

The batch rejects duplicate row IDs, misaligned arrays, complex or non-finite vectors,
non-JSON provenance, credentials, and absolute local paths. PrefScope copies accepted
numerical arrays into read-only float32 arrays. This prevents later caller changes from
invalidating the checks. Callable, precomputed, and `Lens.encode_*` paths also require
returned IDs to match the requested item order. A source may use any backend, but it must
not put runtime tokens in provenance.

For a lightweight function adapter:

```python
from prefscope import CallableRepresentationSource

source = CallableRepresentationSource(my_function, name="my-vectors")
```

`my_function(list_of_items)` must return `RepresentationBatch`.

## Registration

Programmatic registration is optional:

```python
from prefscope import registry

registry.register("representation_source", "my-residuals")(MyResidualSource)
source = registry.make("representation_source", "my-residuals", model=model, layer=20)
```

Custom modules must be imported before `registry.make(...)`. PrefScope does not
yet discover third-party Python packages automatically, and config/CLI loading
of arbitrary source classes is intentionally not enabled yet.

## Current boundary

The public source contract currently accepts finite fixed-width **item-level**
matrices. Do not place ragged arrays into a batch or describe pooled residual features
as token-level evidence. A pretrained SAELens checkpoint is the explicit exception to
pre-SAE pooling: use `Lens.project_saelens_tokens(...)`, which accepts flat token rows
plus item membership and applies the SAE before max pooling. It does not route ragged
arrays through `RepresentationBatch`.

Published lens manifests still use the v2 embedding provenance fields. For an in-memory
custom-coordinate lens, set a portable `representation_contract` mapping on the projector
and the same fields on the source batch; otherwise the output records that the lens did
not declare a compatibility contract. Direct custom-source projection works now. A future manifest schema is still needed to
reconstruct a residual lens from disk without user code.
