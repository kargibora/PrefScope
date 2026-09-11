# Add a lens backend

Use a lens backend when your system can turn normalized `PairItem` rows directly into
aligned sparse codes. This is the right boundary for token SAEs, hosted concept APIs,
or another feature system that does not naturally produce one fixed-width
`RepresentationBatch` per item.

For an embedding model followed by a numerical projector, prefer the existing
`RepresentationSource` path. Both paths produce the same `FeatureBatch`.

## Contract

```python
import numpy as np

from prefscope import FeatureBatch, Lens, LensBackend, LensCapabilities


class LengthBackend(LensBackend):
    """Two numerical text features; replace this logic with your own lens."""

    input_rep = "individual"
    m_total = 2
    activation_polarity = "mixed"
    code_semantics = "view_dependent"

    @property
    def capabilities(self):
        return LensCapabilities(
            ("response_a", "response_b", "response_difference"),
            difference="a_minus_b_after_encoding",
        )

    @property
    def feature_space_identity(self):
        return {
            "feature_space_id": "length-and-word-count-v1",
            "feature_space_status": "declared_pinned_coordinate",
        }

    def featurize(self, items, *, views=None, feature_ids=None, batch_size=None):
        if batch_size is not None:
            raise ValueError("LengthBackend does not support per-call batch_size")
        rows = list(items)
        requested = tuple(views or self.capabilities.views)
        selected = (
            tuple(range(self.m_total))
            if feature_ids is None
            else tuple(feature_ids)
        )

        def encode(text):
            return [len(text.split()), len(text)]

        arrays = {}
        needs_a = bool({"response_a", "response_difference"} & set(requested))
        needs_b = bool({"response_b", "response_difference"} & set(requested))
        z_a = z_b = None
        if needs_a:
            z_a = np.asarray(
                [encode(item.y_a) for item in rows], np.float32
            )[:, selected]
        if needs_b:
            z_b = np.asarray(
                [encode(item.y_b) for item in rows], np.float32
            )[:, selected]
        if "response_a" in requested:
            arrays["z_a"] = z_a
        if "response_b" in requested:
            arrays["z_b"] = z_b
        if "response_difference" in requested:
            arrays["z_diff"] = z_a - z_b

        view_names = {
            "response_a": "z_a",
            "response_b": "z_b",
            "response_difference": "z_diff",
        }
        descriptors = {
            "z_a": {
                "activation_polarity": "nonnegative",
                "code_semantics": "numerical_activity",
            },
            "z_b": {
                "activation_polarity": "nonnegative",
                "code_semantics": "numerical_activity",
            },
            "z_diff": {
                "activation_polarity": "signed",
                "code_semantics": "activity_difference",
                "derivation": "a_minus_b_after_encoding",
            },
        }
        return FeatureBatch(
            row_ids=tuple(item.id for item in rows),
            arrays=arrays,
            roles={view_names[view]: view for view in requested},
            orientations={
                view_names[view]: {
                    "response_a": "absolute_a",
                    "response_b": "absolute_b",
                    "response_difference": "a_minus_b",
                }[view]
                for view in requested
            },
            feature_ids=selected,
            activation_polarity=self.activation_polarity,
            code_semantics=self.code_semantics,
            provenance={
                "backend": "length-demo",
                "views": {name: descriptors[name] for name in arrays},
            },
        )


lens = Lens.from_backend(LengthBackend())
features = lens.featurize(dataset)
```


A backend must:

- accept homogeneous `PairItem` rows with unique, nonmissing IDs;
- return a `FeatureBatch` with exactly aligned row IDs;
- preserve requested feature IDs and their order; without a selection, return every
  ID in `range(m_total)`;
- honor an explicit `batch_size` or raise `ValueError` if it is unsupported;
- declare prompt, response, and contrast capabilities before encoding;
- distinguish direct contrast projection from A-minus-B after per-side encoding;
- keep heavy optional imports inside construction or `featurize`;
- record portable provenance without credentials;
- label raw activity as numerical activity, not semantic presence.

Change `feature_space_id` whenever coordinate order, meaning, preprocessing, or
projection changes. Leave the inherited unbound identity when no stable contract can be
declared. `Lens.featurize(...)` validates and carries a bound declaration into every
output.

`Lens.featurize` adds canonical prompt/response/preference/model/length metadata from
the `PairItem` rows and validates the shared contract. A backend only needs to return
backend-specific extra metadata. Public feature extraction always returns `FeatureBatch`.

## Use the backend

Construct the backend directly and wrap it with `Lens`:

```python
from prefscope import Lens

backend = MyBackend(...)
lens = Lens.from_backend(backend)
features = lens.featurize(items)
```

PrefScope does not require registration or a plug-in module for custom backends. If a
project wants configuration-driven construction, keep that policy in the project rather
than adding it to the shared analysis API.
