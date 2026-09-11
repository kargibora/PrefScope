# Native lens directory

A native PrefScope lens directory is a frozen feature extractor plus optional proposed
annotations. `Lens.load(...)` validates the directory before use.

Common members are:

| member | purpose |
|---|---|
| `manifest.json` | lens type, representation policy, dimensions, and provenance |
| `sae_model.pt` | trained native projector weights |
| `whiten.npz` | optional preprocessing state |
| `feature_names.csv` or `prompt_feature_names.csv` | proposed annotations by `feature_id` |
| `feature_catalog.json` | the same annotation layer with feature-space provenance |

A training directory can also contain the corpus-aligned arrays used to build or inspect
the lens. The manifest declares any such arrays; callers must not infer their meaning from
a filename alone.

## Representation policies

- `difference` computes `z_diff = f(e_A - e_B)`.
- `individual` computes `z_a = f(e_A)`, `z_b = f(e_B)`, and the derived
  `z_diff = z_a - z_b`.
- `prompt` computes `z_prompt = f(e_prompt)`.

The two difference formulas are not interchangeable for a nonlinear projector. Every
paired contrast is oriented A minus B, and the manifest records the derivation.

## Feature-space identity

A published lens binds annotations and activations to the same ordered feature coordinate
space. Changing weights, whitening, preprocessing, coordinate order, or declared feature
meaning requires a new feature-space identity. Unbound custom lenses must be treated as
unbound rather than assigned a guessed identity.

A loaded native lens keeps the identity of its loaded weights and whitener, even if
another process replaces the files on disk. Reload to use the replacement. Saving the
old object refuses changed backing files rather than publishing them under the old
identity.

## Loading and featurizing

```python
from prefscope import Lens

lens = Lens.load("artifacts/lens")
batch = lens.featurize(items, views=("response_a", "response_b"))
```

`Lens.featurize(...)` returns data in memory. To persist it for another local step, use
`save_feature_batch(batch, "artifacts/features")`. That saved `FeatureBatch` directory is
not an analysis result, report bundle, or extension of the lens directory.

## Publishing annotations

Interpretation outputs should be written outside the source lens. Publish reviewed
annotations explicitly with `Lens.save(..., annotations=...)`. Do not overwrite annotation
files in place while leaving a stale catalog digest.
