# Architecture

PrefScope separates feature extraction, numerical calculation, annotation, and result
collection.

```text
PairItem
  -> Lens / LensBackend
  -> FeatureBatch
       -> FeatureMatrix view
            -> small numerical functions
       + FeatureCatalog joined explicitly by feature_id
  -> caller-defined tables and metrics
  -> Report
```

## Core objects

### `PairItem`

One prompt and its A/B responses. Preference values, when present, mean the probability
that A is preferred.

### `Lens`

A user-facing wrapper around a backend. `Lens.featurize(...)` is the only supported
feature-extraction path. Native, precomputed, custom, and SAELens backends declare which
views they can provide.

### `FeatureBatch` and `FeatureMatrix`

A `FeatureBatch` holds row-aligned views from one lens call. A `FeatureMatrix` is one view
with explicit row IDs, feature IDs, role, orientation, and feature-space provenance.
Every paired difference is oriented A minus B. Individual lenses derive
`z_diff = f(e_A) - f(e_B)`, while direct-difference lenses compute
`z_diff = f(e_A - e_B)`.

### `FeatureCatalog`

A catalog stores proposed names and other display annotations. It is separate from
activations. Joins use explicit `feature_id` values and check feature-space identity when
identity is available.

### Numerical analysis functions

The supported functions compute basic activity summaries, boolean overlap counts, and top
rows. They accept `FeatureMatrix` directly. No registry, plan, schema, or key-driven
dispatch sits between the caller and the function.

### `Report`

`Report` is a container for caller-named finite numbers, pandas tables, and JSON metadata.
It has no statistical, privacy, semantic, or presentation policy.

## What is not core

Specialized preference, outcome, context, graph, clustering, and reporting workflows are
kept in `prefscope.recipes`. They are ordinary modules for reuse or adaptation. PrefScope
does not export them from the package root, register them, or run them automatically.

Visualization is also separate. `prefscope.viewer_export` only serializes existing
features, an optional catalog, and caller-provided tables. A separate application owns
charts and interaction.

## Scientific boundary

Numerical activity is not semantic presence. A feature name is not validated meaning.
Association is not causation. PrefScope preserves the data needed for a study, but the
caller owns sampling, hypotheses, inference, privacy review, and claims.
