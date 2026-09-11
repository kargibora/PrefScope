# Current status

PrefScope is an alpha research library.

## Supported

- native, published, precomputed, custom-backend, and optional SAELens feature extraction;
- `PairItem -> Lens.featurize(...) -> FeatureBatch / FeatureMatrix`;
- explicit feature-space and row identity;
- separate `FeatureCatalog` annotations and explicit `feature_id` joins;
- small direct numerical analysis functions;
- a caller-owned `Report` container;
- a thin JSON bridge for separate visualization tools;
- Torch-free base imports.

## Recipes, not framework API

Specialized preference, outcome, context, graph, cluster, and historical reporting code is
retained below `prefscope.recipes`. It is not registered or automatically executed and may
change without compatibility guarantees.

## Not provided

- automatic analysis selection or orchestration;
- a fixed universal result schema;
- statistical-inference policy;
- privacy classification or filtering;
- automatic narratives or scientific claims;
- a built-in visualization application.

Callers remain responsible for study design, data governance, and interpretation.
