# API stability

PrefScope is an alpha research library. The supported surface is intentionally narrow.
The `0.3` API is not backward compatible with `0.2`; see the
[migration guide](../how-to/migrate-from-0.2.md) before upgrading.

## Supported public surface

The package-root API covers:

- `PairItem`, `Dataset`, and dataset adapters;
- `Lens`, `LensBackend`, `LensCapabilities`, and representation sources;
- `FeatureBatch`, `FeatureMatrix`, and their simple persistence helpers;
- `FeatureCatalog` and its explicit join and JSON helpers;
- the numerical functions documented in [Python API](python-api.md#numerical-analysis);
- `Report`;
- configuration types used to build a native lens;
- lazy `SAELensProjector` and `SAELensTextBackend` integration names.

A plain root import remains Torch-free and Python 3.10 compatible.

## Experimental or unsupported

Anything under `prefscope.recipes` is source for reuse, not a stable API. This includes
specialized preference statistics, outcomes, context profiles, feature graphs, clustering,
and legacy visualization preparation.

Internal modules not imported at the package root may change without a compatibility
shim. CLI commands for building and inspecting lenses are alpha interfaces.

## Removed framework surfaces

PrefScope no longer supports analysis components, analysis plans, fixed analysis-result
schemas, config-driven analysis orchestration, report compilers, privacy policies, bundle
readers, or run-observability wrappers. These were removed rather than retained behind
compatibility aliases.

## Semantic guarantees

- preference values mean `P(A preferred)`;
- paired differences use A-minus-B orientation and declare whether projection occurs
  before or after differencing;
- feature joins use explicit `feature_id` values;
- feature-space identity is checked where identity is available;
- numerical activity is not automatically semantic presence;
- feature annotations remain separate from activations.
