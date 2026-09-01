# API stability

PrefScope `0.2` is an alpha library. This page separates supported entry points from
implementation details. A public import can still evolve before `1.0`, but changes need a
changelog entry and a compatibility path when scientifically safe.

## Stable supported-alpha surface

Use these for ordinary library code:

- `prefscope.Lens`, `LoadedLens`, and `load_lens`;
- `prefscope.api` contracts for representations, features, outcomes, plans, artifacts,
  and results;
- `AnalyzeConfig` and `run_analysis`;
- documented functions exported from `prefscope.analysis`;
- `TableDataset`, `HuggingFaceDataset`, `ColumnMapping`, and `prepare_dataset`.

The canonical task-centered flow is:

```text
RepresentationSource → RepresentationBatch → Lens → FeatureBatch
                     → AnalysisDataset → AnalysisPlan → DatasetAnalysisResult
```

`prefscope.api` is the canonical import surface for these typed contracts. Top-level
`prefscope` re-exports remain supported for convenience and backward compatibility.

## Advanced supported surface

These interfaces are public but expect more knowledge of PrefScope internals:

- direct `Lens(...)` construction with injected numerical components;
- `AnalysisComponent` implementations and `AnalysisPlan.from_names(...)`;
- `pairs_to_battles`, `LensSource`, and configuration override helpers;
- custom `RepresentationSource` objects and explicit `load_plugins(...)` calls;
- low-level array functions in `prefscope.analysis`.

## Experimental capabilities

These work in their documented limits but can change more quickly:

- pooled residual and custom-coordinate representation sources;
- custom `lens_rep` artifacts beyond the built-in manifest contract;
- unsafe representation-coordinate overrides;
- token/ragged activation workflows and semantic-role classification;
- `SAELensProjector`, `SAELensTextBackend`, `Lens.from_saelens(...)`, and token-first pretrained SAE inference.
- `LensBackend`, `LensCapabilities`, `Lens.from_backend(...)`, and lens-backend YAML registry dispatch.

## Internal modules

Underscore-prefixed modules, functions, and constants are implementation details. The
`prefscope.pipeline` package primarily contains orchestration. Import its documented
configuration and runner objects only; do not depend on stage helpers or publication
mechanics.

Compatibility facades such as `prefscope.api.analysis` and
`prefscope.api.loaded_lens` keep old import paths working while implementation lives in
smaller focused modules. Documented import paths, object module identities, and method
signatures remain stable. Private globals and monkeypatch points are not public contracts.

## Deprecation policy

Before `1.0`, PrefScope can revise a public alpha API when the current behavior is unsafe
or blocks a coherent interface. Otherwise, a renamed public entry point should keep a
compatibility alias and a changelog note for at least one minor release. Scientific
meaning, artifact schemas, and default changes must be stated explicitly.
