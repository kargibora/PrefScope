# Changelog

Notable user-visible changes are recorded here. PrefScope follows semantic versioning;
the `0.1` series is an alpha API and may still evolve with explicit release notes.

## 0.1.0a0 — 2026-07-16

Initial alpha release.

- Added the reusable `Lens` API for training, loading, saving, and encoding paired or
  homogeneous single-response post-training data.
- Added registered SAE, interpretation, verification, clustering, and lens-
  representation components with YAML/JSON pipeline configuration.
- Added held-out concept-name verification, multi-candidate naming, stratified judgment
  sampling, exact verification budgets, and independent naming/verifier LLM settings.
- Added preference relevance, model diagnosis, reporting, prompt/response maps, and
  viewer-bundle export.
- Added versioned lens manifests, torch-free base imports, optional accelerator and
  clustering dependencies, and restricted wheel/source-distribution contents.
- Added tutorials, extension guides, API/config/artifact references, and smoke/research
  example configurations.
