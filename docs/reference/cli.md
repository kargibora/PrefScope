# CLI reference

```bash
prefscope --help
prefscope COMMAND --help
```

The CLI prepares data, builds lenses, extracts features, and runs interpretation tools.
Analysis and report orchestration are direct Python workflows and do not have registry- or
config-driven CLI commands.

## Data

- `prefscope inspect` — inspect a corpus or annotation table.
- `prefscope init-demo` — create a local demo dataset.
- `prefscope build-corpus` — normalize a paired corpus.
- `prefscope prepare-dataset` — map a table into PrefScope's canonical row form.

## Native lens construction

- `prefscope build-lens`
- `prefscope embed-corpus`
- `prefscope embed-prompts`
- `prefscope build-prompt-lens`
- `prefscope package-lens`

Use each command's `--help` output for its alpha flags. Native lens training can require
optional model and Torch dependencies.

## Feature extraction and interpretation

- `prefscope encode-dataset`
- `prefscope interpret name`
- `prefscope interpret verify`

These commands create or annotate lens artifacts. Proposed names do not by themselves
establish feature meaning or fidelity.

## Token SAE tools

- `prefscope extract-activations`
- `prefscope train-token-sae`
- `prefscope summarize-activations`

These commands are optional training utilities and can require heavy dependencies.

## Viewer bridge

The separate entry point packages an already saved `FeatureBatch` with an explicit built
Viewer directory:

```bash
prefscope-export-viewer \
  --features artifacts/features \
  --viewer-dist path/to/viewer/dist \
  --catalog artifacts/feature_catalog.json \
  --table activity=results/activity.csv \
  --out results/viewer-site
```

`--viewer-dist` must contain a compatible root `viewer-build.json`. `--out` is a new
self-contained static-site directory and must not exist. The command does not calculate an
analysis, build the Viewer, or start a server.

## Removed commands

Commands that dispatched fixed analyses, compared responses automatically, generated
reports, classified roles, or ran YAML workflows were removed. Use the direct numerical
functions in [Python API](python-api.md#numerical-analysis), and call any specialized
recipe explicitly when a study needs it.
