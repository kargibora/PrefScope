# Durable analysis results

`save_analysis_result(...)` persists a task-centered analysis without retaining the
input feature and outcome arrays. `load_analysis_result(...)` verifies that directory and
returns either a detached result or, only after an exact identity check, a live
`DatasetAnalysisResult`.

```python
from prefscope import load_analysis_result, save_analysis_result

path = save_analysis_result(result, "results/my_analysis")
detached = load_analysis_result(path)                 # LoadedAnalysisResult
attached = load_analysis_result(path, dataset=data)   # DatasetAnalysisResult
```

These functions are available from `prefscope`, `prefscope.api`, and
`prefscope.api.analysis`.

## Directory contract (schema 1)

A durable result is one closed directory:

```text
my_analysis/
  manifest.json
  <artifact_name>.parquet
  ...
```

The manifest has `artifact_type: "prefscope.dataset_analysis_result"` and
`schema_version: 1`. Its exact top-level fields are:

- `n_rows`, ordered `row_ids`, and `row_ids_sha256`;
- `group_source`, canonical first-appearance `group_codes`, and
  `group_partition_sha256`;
- `artifacts`, an ordered list of artifact declarations.

Each artifact declaration contains `name`, `estimand`, `n_rows`, ordered `columns`,
portable `metadata`, a versioned `table_schema`, the canonical
`<name>.parquet` filename, and its SHA-256 digest. The logical `TableContract`, not a
Parquet or pandas physical dtype string, is the portability contract.

Saving requires every artifact to have a `TableContract` and a default unnamed
`RangeIndex`. `TableContract.from_manifest(...)` is the strict inverse of
`to_manifest()`; it accepts exactly the eight declared schema fields. Loading rejects
unsupported schemas, malformed or non-finite JSON, duplicate or extra fields, undeclared
files, unsafe file types, changed files, hash mismatches, and tables that fail the
declared contract. Treat a successfully loaded directory as the unit of exchange; do not
copy only its tables.

### Summary-table budgets

Durable results are for bounded summary tables, not corpus-scale row exports:

| budget | per artifact | whole result |
|---|---:|---:|
| artifacts | — | 64 |
| rows | 500,000 | 1,000,000 |
| columns | 256 | — |
| cells before pandas decoding | 8,000,000 | 16,000,000 |
| compressed Parquet bytes | 128 MiB | 256 MiB |
| Parquet row groups | 2,048 | — |
| declared uncompressed Parquet bytes | 256 MiB | 512 MiB |
| estimated decoded DataFrame memory | 128 MiB | 256 MiB |

The manifest limit is 16 MiB. Loaders preflight Parquet metadata and total budgets before
pandas allocation. Each verified Parquet snapshot uses a 2 MiB in-memory spool and rolls
the remainder to a temporary file.

### Publication boundary

The writer stages and validates every table, writes `manifest.json` last inside the
staging directory, fsyncs data, and uses recovery backups/quarantine where replacement
is needed. Do not describe overwrite/replacement as a linearizable atomic directory
exchange: multiple renames can leave a brief absent path, and process death can leave a
recoverable backup for the next writer. Publication locks are persistent, never-unlinked
`.lock` regular files protected by advisory `flock`; before mutation, the shared publisher
requires a trusted parent and a current-user, single-link lock inode. Only cooperating
writers obey the lock,
and operators must not delete them.

For an absent destination on Darwin and Linux, the final no-overwrite install uses the
platform's atomic no-replace rename. A destination that appears during staging is
preserved rather than moved. Existing or racing unexpected occupants are refused,
preserved, or quarantined for recovery; the writer never silently adopts their files.

## Detached versus attached results

`load_analysis_result(path)` returns `LoadedAnalysisResult`. It contains:

- `dataset_reference: AnalysisDatasetReference`, with ordered row identity and the
  independent-group partition but no feature, outcome, or group-label arrays;
- immutable, name-aligned `AnalysisArtifact` objects and their validated DataFrames;
- `artifact(name)` and `to_manifest()` convenience methods.

It deliberately is **not** a `DatasetAnalysisResult` and has no `.dataset`. It is safe for
table-only reporting, but it cannot run a new analysis that needs the original arrays.

Pass `dataset=` to reattach. Reattachment succeeds only when all of these match exactly:

1. the ordered row IDs;
2. `group_source`;
3. the canonical independent-group partition.

A reordered dataset, a different grouping, or merely equivalent-looking labels fail.
After the check, the loader returns a `DatasetAnalysisResult` backed by the caller's
complete `AnalysisDataset`.

See also [Python API](python-api.md), [Report bundles](report-bundle.md), and
[API stability](api-stability.md).
