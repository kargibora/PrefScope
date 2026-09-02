# Report bundles

Phase 1 provides the typed, privacy-aware **reporting foundation**. It can describe,
validate, read, stage, and publish report bundle schema v3 with transactional recovery.
The report compiler, section planners, and HTML/static renderer are not built; those are Phase 2. Phase 1 does
not ask a browser to recompute analyses.

The public foundation lives in `prefscope.reporting`. It is experimental during the
`0.2` alpha series.

## Bundle v3

A complete bundle is a closed directory rooted by the canonical completion record:

```text
report/
  bundle_manifest.json
  <paths declared by ready artifacts only>
```

`bundle_manifest.json` has `schema: "prefscope.report.bundle"` and `version: 3`.
Its exact top-level fields are `schema`, `name`, `version`, `title`, `status`, ordered
`sections`, `capabilities`, mandatory `lineage`, `artifacts`, `errors`, and the persisted
`privacy` policy.

Each artifact declares unique non-empty `source_refs` into `lineage.sources`. A ready
artifact also declares its own safe relative POSIX `path`, `media_type`, SHA-256, schema
name/version, section, evidence layer, orientation and coordinates. It states the
estimand, units, support, missing-value rule, tie rule, test, multiplicity rule, and
privacy class. The manifest is the only artifact inventory. Readers must not guess
filenames, scan for a familiar table, or treat an undeclared stale file as current.

## Mandatory lineage

`ReportLineage` is not optional. Its four exact blocks are:

- `dataset: DatasetLineage` — `dataset_sha256`, `row_ids_sha256`,
  `group_partition_sha256`, `group_source`, `n_rows`, and `n_groups`;
- `sources: tuple[SourceArtifactReference, ...]` — one or more unique `source_id`
  entries, each with `artifact_type`, positive `schema_version`, and `sha256`;
- `compiler: CompilerProvenance` — `compiler_name`, `compiler_version`,
  `report_spec_name`, positive `report_spec_version`, and `report_spec_sha256`;
- `sampling: SamplingProvenance` — `method`, `sampling_frame_sha256`, nullable
  browser-safe `seed`, `population_count`, `sampled_count`, and
  `max_examples_per_feature`.

Sampling population must equal dataset `n_rows`; sampled count cannot exceed it.
Capabilities `n_rows` and `n_groups` must exactly match dataset lineage. Every artifact
source reference must resolve to a declared source. These hashes bind the report to its
exact dataset, source artifacts, compiler/spec, and sampling frame; filenames are not provenance.

```text
write_report_bundle(directory, manifest, artifacts, *, overwrite=False) -> ReportBundle
load_report_bundle(directory, *, verify_hashes=True, reject_stale=True) -> ReportBundle
```

`write_report_bundle(...)` stages and validates a clean directory, writes artifacts
first, and writes `bundle_manifest.json` **last** as the staged completion marker. It
fsyncs the tree and uses backups, quarantine, and next-writer recovery on failure.
`load_report_bundle(...)` reads that exact filename, validates the strict v3 schema and
privacy policy, rejects unsafe paths and file types, checks the declared inventory, and
verifies hashes by default.

A directory without a valid v3 `bundle_manifest.json` is not a report bundle v3.
`overwrite=True` replaces only an existing valid managed bundle and revalidates its
identity before the destructive rename; it does not adopt an arbitrary directory. A
racing or unmanaged destination is preserved or quarantined, not silently moved into the
bundle.

Publication is not a linearizable atomic directory exchange for concurrent readers or
process death. Overwrite uses multiple renames and can expose a brief absent destination
or leave a recoverable backup. For a new destination on Darwin and Linux, no-overwrite
uses the platform's atomic no-replace rename. A destination that appears during staging or at final install is
preserved rather than moved.

The per-destination `.lock` file is a persistent, never-unlinked `0600` regular file
opened without following symlinks and guarded by nonblocking advisory `flock`. Before any
metadata write, publication requires a trusted destination parent and a current-user,
single-link lock inode. This blocks planted symlink/hard-link mutation. Locking is loaded
lazily so base imports remain portable; publication itself needs a platform with secure
`flock`, `O_NOFOLLOW`, and `O_CLOEXEC`. The lock coordinates cooperating writers only. Do
not delete it.

This does not change the existing [viewer bundle](viewer-bundle.md). Its
`bundle_manifest.json` remains schema version 2 with the documented v2 artifacts and
semantics. A v3 report loader does not reinterpret v2, and a v2 viewer must not infer v3
files.

## Section and artifact states

Sections and artifacts use the same explicit state model:

| status | required detail | meaning |
|---|---|---|
| `ready` | no reason or error | content is present and validated |
| `unavailable` | `not_applicable`, `input_absent`, or `insufficient_support` | absence is expected and is not a processing failure |
| `error` | `processing_error` plus a typed, sanitized `ReportError` | production failed |

An unavailable or error artifact has no path, media type, or hash. Section roll-up is
exact:

- every artifact's evidence layer must equal its owning section's layer;
- a `ready` section needs at least one ready artifact and no error artifacts; it may also
  own unavailable artifacts;
- an `unavailable` section may own only unavailable artifacts;
- an `error` section needs at least one error artifact and no ready artifacts; it may
  also own unavailable artifacts;
- `capabilities.evidence_layers` must equal exactly the set of evidence layers in ready
  sections/artifacts.

Report-level states are exact:

| report status | required content |
|---|---|
| `ready` | no processing errors; unavailable content is allowed |
| `partial` | at least one ready section or artifact **and** at least one processing error |
| `failed` | at least one processing error and no ready section or artifact |

Typed errors distinguish failed processing from legitimate absence. Renderers must show
these states rather than silently omit panels or convert absence of evidence into a
negative claim.

## Evidence ladder

Every section and artifact names its evidence layer. Keep the layers separate:

```text
raw_axis
  → proposed_name
  → extreme_fidelity
  → semantic_presence
  → feature_role / response_scope / context
  → model_tendency / outcome_association
  → descriptive / inferential
```

`metadata` and `provenance` describe the inputs and lineage around this ladder. A proposed
name is not held-out fidelity. Extreme fidelity is not calibrated semantic presence.
Context evidence is not automatically model tendency. An outcome association is
dataset- and outcome-specific, descriptive unless its artifact explicitly declares valid
inference, and never becomes causal through rendering. Store the exact layer; do not let
a frontend promote it based on a filename, column name, or visual style.

## `ReportDataset` boundary

`ReportDataset` accepts exactly `AnalysisDataset`, a live `DatasetAnalysisResult` whose
`.dataset` is an `AnalysisDataset`, or `LoadedAnalysisResult`. It exposes canonical
first-appearance `group_codes`; absent live group IDs become one group per row, while a
detached result uses the saved group codes. Detached input is table-only and cannot carry
`feature_catalogs`, because it has no feature-identity proof. Live catalogs may be
`FeatureCatalog` objects or compatibility DataFrames; they must be a subset of feature
sets and match their complete feature-ID order. Typed catalogs additionally validate a
declared feature-space identity when the live matrix carries one.

Caller row metadata must be a DataFrame whose `row_id` column proves exact analysis row
order:

```python
import pandas as pd
from prefscope.reporting import ReportDataset

rows = result.dataset.row_ids
row_metadata = pd.DataFrame({
    "row_id": rows,
    "split": ["test"] * len(rows),
})
report_data = ReportDataset(result, row_metadata=row_metadata)
```

If omitted, `row_metadata` is just the ordered `row_id` column. The wrapper deep-copies
catalogs and row metadata; it does not compile report sections.

## Canonical JSON tables

Phase 1 table artifacts use canonical JSON-table wire format v1:

```json
{
  "format": "prefscope.json_table",
  "version": 1,
  "schema": {
    "name": "feature_estimate",
    "version": 1,
    "required_columns": ["feature_id", "estimate"],
    "dtypes": {"feature_id": "integer", "estimate": "float"},
    "unique_key": ["feature_id"],
    "orientation": "as_declared",
    "units": {"estimate": "unitless"},
    "allow_extra_columns": false
  },
  "records": [{"feature_id": 7, "estimate": 0.12}]
}
```

The actual `schema` object is the complete `TableContract` manifest. The envelope fields
are exactly `format`, `version`, `schema`, and `records`. Column and record keys must
already use canonical normalized names; the wire parser rejects noncanonical names rather
than normalizing them. Extra columns are not allowed. Values must have portable finite
logical types, and the table must satisfy its declared unique key, orientation, and units.
`TableContract.from_manifest(...)` strictly parses the exact eight-field schema object;
`table_contract_from_manifest(...)` adds report privacy-safety validation.
`table_to_json_table(...)` applies privacy before output, and
`parse_json_table(...)` revalidates the envelope, privacy, logical dtypes, and contract.

### Preparing JSON payloads

The writer distinguishes raw data from already-sanitized payloads:

- Use `json_payload(raw_value, privacy_policy=policy)` for raw data. It applies missing
  conversion, typed privacy roles, ID hashing/text escaping, and canonical JSON encoding.
  Hash those returned bytes and supply the same bytes to the writer.
- `table_to_json_table(...)` returns an already-sanitized table object. An object supplied
  directly to `write_report_bundle(...)` is treated as already sanitized, canonically
  encoded once, and validated without another sanitation pass. This avoids double-hashing
  opaque IDs and double-escaping text.
- JSON bytes supplied directly must already be strict canonical JSON and policy-valid.
  A raw object with `NaN` is not converted by the writer and fails; pass it through
  `json_payload(...)` first.

`canonical_json_bytes(value, privacy_policy=policy)` is the lower-level equivalent when
sanitation is requested. Without a policy, it only performs canonical finite JSON
encoding and size/depth checks.

The browser's Phase-1 job is to display the declared records and metadata. It must not
re-estimate effects, p-values, confidence intervals, multiplicity corrections, support,
semantic thresholds, or availability from lower-level rows.

## Privacy profiles and threat model

A report persists an explicit `PrivacyPolicy`. It applies recursively. Normalized fields
have typed roles:

- `allow_fields`: numeric, boolean, or null scalars only; generic strings are forbidden;
- `categorical_fields`: strings in the field's exact declared non-empty enum;
- `object_fields`: mapping containers whose nested keys are classified recursively;
- `list_fields`: list containers; in shareable mode every element is a typed object;
- `text_fields`: strings governed by `none`, `snippets`, or `full`;
- `id_fields`: non-empty identifiers, made bundle-scoped opaque IDs when enabled;
- `cell_count_fields`: integer/null counts checked against `minimum_cell_count`;
- `redact_fields`: fields required to become and remain null.

All roles, including `redact_fields`, are disjoint after name normalization. Shareable
mode rejects an unknown field at any nesting level and rejects a container declared with
the wrong role. Numbers must be finite and integers browser-safe.

Small cells are not silently fixed. A declared count below `minimum_cell_count` is
rejected; the producer must suppress the cell before export. Credential-like material,
PII-bearing keys, direct email or phone literals, unsafe control/format text, and cycles
fail closed in a bounded preflight **before** redaction roles are applied. Thus
`redact_fields` is not a way to admit raw PII or credentials. HTML escaping only makes
transport inert; it is not PII removal. PII checks cover field-name heuristics and direct
email/phone patterns; they are not a general de-identification guarantee.

Snippet limits count semantic characters after HTML unescaping, not the longer transport
encoding. Sanitization truncates to `snippet_chars`, strips trailing whitespace, then may
append one `…`; validation permits that single extra semantic character only as the
truncation marker. `validate_html_neutral_snippet(...)` also requires one canonical
HTML-neutral encoding, so alternate/double escapes and raw markup are rejected.

Artifact privacy tiers further restrict the bundle policy: `public` and `aggregate`
forbid IDs and text; `opaque_rows` permits policy-valid opaque IDs but no text;
`text_snippets` permits only canonical bounded snippets; `local_full_text` requires a
compatible local/full-text policy.

### `local`

The local profile assumes the bundle remains inside the operator's trusted environment.
It can permit full text, stable identifiers, smaller cells, and local non-JSON artifacts.
Non-JSON payloads must be bytes or an explicit `PathPayload`; they cannot carry a
JSON-table contract. The profile still rejects credential-like material, PII-bearing
keys, and direct email/phone literals. **Local does not mean safe to publish.** Other raw
text, IDs, rare cells, and linked external data can identify people or reveal private
training data.

### `shareable`

The shareable profile assumes the recipient and hosting environment are outside the
trusted boundary. It requires opaque IDs, forbids full text, recursively rejects every
unclassified field or wrongly typed container, and defaults to no text and a minimum cell
count of five. Opaque IDs are salted per bundle: they support within-bundle linkage, not cross-bundle identity, and they are
not proof of anonymity. Producers must still consider rare combinations, free text,
membership inference, and auxiliary-data linkage.

**Phase 1 shareable bundles are JSON-only.** Every ready shareable artifact must use a
supported JSON media type and pass canonical JSON and persisted-policy validation.
Binary, HTML, image, Parquet, and other non-JSON artifacts are local-only in this phase.
A future renderer does not weaken this publication boundary.

See also [Durable analysis results](analysis-result.md),
[Feature bundle reader](feature-bundle-reader.md), [Run observability](observability.md),
and [API stability](api-stability.md).
