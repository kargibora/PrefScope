# Static Viewer bundle

`prefscope.viewer_export` packages already-computed data with an explicit, already-built
Viewer. It is a serializer and static-site packager. It does not build the Viewer, run an
analysis, apply privacy policy, or orchestrate reports.

```python
from prefscope.viewer_export import export_viewer_bundle

export_viewer_bundle(
    features,
    "results/viewer-site",
    viewer_dist="path/to/viewer/dist",
    catalog=catalog,
    prompt_features=prompt_features,  # optional, exactly aligned prompt-role batch
    prompt_catalog=prompt_catalog,    # optional, belongs to the prompt space
    tables={"activity": activity_table},
)
```

The destination must not exist. PrefScope assembles the bundle in a sibling staging
directory and renames it into place only after all files have been written. It never
replaces an existing destination.

## Required Viewer build

`viewer_dist` is required and must be a static Viewer build directory. PrefScope does not
run a package manager or Viewer build command. The directory must contain `index.html`
and this compatible `viewer-build.json` at its root:

```json
{
  "schema": "prefscope.viewer_build",
  "schema_version": 1,
  "package": "@prefscope/viewer",
  "version": "0.1.0",
  "supported_data_schemas": [
    {"schema": "prefscope.viewer_data", "versions": [1, 2]}
  ]
}
```

The version is supplied by the Viewer build. The supported-schema list may contain other
entries or versions, but it must include `prefscope.viewer_data` version `2`. The shared
Viewer supports versions `1` and `2`; this exporter always writes version `2`. Treat
`viewer_dist` as trusted publishable executable content. PrefScope rejects links, special
files, embedded `data/`, and reserved paths, but it does not audit JavaScript behavior or
external network references.

## Directory layout

The output is a self-contained static-site directory:

```text
viewer-site/
├── index.html                         # copied from viewer_dist
├── assets/...                         # copied from viewer_dist
├── viewer-build.json                  # copied from viewer_dist
├── data/
│   └── viewer-data.json               # generated PrefScope data
└── viewer-bundle.json                 # generated outer manifest
```

Every file from `viewer_dist` is copied without changing its contents. The `data/`
directory and `viewer-bundle.json` are reserved and must not already exist in the Viewer
build. The Viewer's standard `npm run build` produces a dataset-free rich Viewer.
Use that build without adding data files before export.

`data/viewer-data.json` has schema `prefscope.viewer_data`, version `2`. It contains ordered
row and feature IDs, an explicit feature-space identity/status, each supplied feature view
and its declared semantics, row metadata, feature provenance, an optional catalog with its
own feature-space identity/status, and caller-provided tables with index and column
axis names. Catalog rows may annotate a subset of the batch, but may not name feature IDs
outside it. When both the batch and catalog carry feature-space identities, they must
match. Unbound catalogs use the explicit feature-ID join as a documented weaker mode.
Integer-typed identifiers, metadata, provenance, table axes, and table cells
must be within JavaScript's safe-integer range so their values remain exact. Finite
floating-point table values use Python's round-trip JSON representation, including large
integral-valued floats; missing values become `null`, while positive and negative infinity
are rejected. Date/time values use ISO 8601 strings, interval labels retain their left,
right, and closed fields, and duplicate generic table labels are preserved. No maps,
clusters, distributions, examples, coactivation, labels, or narrative are derived during
export.

## Response and prompt spaces

Version 2 keeps every version-1 root field and adds `prompt`. It is `null` when no prompt
batch is supplied. Otherwise it is a complete version-1 single-space object with its own
`schema: "prefscope.viewer_data"`, `schema_version: 1`, `row_ids`, `feature_ids`,
`feature_space`, `views`, `row_metadata`, `provenance`, `catalog`, and empty `tables`.
It has no recursive `prompt` field.

The root can hold response A/B views, a `response_difference` view, or a single response A
view. Declare view roles explicitly; the exporter does not infer them. Every view in
`prompt_features` must have role `prompt`. Its row IDs must exactly equal the root row IDs
in the same order. The exporter does not reorder or guess joins. `prompt_catalog` requires
`prompt_features` and is checked against that batch, not against the response batch.
Feature-space identities and feature IDs are independent between the two spaces. The same
numeric feature ID can refer to different features in response and prompt data.

## Viewer metadata and caller-computed maps

Root `row_metadata` maps keys to arrays aligned with `row_ids`. The Viewer consumes:

| Key | Meaning |
| --- | --- |
| `prompt` | Prompt text. |
| `response_a` or `response` | Response A text; `completion_a` is accepted by the legacy UI. |
| `response_b` | Response B text. |
| `model_a`, `model_b` or `model` | Caller-supplied model labels. |
| `language`, `source` | Caller-supplied row labels. |
| `preference_probability` | Numeric P(A preferred), in `[0, 1]`, for continuous preference scores. |
| `winner` | `a`, `b`, or `tie` for hard outcomes. |

Supply the preference field that matches the data. A probability is not a hard winner.
Missing preference data means no winrate. Missing model labels stay missing. Export does
not add default model names, preference scores, statistics, or other inferred fields.

Optional root tables `feature_map` and `prompt_feature_map` contain caller-computed
`feature_id`, `x`, and `y` columns. The Viewer joins their rows by `feature_id` within the
response and prompt spaces respectively, never by table position or across spaces. All
other supplied table columns and tables are preserved. Export does not compute map
coordinates or validate statistical meaning.

The rich Viewer also consumes these optional caller-owned tables. All live in the root
`tables` object, including tables for the independent prompt space:

| Table | Columns / interpretation |
| --- | --- |
| `prompt_pole_labels` | `feature_id`, `pole` (`positive` or `negative`), `name`. Proposed names joined to the prompt space by ID and pole; not verification results. |
| `feature_map_meta`, `prompt_feature_map_meta` | One row describing a supplied map, such as `basis`, `projection`, `metric`, or `seed`. Bare coordinates do not establish decoder geometry. |
| `feature_clusters`, `prompt_feature_clusters` | One supplied community per row: `cluster_id`, `feature_ids`, optional `label` and `members`. Membership IDs belong to the response or prompt space respectively. The Viewer does not infer communities. |
| `paired_concept_shifts` | Caller-computed shift rows keyed by response `feature_id`, with the existing paired-comparison fields such as `n_pairs`, `prevalence_a`, `prevalence_b`, `delta_b_minus_a`, and optional `p_value`, `q_value`, intervals, or `response_scope`. |
| `paired_context_shifts` | Shift rows additionally keyed by prompt `region_id` and optional `region_pole`; prompt and response IDs remain separate. |
| `paired_comparison_meta` | One row of supplied comparison metadata, such as side display names, confidence level, and activity interpretation. |
| `model_contrast` | One `payload` row containing an existing direct-comparison report, including its `feature_space_id`, full-dataset `n_pairs`, side display names, feature rows, and observable checks. Requires real separate A/B responses; feature IDs and linked feature IDs must belong to the response space. Supplied statistics are preserved, not recomputed. |
| `model_contrast_examples`, `model_contrast_observable_examples` | Saved evidence selections keyed by `feature_id` or `behavior_id`, plus canonical `row_id` and `direction` (`a_only` or `b_only`). Saved measurements can accompany each row. The Viewer joins full prompt and A/B text from canonical row metadata when evidence is opened. |

Without caller-computed inferential results, the Viewer can show descriptive activity,
coactivation, and aligned-pair summaries from the exported values. It does not invent
significance, verification, context classifications, or calibrated semantic presence.
Difference-only data retain their original pair code and orientation; they cannot supply
separate A/B activations or per-answer prevalence. Supplied maps stay authoritative; an
activity-based fallback is labeled as such, not as decoder geometry.

The Python export remains serialization-only. These display projections belong to the
Viewer; they do not add numerical analysis APIs or run model inference during export.

## Bundle manifest

Root `viewer-bundle.json` has schema `prefscope.viewer_bundle`, version `1`. It records:

- `producer.package` and `producer.version` for the PrefScope serializer;
- `viewer.package` and `viewer.version` from `viewer-build.json`;
- `viewer.build`, including path, schema version, and supported data schemas;
- `viewer.build_sha256`, a deterministic identity for the exact copied Viewer files;
- `data`, with `data/viewer-data.json` and its schema/version;
- `files`, sorted by relative POSIX path, with `path`, `size_bytes`, and `sha256` for every
  copied Viewer file and the generated data file.

The inventory excludes only the outer `viewer-bundle.json`. `build_sha256` is SHA256 of the
UTF-8 canonical JSON encoding of the copied-Viewer inventory: keys sorted, entries sorted
by path, and separators `,` and `:` without extra whitespace. The generated data file is
not part of the Viewer build identity. A compatible packaged Viewer validates the outer
manifest and its own version, then verifies the declared data size and SHA256 before
parsing. It does not fall back to unrelated legacy data when bundle validation fails.

## CLI

```bash
prefscope-export-viewer \
  --features artifacts/features \
  --viewer-dist path/to/viewer/dist \
  --catalog artifacts/feature_catalog.json \
  --prompt-features artifacts/prompt_features \
  --prompt-catalog artifacts/prompt_feature_catalog.json \
  --table feature_map=results/feature_map.csv \
  --table prompt_feature_map=results/prompt_feature_map.csv \
  --out results/viewer-site
```

Tables can be CSV, Parquet, JSON, or JSONL. `--out` names a new directory, not a JSON file.
Serve the directory over HTTPS, or on a trustworthy local loopback origin such as
`http://127.0.0.1:8000/`. The packaged Viewer's Web Crypto checksum verification requires
a secure browser context; ordinary LAN HTTP origins are not supported.
