# Run observability

`prefscope.observability` provides automatic, opt-in run events, a Torch-free event
schema, and a secure JSON Lines writer. The event stream is a **log**, not an analysis or
report result.

## Quick start

Add one context around existing PrefScope work. Supported operations record themselves;
you do not need to call `record(...)`:

```python
from prefscope.observability import observe_run

with observe_run("events.jsonl", pretty=True):
    experiment()
```

This writes the durable JSONL log and prints compact progress lines to standard error.
For an existing program, enable the same behavior without changing its code:

```bash
PREFSCOPE_EVENTS_PATH=events.jsonl \
PREFSCOPE_EVENTS_PRETTY=1 \
python experiment.py
```

For production, use an existing resolved private directory; on macOS, use an existing
directory under `/private/tmp`, not `/tmp`.

For a normal end-to-end run over two completions, use the colored lens example. It
contains one `observe_run(...)` wrapper and no manual event calls:

```bash
python examples/advanced/presentations/compare_completions.py \
  --lens-repo <owner/lens-repository> \
  --subfolder <completion-lens>
```

For an individual/completion lens, the example reports Completion A, Completion B, and
their strongest post-encoding differences. For a direct difference lens, it instead
reports one signed `f(e_A - e_B)` table and never pretends to score either completion
separately. Direct SAELens checkpoints normally have no bundled concept names, so the
example shows feature IDs unless a lens supplies names; supplied names are explicitly
marked as proposed labels. Raw activation is not semantic presence or a quality score.

`PREFSCOPE_EVENTS_PRETTY` alone does **not** activate recording or terminal output. An
event path is always required, either through `observe_run(...)` or
`PREFSCOPE_EVENTS_PATH`.

## Automatic recording and terminal view

The public signature is:

```text
observe_run(path, *, pretty=None, run_id=None, durable=True,
            max_event_bytes=1048576, monotonic=None, utc_now=None)
    -> context manager[RunContext]
```

`pretty=True` enables the terminal view. `pretty=False` disables it, even when
`PREFSCOPE_EVENTS_PRETTY` is truthy. The default `pretty=None` consults
`PREFSCOPE_EVENTS_PRETTY`. The recognized truthy values are `1`, `true`, `yes`, and `on`
(case-insensitive, with surrounding whitespace ignored).

The terminal view is only a best-effort presentation of automatic events that were
successfully persisted. The JSONL file is the durable source of truth. Presentation
errors never change the operation or the event log. Lines contain only code-owned labels
and allowlisted structural details, such as durations, counts, safe status values, and
normalized built-in error or warning types. They never show event messages, paths, caller
IDs, array contents, prompt/response text, or arbitrary data fields. These guarantees
apply to PrefScope's added progress lines. PrefScope does not suppress or rewrite the
program's own standard output/error, Python's normal warning display, or dependency
progress bars; those independent outputs retain their existing behavior and privacy
policy. Pretty output is bounded; after 200 lines, one suppression notice replaces
further PrefScope progress lines for that run.

Rich is optional. The terminal presenter imports it lazily only when the terminal view
is enabled. If Rich is absent or cannot initialize, the same compact lines use a
plain-text fallback. The disabled presenter path does not load the console module or
import Rich itself; unrelated installed dependencies may manage their own imports.

The context owns and closes its recorder. Recording is best effort: supported operations
attempt `started`, then `completed` or `failed`, but an event-write failure does not
change the operation. Either the started event or the terminal event can therefore be
missing. Nested observed operations are linked when their events are written. A nested
`observe_run(...)` writes to its innermost recorder and restores the outer recorder when
it exits. The yielded `RunContext` exposes `recorder`, `run_id`, `path`, and `closed`, plus
`record(...)`, `record_failure(...)`, and idempotent `close()` methods. These manual APIs
are available for custom events, but supported automatic operations do not require them.

Automatic recording is disabled unless either an `observe_run(...)` context is active or
`PREFSCOPE_EVENTS_PATH` is set. With neither opt-in, PrefScope does not create an event
file, a hidden directory, or any other observability output. Inactive hooks also skip
result-summary traversal. Observability does not read result properties merely to decide
that recording is disabled.

The environment variables provide lazy, process-local activation. PrefScope samples the
path and pretty setting together when an instrumented operation first creates the run.
Importing PrefScope or setting the variables without running such an operation does not
create the file or print progress. Once opened successfully, that recorder and its pretty
setting remain fixed until process exit; later environment changes do not replace them.
Each process must use its own file. Do **not** point several workers or processes at one
JSONL path; merge separate logs afterward if needed.

Environment activation is best effort. If the path is unavailable or fails the secure
writer checks, PrefScope leaves the instrumented operation unchanged and emits no log,
terminal view, or observability error. In contrast, `observe_run(...)` opens its recorder
when the context is entered and raises when the path cannot be opened securely. Use the
explicit context when recorder setup failure must be visible to the caller.

The secure writer rejects symlinks in every path component. On macOS, `/tmp` is normally
a symlink to `/private/tmp`, so an environment-selected `/tmp/...` path is silently
skipped. Use `/private/tmp/...` or a resolved temporary path:

```python
from pathlib import Path
import tempfile

events_path = Path(tempfile.mkdtemp()).resolve() / "events.jsonl"
```

To inspect the durable log directly:

```bash
tail -f /private/tmp/prefscope/events.jsonl | jq -c '{stage,status,data}'
```

Use `jq . /private/tmp/prefscope/events.jsonl` to pretty-print an existing log.

## Automatic event contract

Every automatic operation gets a new `operation_id`. Its events contain:

- `operation_id`, shared by that operation's `started`, `completed`, or `failed` event;
- `parent_operation_id`, set to the enclosing operation ID or `null` at the root;
- `duration_seconds` on `completed` and `failed` events;
- only operation-specific fields that the instrumented call site explicitly supplies.

The runtime does not generically inspect or serialize arguments, return values, local
variables, or tracebacks. Each instrumented call site can supply only explicit,
allowlisted structural metadata such as row counts, feature counts, shapes, schema
versions, artifact counts, or hard-allowlisted built-in feature-view names. It never
supplies array/code contents. Raw prompt/response text, table values, filesystem paths,
dataset or row IDs, lens IDs, and other caller-supplied identifiers are excluded.

An explicit `run_id` is the exception to that ID rule. It is a caller-supplied correlation
identifier and is intentionally persisted in every event. Keep it opaque and
non-sensitive. Do not use a dataset, row, user, model, path, prompt, or other private value
as the run ID.

An automatic failure records only a normalized, fixed-safe `error_type` plus the generic
message `"<stage> failed"`. Built-in exception types retain their name; other types become
`Exception`. It does not record `str(error)`, exception arguments, or a traceback,
because those can contain text, paths, or IDs. The manual `JsonlRecorder.record_failure`
API has a different contract and includes a sanitized exception string; use it only when
the exception message is safe for the log.

### Supported boundary

Automatic coverage is intentionally at selected public Lens and durable-artifact
boundaries. The current stage mapping is:

| stage | supported public operations | safe operation fields, when applicable |
|---|---|---|
| `load_lens` | `Lens.from_config`, `Lens.from_dir` / `Lens.load`, `Lens.from_saelens` | `source_kind`, validated `input_rep`, `n_features` |
| `fetch_lens` | `Lens.from_pretrained` (with a nested `load_lens`) | `source_kind`, validated `input_rep`, `n_features` |
| `featurize` | `Lens.featurize` | validated `input_rep`, `n_rows`, `n_features`, `n_views`, `views`, `shapes` |
| `project_representations` | `Lens.project_representations` | validated `input_rep`, `n_rows`, `n_features`, `n_views`, `views`, `shapes` |
| `encode` | `Lens.encode`, `Lens.encode_one` | validated `input_rep`, `n_rows`, `n_features`, `shape` |
| `encode_pairs` | `Lens.encode_pairs` / `Lens.project`, `Lens.encode_items` | validated `input_rep`, `n_rows`, `n_features`, `shape` |
| `save_lens` | `Lens.save` | validated `input_rep`, `overwrite`, `inference_only`, `has_annotations` |
| `analyze_preference` | `Lens.preference_relevance` | validated `input_rep`, `grouped`, result-table `output_rows`, `output_features`, `shape` |
| `feature_bundle.load` | `load_feature_batch` | safe `requested_view_count`; then `n_rows`, `n_features`, `n_arrays`, `shapes` |
| `feature_bundle.save` | `save_feature_batch` | boolean `overwrite`; then `n_rows`, `n_features`, `n_arrays`, `shapes` |
| `load_feature_source` | `FeatureBundleReader.open` | `n_rows`, `n_features`, `n_views`, `artifact_count` |
| `analyze_dataset` | `analyze_dataset` | `n_rows`, `n_features`, safe `n_groups`, `n_views`, `artifact_count` |
| `analysis_result.load` | `load_analysis_result` | `attached`; then `n_rows`, `n_groups`, `artifact_count`, `shapes` |
| `analysis_result.save` | `save_analysis_result` | `n_rows`, `n_groups`, `artifact_count`, `shapes` |
| `report_bundle.load` | `load_report_bundle` | `n_rows`, `n_groups`, `artifact_count`, safe `status` / `profile`, capability counts |
| `report_bundle.write` | `write_report_bundle` | boolean `overwrite`; then the same structural fields as load |

Encoding aliases and their delegates are coalesced into one outer event rather than
creating duplicate nested events. Native and SAELens `Lens.from_config` routes also keep
one outer `load_lens` event. A Hub config intentionally emits outer `load_lens` →
`fetch_lens` → inner directory `load_lens`, because fetching and loading are distinct
operations. Direct `Lens.from_pretrained` similarly nests `fetch_lens` and `load_lens`.
Lens result view
names are emitted only for the built-in allowlist `z_prompt`, `z_a`, `z_b`, and `z_diff`;
custom names retain counts and an ordered shape list only. Exact safe fields can be absent
when they are not known. They never include the arguments or values listed as excluded
above.

This is not whole-program tracing. Direct `Lens(...)` construction,
`Lens.from_backend`, training, and `project_saelens_tokens` have no dedicated automatic
stage (although its final load can emit `load_lens`, and training can emit PrefScope log
events). Analysis functions other than
`analyze_dataset` and the Lens
preference method above, cheap properties, internal helper calls, and `print(...)` output
are not currently captured. Not every function in
`prefscope.analysis` or `prefscope.pipeline` is instrumented. Use explicit
`RunContext.record(...)`, the logging bridge, or the manual recorder when you need an
event outside the supported boundary. Do not infer that an unlogged operation did not
run.

## Logging and warning bridges

While automatic recording is active, records below the `prefscope` Python logger are
bridged to stage `logging`. The bridge records only normalized, fixed-safe `logger` and
`level` categories, current operation links, and, when present, a normalized safe
`error_type`. These are bounded categories, not arbitrary original logger names, custom
level names, or exception class names. Its event message is empty. It excludes the
formatted log message and arguments, traceback, and stack text. It does not capture
unrelated application loggers or standard output.

Emitted Python warnings are bridged to stage `warnings`. The event records only a
normalized, fixed-safe warning `category` and current operation links; it does not retain
an arbitrary original category name. Its message is empty. It excludes warning text,
filename, line number, and source text. Normal warning filters still decide what is
emitted, and warning output is forwarded. Python's warning hook is process-global, but an
explicit run uses a `ContextVar`. Inherited asynchronous contexts route warnings to the
correct explicit run. A newly spawned raw thread does not inherit that context, so its
warning is normally forwarded without an automatic event unless the caller copies the
context. An environment-selected recorder is process-global and can capture warnings from
any thread. Do not overlap independent observed runs in one process. Warning capture is
best effort.

## Event schema v1

Each line is one strict UTF-8 JSON object with exactly these fields:

| field | meaning |
|---|---|
| `schema_version` | integer `1` |
| `run_id` | non-empty run identifier |
| `timestamp` | canonical ISO 8601 UTC timestamp ending in `Z` |
| `elapsed_seconds` | finite, non-negative monotonic run time |
| `stage` | non-empty stage name |
| `status` | `started`, `completed`, `info`, `warning`, or `failed` |
| `message` | short human-readable observation |
| `data` | bounded JSON object with structured context |

`RunEvent` rejects unknown fields, non-finite numbers, cycles, invalid UTF-8, excessive
nesting, excessive node counts, and oversized strings. Event data is copied into an
immutable form. Credential-like keys and common embedded credential forms are redacted
to `[REDACTED]` in identifiers, messages, and nested data.

Credential scanning is defense in depth, not a general privacy filter. It does not make
raw prompts, responses, tables, arbitrary paths, or personal data suitable for a log.
Record counts, stage names, public schema versions, timings, and sanitized error types.
Do not record secrets or dataset rows and rely on redaction afterward.

## Manual JSONL recording

The existing recorder remains available when you want to define events yourself:

```python
from prefscope.observability import JsonlRecorder

with JsonlRecorder("run/events.jsonl") as events:
    events.record("encode", "started", data={"n_rows": 1200})
    try:
        run_encode()
    except Exception as error:
        events.record_failure("encode", error)
        raise
    events.record("encode", "completed")
```

`JsonlRecorder` appends one event per line. The default serialized-event limit is 1 MiB.
It uses a securely opened regular file, rejects symlinks and unsafe parent components,
sets mode `0600`, and calls `fsync` after each event by default. Set `durable=False` only
when losing recent log events is acceptable. Writes are thread-safe within one recorder;
separate processes or recorders that share a path need external coordination.

`record_failure(...)` stores the exception type and string, but not a traceback or
exception arguments. `RecorderLoggingHandler` maps Python log levels to event statuses and
omits traceback and stack text. `capture_warnings(...)` can bridge emitted warnings; its
hook is process-global, so do not overlap warning bridges for separate runs. These manual
bridges remain useful when no automatic context is active.

## Tests and examples

Run the focused tests from the repository root:

```bash
.venv/bin/python -m pytest -q \
  tests/test_observability.py \
  tests/test_observability_console.py \
  tests/test_observability_runtime.py \
  tests/test_automatic_lens_observability.py \
  tests/test_automatic_analysis_observability.py \
  tests/test_automatic_io_observability.py
```

The examples above need no model. The recorder, runtime, and console modules are
Torch-free. The focused tests use synthetic or temporary artifacts.

`local/verify_automatic_observability.py` is a long local integration **self-test** for a
real SAELens checkpoint. It is not an experiment template or recommended workflow. Use
the minimal context or environment examples at the top of this page in application code.

## Logs are not results

JSONL order and delivery describe what the recorder observed. They are not a transaction
record for scientific output. A missing `completed` event does not prove that output is
absent, and a `completed` event does not validate an artifact. Never reconstruct
estimates, support, section availability, or report manifests from logs.

Use [durable analysis results](analysis-result.md) for computed tables and
[report bundles](report-bundle.md) for presentation artifacts.
