# PrefScope CLI Reference

Installing PrefScope provides the `prefscope` console entry point. Every subcommand is
invoked as:

```
prefscope <command> [args]
```

(`interpret` has nested subcommands: `prefscope interpret name ...`.)

Source of truth: `prefscope/cli/commands/` (domain argument registration), composed by
`prefscope/cli/parser.py` (`build_parser`). Embedder/LLM defaults
come from `prefscope/config.py` (`CONFIG`) and `prefscope/interpret/llm.py`
(`DEFAULT_MODEL = "deepseek/deepseek-v3.2"`, `DEFAULT_API_BASE =
"https://openrouter.ai/api/v1"`). Config defaults: `embed_model_id =
"Qwen/Qwen3-Embedding-8B"`, `max_tokens = 4096`, `embed_batch_size = 32`,
`cache_dir` uses `PREFSCOPE_CACHE_DIR` when set, otherwise the operating system's
user cache directory (for example `~/.cache/prefscope` on Linux).

Commands at a glance:

| Group | Commands |
|-------|----------|
| Inspect / demo | `inspect`, `init-demo`, `sae-metrics`, `select-lens` |
| Corpus / embed | `prepare-dataset`, `build-corpus`, `embed-corpus`, `embed-prompts` |
| Lens build / publish | `build-lens`, `build-prompt-lens`, `package-lens` |
| Apply a lens | `analyze`, `encode-dataset`, `concepts`, `extract-concepts` |
| Interpret | `interpret name`, `interpret verify`, `interpret calibrate-presence`, `interpret classify-role`, `name-prompts`, `cluster-features` |
| Pipeline | `run` |
| Analyze | `win-relevance`, `associate-outcomes`, `screen-confounds`, `elicit`, `conditional-delta`, `compare-responses`, `context-profile`, `feature-relations` |
| Diagnose | `diagnose`, `report`, `build-bank`, `validate-diagnosis` |
| Token-SAE | `extract-activations`, `train-token-sae`, `summarize-activations` |

The recommended dataset-analysis path is `prepare-dataset → build-lens → run`; use
`concepts` or `report` for downstream analysis. `extract-concepts` is the short path
for applying an already published lens to one prompt and optional response. The
remaining commands expose individual stages for advanced and cluster-scheduled runs.

A **shared embedder flag block** appears on `build-lens`, `embed-corpus`,
`embed-prompts`, and `diagnose`. It is documented once below and referenced.

### Shared embedder flags

| flag | type / choices | default | meaning |
|------|----------------|---------|---------|
| `--device` | cuda \| mps \| cpu | `cuda` | embedder device |
| `--embed-model-id` | str | Qwen 8B config default; `diagnose` uses the lens manifest | embedding model id |
| `--embed-model-revision` | str | `None` | pinned model revision recorded in provenance |
| `--embed-batch-size` | int | `32` (CONFIG) | embed batch size |
| `--max-tokens` | int | `4096` (CONFIG) | max tokens per text |
| `--cache-dir` | str | `None` → CONFIG.cache_dir | embedding cache dir |
| `--cache-workers` | int | `32` | parallel threads reading cached embeddings |
| `--embed-backend` | hf \| vllm \| vllm-server | `hf` | embedding backend |
| `--tensor-parallel-size` | int | `1` | vLLM tensor-parallel GPUs |
| `--embed-api-base` | str | `None` | vllm-server OpenAI `/v1` URL |
| `--embed-api-key-env` | str | `OPENAI_API_KEY` | env var holding server API key |

---

## Inspect

### `inspect`
Battle-table sanity summary (corpus or annotations). Provide exactly one of
`--corpus` / `--annotations`.

```
prefscope inspect (--corpus PARQUET | --annotations JSON [JSON ...])
```

| flag | type | default | meaning |
|------|------|---------|---------|
| `--corpus` | str | `None` | merged corpus parquet from `build-corpus` (label-free) |
| `--annotations` | str+ (nargs `+`) | `None` | OpenJury annotation JSON(s) |

### `init-demo`

Write a deterministic synthetic corpus and a matching pipeline configuration. Unlike a
repository-relative example, this works from an ordinary wheel installation.

```bash
prefscope init-demo --out demo
```

The directory contains `sample_corpus.parquet` and `quickstart.yaml`; the command prints
the matching `build-lens` invocation. It refuses a non-empty directory unless `--force`
is explicit.

### `sae-metrics`
Redundancy + fit-health metrics for a lens (decoder cosine, MI, FVU, dead-frac,
L0). NOT an absorption score. Prints JSON; optionally appends a row to a CSV for
M-sweeps.

```
prefscope sae-metrics --lens-dir DIR [--out CSV]
```

| flag | type | default | meaning |
|------|------|---------|---------|
| `--lens-dir` | str | required | lens directory |
| `--out` | str | `None` | CSV to append a metrics row to (M-sweep tables) |

---

## Corpus / embed

### `build-corpus`
Build a merged label-free battle corpus from Hugging Face arenas.

```
prefscope build-corpus --source SRC [SRC ...] --out PARQUET [...]
```

| flag | type / choices | default | meaning |
|------|----------------|---------|---------|
| `--source` | str+ (nargs `+`) | required | arena sources, e.g. `lmarena-100k lmarena-140k comparia` |
| `--out` | str | required | output corpus parquet |
| `--split` | str | `train` | HF split |
| `--limit` | int | `None` | cap battles per source (quick trials) |
| `--hf-token-env` | str | `HF_TOKEN` | env var holding HF token (gated `comparia`) |
| `--keep-labels` | flag | `False` | carry the human vote as `human_pref` (= P(A preferred)) for win-relevance |

### `embed-corpus`
Embed one shard of a corpus into the cache (parallel multi-GPU pre-pass; then
run `build-lens` to train from the warm cache). No training.

```
prefscope embed-corpus --corpus PARQUET [--shard I --num-shards N] [embedder flags]
```

| flag | type | default | meaning |
|------|------|---------|---------|
| `--corpus` | str | required | merged corpus parquet |
| `--shard` | int | `0` | this shard index in `[0, num-shards)` |
| `--num-shards` | int | `1` | total shards (= parallel GPU processes) |

Plus all [shared embedder flags](#shared-embedder-flags).

### `embed-prompts`
Embed prompts alone → a `battle_id`-aligned `e_prompt.npy` (+ `meta.parquet`) for
the prompt lens. With `--num-shards > 1`, only warms the cache for the shard
(no dump).

```
prefscope embed-prompts --corpus PARQUET --out DIR [--shard I --num-shards N] [embedder flags]
```

| flag | type | default | meaning |
|------|------|---------|---------|
| `--corpus` | str | required | merged corpus parquet |
| `--out` | str | required | output dir for `e_prompt.npy` + `meta.parquet` |
| `--shard` | int | `0` | shard index |
| `--num-shards` | int | `1` | `>1`: only warm cache for this shard |

Plus all [shared embedder flags](#shared-embedder-flags).

---

## Lens build

### `build-lens`
Embed + train a frozen SAE lens. Provide exactly one of `--annotations` /
`--corpus` (or `--from-embeddings` to skip embedding).

```
prefscope build-lens (--annotations JSON... | --corpus PARQUET | --from-embeddings DIR) --out DIR [...]
```

| flag | type / choices | default | meaning |
|------|----------------|---------|---------|
| `--annotations` | str+ (nargs `+`) | `None` | OpenJury annotation JSON(s) |
| `--corpus` | str | `None` | merged corpus parquet |
| `--dump-embeddings` | str | `None` | also save assembled `e_a`/`e_b`/`meta` here for later `--from-embeddings` |
| `--from-embeddings` | str | `None` | train from a dumped embedding set (skip corpus + cache scan + embed) |
| `--out` | str | required | output lens directory |
| `--max-train-rows` | int | `None` | deterministic reservoir cap for SAE training rows |
| `--m-total` | int | `128` | SAE feature count M |
| `--k` | int | `16` | top-k active per row |
| `--matryoshka-prefix` | int+ (nargs `+`) | omitted (`[]`) | opt-in nested Matryoshka prefix lengths (m_total appended automatically) |
| `--whiten` | none \| standardize \| pca | `none` | input whitening (stored, re-applied at projection) |
| `--whiten-eps` | float | `1e-5` | whitening epsilon |
| `--input-rep` | difference \| individual | `difference` | SAE input: `e_a-e_b` (WIMHF) or pooled `[e_a; e_b]` |
| `--sae-type` | str (registered SAE) | `auto` | `difference` → signed `batchtopk`; `individual` → non-negative `batchtopk-relu`; explicit built-ins also include `signed-batchtopk`, `jumprelu`, `simple-topk` |
| `--sparsity-coef` | float | `1e-3` | jumprelu: L0 sparsity penalty λ |
| `--bandwidth` | float | `1e-3` | jumprelu: straight-through-estimator rectangle-kernel bandwidth ε |
| `--sparsity-warmup-steps` | int | `0` | jumprelu: linearly warm λ over this many optimizer steps; 0 disables |
| `--val-frac` | float | `0.1` | validation fraction |
| `--batch` | int | `512` | SAE train batch |
| `--n-epochs` | int | `200` | SAE epochs |
| `--seed` | int | `0` | RNG seed |

Plus all [shared embedder flags](#shared-embedder-flags).

### `build-prompt-lens`
Train a standard (non-difference) SAE on prompt embeddings (the prompt-concept
matrix). Reads an `embed-prompts` dump.

```
prefscope build-prompt-lens --from-embeddings DIR --out DIR [...]
```

| flag | type | default | meaning |
|------|------|---------|---------|
| `--from-embeddings` | str | required | `embed-prompts` dump dir (`e_prompt.npy` + `meta.parquet`) |
| `--out` | str | required | output prompt-lens directory |
| `--max-train-rows` | int | `None` | deterministic reservoir cap for SAE training prompts |
| `--m-total` | int | `64` | SAE feature count |
| `--k` | int | `8` | top-k active |
| `--matryoshka-prefix` | int+ | omitted (`[]`) | opt-in Matryoshka prefix lengths |
| `--sae-type` | str | `auto` | auto uses non-negative `batchtopk-relu`; explicit architectures are also accepted |
| `--sparsity-coef` | float | `1e-3` | jumprelu L0 penalty λ |
| `--bandwidth` | float | `1e-3` | jumprelu STE bandwidth ε |
| `--sparsity-warmup-steps` | int | `0` | jumprelu λ warmup steps |
| `--val-frac` | float | `0.1` | validation fraction |
| `--batch` | int | `512` | train batch |
| `--n-epochs` | int | `200` | epochs |
| `--seed` | int | `0` | seed |
| `--device` | cuda \| mps \| cpu | `cpu` | SAE device (default cpu, unlike build-lens) |
| `--embed-model-id` | str | `Qwen/Qwen3-Embedding-8B` | label only (recorded in manifest) |

Note: `build-prompt-lens` has no `--embed-model-id` embedding effect — it only
labels the manifest.

### `package-lens`

Create and validate a compact inference artifact for the Hugging Face Hub:

```bash
prefscope package-lens --lens-dir DIR --annotations INTERPRET_DIR \
  --model-card README.md --out RELEASE_DIR
```

The command builds a complete new directory, validates it, and then publishes it. It
copies the checkpoint, optional whitening transform, and annotations. It leaves out
corpus text and row-aligned training codes. `--overwrite` replaces the whole destination;
it does not merge old files.
See [Publish a lens](../how-to/publish-a-lens.md).

| flag | meaning |
|------|---------|
| `--lens-dir DIR` | required source lens directory |
| `--annotations PATH ...` | optional interpretation directory or CSV files |
| `--model-card FILE` | optional Markdown copied as `README.md` |
| `--device cpu|cuda|mps` | device used while validating the checkpoint; default `cpu` |
| `--out DIR` | required destination |
| `--overwrite` | replace an existing destination safely |

---

## Apply a lens

### `analyze`

Apply already-trained prompt and individual-response lenses to a new dataset through one
strict YAML config:

```bash
prefscope analyze --config analysis.yaml \
  [--data FILE | --hf-dataset OWNER/DATASET] [--out DIR] [--device DEVICE] \
  [--set PATH=VALUE ...] [--fresh]
```

The workflow composes `prepare-dataset`, frozen-lens encoding, concept export,
prompt–response relationships, applicable paired/preference analyses, and viewer-data
export. It never trains or interprets a lens. `--set` values use YAML syntax, so numbers,
booleans, lists, and `null` keep their types. Unknown keys are rejected. Named flags are
shortcuts for common overrides and take precedence over `--set`.
Set `viewer.output_dir` when the bundle should be written directly into a separate
PrefScope-Viewer checkout.

| flag | meaning |
|------|---------|
| `--config PATH` | required analysis config |
| `--set PATH=VALUE` | override any config key; repeat as needed |
| `--data FILE` / `--hf-dataset ID` | replace the configured dataset source |
| `--out DIR` | replace `out_dir` |
| `--repo ID` | replace the shared Hub lens repository |
| `--completion-lens SOURCE` / `--prompt-lens SOURCE` | replace either local or `hf://` lens |
| `--completion-subfolder NAME` / `--prompt-subfolder NAME` | replace Hub lens subfolders |
| `--revision REV` | replace the Hub lens revision |
| `--device DEVICE` | replace the configured device |
| `--presence-policy POLICY` | `calibrated`, `positive_nonzero`, or `mixed` |
| `--top-k N` | limit exported features per item |
| `--viewer` / `--no-viewer` | enable or disable viewer export |
| `--fresh` | replace a safely recognized analysis output instead of resuming |

To run rating analysis in the same workflow, retain each source rating in
`data.columns.metadata` and configure one outcome family:

```yaml
data:
  columns:
    metadata: [helpfulness, correctness, conversation_id]
analysis:
  group_col: conversation_id       # optional; otherwise group_id or prompt hash
  outcomes:
    columns: [helpfulness, correctness]
    kind: multi_continuous
    normalization: auto
    code_array: z_a
    min_units: 3
```

The workflow writes `outcome_associations.csv` plus normalization provenance. Its resolved
state fingerprints mutable Hub dataset and lens sources by their commit SHA.

Outputs are resumable only when the resolved config and the content fingerprints of
local datasets, lenses, and annotation inputs match. A changed input at the same path is
never treated as completed work. `--fresh` removes an existing output only when it carries
a valid PrefScope `analysis_state.json` ownership marker; it refuses filesystem roots,
the current directory, the home directory, Git repositories, unrecognized directories,
and directories that contain an input. Use another `--out` when in doubt. See
the [Analyze Config Schema](analyze-config-schema.md) for every accepted key and
[`examples/workflows/analyze-published-lenses.yaml`](../../examples/workflows/analyze-published-lenses.yaml)
for a runnable minimal config.

### `prepare-dataset`

Load a local table or Hugging Face dataset split and materialize PrefScope's
canonical schema. This separates source-specific parsing from model inference and
records the resolved mapping in `<out-stem>.prefscope.json`.

```bash
prefscope prepare-dataset \
  (--data INPUT | --hf-dataset OWNER/REPO | --spec dataset.yaml) \
  --out canonical.parquet [...]
```

The canonical columns are `prompt`, `completion_a`, optional `completion_b`,
optional `human_pref = P(A preferred)`, model identifiers, source, stable source
row/item ids, and scalar columns explicitly retained with repeatable `--keep-column` (or
`columns.metadata` in a spec). Use retained columns for ratings, outcome attributes, and
group identifiers. `--label-mode probability` accepts numeric values in `[0,1]`.
`--label-mode winner` requires explicit `--a-wins-value` and
`--b-wins-value` declarations. `--label-mode a-wins` is the shortcut for a
chosen/rejected layout. Structured conversation columns can be selected with
`--prompt-role user:first` and `--response-role assistant:last`. By default empty required
text rows are dropped; `--fail-on-empty` instead rejects and reports their source rows.

For large Hub sources, `--streaming --limit N` reads a bounded prefix without
downloading the complete split. `--hf-name`, `--split`, `--hf-revision`, and
`--hf-token-env` select a configuration, split, revision, and gated-data token. Mutable
Hub refs resolve before loading. The sidecar records requested/resolved commit SHAs and an
ordered `canonical_table_hash`; tokens are never persisted.

Important direct flags:

| flag | meaning |
|------|---------|
| `--data FILE` / `--hf-dataset ID` / `--spec FILE` | choose one local, Hub, or reusable-spec source; `--mapping` is an alias for `--spec` |
| `--out FILE` | required canonical output table |
| `--hf-name`, `--split`, `--hf-revision`, `--hf-token-env` | select the Hub subset, split, revision, and token environment variable |
| `--streaming --limit N` | stream a bounded Hub prefix |
| `--prompt-col`, `--response-col`, `--response-2-col` | map prompt and response columns |
| `--label-col`, `--label-mode` | map the preference column and its meaning |
| `--a-wins-value`, `--b-wins-value`, `--tie-value` | define winner tokens; each can be repeated |
| `--model-col`, `--model-2-col` | map model names |
| `--id-col`, `--language-col` | map source row IDs and languages |
| `--keep-column NAME` | retain a rating, group ID, or other scalar column; repeat as needed |
| `--prompt-role`, `--response-role`, `--response-2-role` | select messages from structured conversations |
| `--single` | force single-response mode |
| `--fail-on-empty` | reject empty required text instead of dropping its row |

A reusable YAML spec has this shape:

```yaml
source:
  type: huggingface
  path: organization/dataset
  split: train
  revision: main
columns:
  prompt: chosen
  response_a: chosen
  response_b: rejected
  metadata: [helpfulness, correctness, conversation_id]
text:
  prompt_role: user:first
  response_a_role: assistant:last
  response_b_role: assistant:last
label:
  mode: a-wins
```

### `encode-dataset`

Encode arbitrary `(prompt, response[, response_2])` data into reusable dense
`z_*.npy` sparse-code arrays plus aligned `meta.parquet`. The lens manifest
selects the embedding model.

```bash
prefscope encode-dataset \
  (--lens-dir DIR | --lens hf://OWNER/REPO[/SUBFOLDER]) \
  --data INPUT --out DIR [...]
```

An individual paired lens writes `z_a.npy`, `z_b.npy`, and `z_diff.npy`; a
single-response dataset writes `z_a.npy`; a prompt lens writes `z_prompt.npy`
and ignores response flags. Encoding builds a complete validated sibling staging bundle
and publishes it by whole-directory replacement; output may not overlap its source lens or
input data, and stale arrays cannot survive a rebuild. Non-empty destinations are refused
unless `--overwrite` explicitly requests a validated replacement. Every bundle contains
aligned `meta.parquet` and a `battles.parquet` compatibility view, so completion and prompt bundles from the
same prepared table can be passed directly to `elicit`. Canonical metadata keeps
`human_pref` when available. Repeat `--metadata-col COLUMN` to retain extra fields such
as a shared prompt-group identifier for downstream grouped inference.

| flag | meaning |
|------|---------|
| `--lens-dir DIR` / `--lens SOURCE` | choose a local lens, or a local/`hf://` source |
| `--revision`, `--subfolder`, `--hub-cache-dir` | select and cache a Hub lens version |
| `--hf-token-env`, `--local-files-only` | private-lens token variable and offline mode |
| `--data FILE`, `--out DIR` | required source table and output bundle |
| `--overwrite` | replace a non-empty output with one validated bundle |
| `--prompt-col`, `--response-col`, `--response-2-col` | map text columns |
| `--model-col`, `--model-2-col`, `--label-col` | preserve model and label columns |
| `--metadata-col NAME` | preserve another scalar column; repeat as needed |
| `--device`, `--embed-batch-size`, `--cache-dir`, `--cache-workers` | embedding device and cache settings |
| `--embed-backend`, `--tensor-parallel-size` | choose the embedding backend and vLLM GPU count |
| `--embed-api-base`, `--embed-api-key-env` | configure a `vllm-server` endpoint |

### `concepts`

Apply a local or Hugging Face prompt/individual lens and stream every active
feature to a filterable long table.

```bash
prefscope concepts --lens hf://owner/repository[/subfolder] \
  --data INPUT --out concepts.parquet [...]
```

| flag | type / choices | default | meaning |
|------|----------------|---------|---------|
| `--lens` | str | required | local directory or `hf://owner/repository[/subfolder]` |
| `--revision` | str | `None` | Hub branch, tag, or commit |
| `--subfolder` | str | `None` | lens directory inside a multi-lens Hub repository |
| `--hub-cache-dir` | str | `None` | local cache for Hub lens snapshots |
| `--hf-token-env` | str | `None` | environment variable holding a private-repository token |
| `--local-files-only` | flag | `False` | use only an already cached Hub snapshot |
| `--annotations` | path* | `None` | external interpretation CSV(s)/directory to merge |
| `--data` | str | required | parquet, CSV, JSONL, or JSON input |
| `--out` | str | required | long `.parquet`, `.csv`, or `.jsonl` output |
| `--prompt-col` | str | `prompt` | prompt column |
| `--response-col` | str | `response` | first response; ignored by prompt lenses |
| `--response-2-col` | str | `None` | optional second response; emits sides A and B |
| `--batch-size` | int | `128` | bounded inference/export batch |
| `--device` | cpu \| cuda \| mps | `cpu` | embedder and SAE device |
| `--pole` | any \| positive \| negative | `any` | signed activation pole to retain |
| `--min-abs-activation` | float | `0` | raw magnitude floor |
| `--top-k` | int | `None` | optional per-item cap; default keeps every active feature |
| `--include-zero` | flag | `False` | emit silent features too (can approach `N × M`) |
| `--fidelity-only` | flag | `False` | require bundled `fidelity_pass` |
| `--semantic-presence-only` | flag | `False` | apply bundled feature thresholds |
| `--include-text` | flag | `False` | duplicate text into every long-table row |

The output always includes source `row_id`, side, rank, feature id, raw and
absolute activation, pole, whether that pole matches the concept name,
semantic-presence status, and every annotation column bundled with the lens.

### `extract-concepts`

Apply prompt and response lenses to one literal prompt/completion pair and print a small
human-readable table (or JSON):

```bash
prefscope extract-concepts --repo OWNER/REPOSITORY \
  --prompt-subfolder prompt --completion-subfolder completion \
  --prompt "Explain why the sky is blue." \
  --completion "Short wavelengths scatter more strongly." --device cuda
```

Use `--prompt-lens PATH_OR_HF_SOURCE --completion-lens PATH_OR_HF_SOURCE` instead of
`--repo` for separately stored lenses. Prompt and completion lenses are independently
optional; supply at least one. Repository subfolders are explicit so experiment-specific
names or widths are never guessed. Use `--revision REV` to pin a Hub version.
`--presence-policy calibrated` omits uncalibrated axes, `positive_nonzero` treats every
positive firing as exploratory presence, and `mixed` records which basis was used for
each result. Verified names are required by default; `--include-unverified` relaxes that
gate. `--top 0` returns every retained concept and `--json` produces machine-readable
output.

### Preference analysis from an encoded dataset

`win-relevance` accepts either a built lens/corpus pair or the reusable output of
`encode-dataset`:

```bash
prefscope win-relevance \
  --encoded-dir analysis/codes --all-features \
  --out analysis/win_relevance.csv
```

It writes per-feature raw and length-controlled preference association plus
`win_relevance_summary.json` with labeled count, tie rate, decisive A win rate,
and A win rate treating a tie as half a win. Rows with a missing label are
excluded; string winner labels must first be normalized by `prepare-dataset`.
When both model columns were mapped, it also writes
`win_relevance_models.csv` with per-model battle count, wins/losses/ties, and
win rate treating ties as half a win.
For chosen/rejected layouts with the chosen response always on side A, the
winner-oriented `preferred_minus_rejected_mean`, `preferred_side_rate`, and
exact sign-test columns remain defined, while outcome-correlation/logistic
columns are correctly missing because the A/B outcome has no variation.

---

## Interpret

### `interpret name`
Label each feature from its top-activating examples. Completion lenses use their
pair/response source; prompt lenses use `--lens-kind prompt --corpus ...`.

```
prefscope interpret name --lens-dir DIR --out CSV (--annotations JSON... | --corpus PARQUET) [...]
# Prompt lens:
prefscope interpret name --lens-dir DIR --lens-kind prompt --corpus PARQUET --out CSV [...]
```

Common interpret flags (shared by `name` and `verify`):

| flag | type / choices | default | meaning |
|------|----------------|---------|---------|
| `--lens-dir` | str | required | lens directory |
| `--annotations` | str+ | `None` | annotation JSON(s) the lens was built from |
| `--corpus` | str | `None` | merged corpus parquet the lens was built from |
| `--out` | str | required | output CSV |
| `--backend` | openai \| claude-cli \| codex-cli | `openai` | LLM backend |
| `--model` | str | `deepseek/deepseek-v3.2` | LLM model id |
| `--api-base` | str | `https://openrouter.ai/api/v1` | OpenAI-compatible base URL |
| `--api-key-env` | str | `OPENROUTER_API_KEY` | env var holding API key |
| `--max-tokens` | int | `2000` | maximum output-token budget per interpretation request |
| `--reasoning-effort` | none \| minimal \| low \| medium \| high | `None` | optional OpenAI/OpenRouter reasoning control; `none` disables reasoning |
| `--verify-frac` | float | `0.2` | held-out fraction (carried into naming/verify split) |
| `--seed` | int | `0` | seed |
| `--concurrency` | int | `1` | features sent to the LLM in parallel (thread pool) |
| `--resume` | flag | enabled | skip matching feature rows already checkpointed at `--out` |
| `--fresh` | flag | disabled | discard the prior output, resume sidecar, and usage ledger before running |

`interpret name` and `interpret verify` checkpoint each completed feature atomically.
Re-running the same command resumes by default; it skips completed feature ids and keeps
the usage total cumulative across invocations. `<stem>.resume.json` records the lens,
input files, model, sampling settings, and interpreter implementation. A mismatch fails
instead of silently mixing methods; use `--fresh` only when you intentionally want a new
run. An existing CSV without this sidecar is never overwritten implicitly.

LLM-backed commands show cumulative request, input-token, output-token, and cost
accounting in their progress bar. They also write usage sidecars beside `--out`:
`<stem>.usage.json` is the cumulative summary and `<stem>.usage.jsonl` is the
append-only per-request audit log. OpenRouter costs are the exact provider-reported
charges in credits; another OpenAI-compatible server may report tokens but no cost.
Prompts and model responses are never stored in the usage log.

`name`-only flags:

| flag | type / choices | default | meaning |
|------|----------------|---------|---------|
| `--features` | int* (nargs `*`) | `None` | subset of feature ids (default all) |
| `--lens-kind` | completion \| prompt | `completion` | `prompt` reads `z_prompt.npy` and prompt text from `--corpus` |
| `--name-mode` | str | `auto` | interpreter strategy: `auto` picks individual/pairwise from a completion manifest; `--lens-kind prompt` uses `single-text`; explicit registered names are also accepted |
| `--n-active` | int | `10` | top-active examples per feature |
| `--n-zero` | int | `10` | zero/inactive examples per feature |
| `--negatives` | random \| close | `random` | choose random silent controls or harder similar controls |
| `--pole` | positive \| negative | `None` | pole to name on a signed lens; legacy signed individual lenses still require an explicit one-pole acknowledgement |
| `--abbreviate` | flag | `False` | run WIMHF abbreviate-concept step |
| `--debug-responses` | str | `None` | dir to dump each feature's raw LLM response |

### `interpret verify`
Held-out fidelity of named axes (close-negative falsification gate).

```
prefscope interpret verify --lens-dir DIR --names CSV --out CSV (--annotations JSON... | --corpus PARQUET) [...]
```

Shares the common interpret flags above, plus:

| flag | type / choices | default | meaning |
|------|----------------|---------|---------|
| `--names` | str | required | `feature_names.csv` from `name` |
| `--features` | int* | `None` | optional subset of feature IDs |
| `--verify-mode` | str | `auto` | verifier strategy: `auto` picks individual/pairwise from `input_rep`; registered names: `individual`, `pairwise`, `prompt` |
| `--n-per-bucket` | int | `10` | examples per active/zero bucket |
| `--sampling` | extremes \| random-active \| quantile-stratified | `extremes` | strongest activations, uniform ordinary activations, or weak-to-strong quantile coverage (`stratified-random` is a legacy alias for `random-active`) |
| `--n-examples` | int | `None` | total per-feature label budget; overrides per-bucket allocation |
| `--min-success-rate` | float | `0.8` | minimum parseable LLM response rate |
| `--min-bucket` | int | `5` | minimum successful labels in every required bucket |
| `--pole` | `positive` | `None` | required acknowledgement for positive-pole-only verification of a legacy signed individual/prompt lens |
| `--fidelity-threshold` | float | `0.3` | minimum positive correlation to pass (with Bonferroni p<0.05) |
| `--lens-kind` | completion \| prompt | `completion` | `prompt` verifies prompt-lens concepts on `z_prompt` + prompt text (needs `--corpus`) |
| `--negatives` | str | `random` | prompt verify: `random` silent prompts or `close` (needs `--embeddings`) |
| `--embeddings` | str | `None` | deprecated compatibility flag; prompt `close` controls use aligned prompt SAE code space |

The output records `verification_sampling` and `verification_scope`; results produced on
ordinary/random activations should not overwrite the default extreme-fidelity artifact.

### `interpret calibrate-presence`
Selects a feature-specific activation threshold and then confirms it on disjoint prompt
groups. Selection uses activation-stratified evidence only to fix the cutoff. Confirmation
uniformly samples independent groups conditional on activation above that fixed cutoff;
only its precision Wilson lower bound and the silent-control leakage Wilson upper bound
can set `presence_pass=True`.
By default only `fidelity_pass=True` names are calibrated.

```bash
prefscope interpret calibrate-presence --lens-dir DIR --names NAMES \
  --fidelity FIDELITY --out feature_calibration.csv (--corpus PARQUET | --annotations JSON...) [...]
```

Important flags are `--n-per-bin 4`, `--n-top 20`, `--n-zero 10`,
`--target-precision 0.8`, `--min-above 20`, and `--max-silent-rate 0.2`. `n-zero`
is a floor; confirmation increases it when needed for the silent-leakage Wilson UCB. Prompt groups
are split deterministically 50/50 from the signed seed; fewer than two independent groups
fail closed. Outputs distinguish `selection_status`, `confirmation_status`, selection and
confirmation precision/coverage, group counts, and phase-tagged audit samples. The command
has the same `--resume`/`--fresh`, usage ledger, backend, model, and concurrency contract
as name/verify. It also accepts `--features`, `--lens-kind`, `--pole`, `--batch-size`,
`--seed`, and `--verify-frac`. Use `--all-named` only when calibrating fidelity failures
intentionally.

### `context-profile`

Relate response features to aligned prompt features. The completion and prompt lens rows
must have the same IDs in the same order.

The command has two modes.

**Prompt-link mode** does not need calibration or an LLM. Omit `--calibration`. It tests
whether the strongest response-feature rows are enriched for prompt features across
several prompt-activation tails:

```bash
prefscope context-profile --completion-lens CDIR --prompt-lens PDIR \
  --prompt-names PDIR/prompt_feature_names.csv \
  --out CDIR/feature_prompt_linkage.csv
```

Main prompt-link controls are `--top-n`, `--min-top-examples`,
`--prompt-tail-fractions`, `--min-tail-overlap`, `--min-link-lift`,
`--link-q-threshold`, and `--min-link-scales`. Output rows report support, enrichment,
and stability across tail sizes.

**Calibrated context mode** requires `--calibration`. It uses semantic response presence
to classify prompt dependence and can also write model-specific context results:

```bash
prefscope context-profile --completion-lens CDIR --prompt-lens PDIR \
  --calibration CDIR/feature_calibration.csv \
  --prompt-names PDIR/prompt_feature_names.csv \
  --prompt-fidelity PDIR/prompt_feature_fidelity.csv \
  --prompt-clusters PDIR/prompt_feature_clusters.csv \
  --out CDIR/feature_context.csv \
  --model-out CDIR/model_feature_context.parquet
```

`--model-out` is valid only in calibrated mode. `--prompt-calibration` supplies prompt
thresholds. `--prompt-presence-policy` defaults to `mixed` for compatibility and can be
set to `calibrated`. Prompt contexts can overlap; PrefScope does not force each prompt
into one strongest concept. Category and model-support thresholds are exposed as flags;
run `prefscope context-profile --help` for their exact defaults.

Shared flags include `--names`, `--chunk-rows`, and `--min-context-occurrences`.
Calibrated mode also exposes:

| flag | default | meaning |
|------|---------|---------|
| `--min-model-context-battles` | `20` | rows needed for one model and context |
| `--min-model-context-discordant` | `3` | differing A/B pairs needed for a sign test |
| `--min-stable-contexts` | `3` | contexts needed for a stability claim |
| `--consistency-threshold` | `0.75` | required share of contexts with one direction |
| `--q-threshold` | `0.05` | multiple-test threshold |
| `--general-min-contexts` | `5` | context count needed for `general` |
| `--general-max-context-share` | `0.5` | largest allowed share from one context |
| `--general-max-prompt-dependence` | `0.5` | largest allowed prompt-dependence score |
| `--min-choice-ratio` | `0.15` | minimum paired model-choice rate |
| `--prompt-content-max-choice` | `0.15` | maximum choice rate for `prompt_content` |

### `interpret classify-role`

Experimental. Labels already-named response features by the kind of property they
describe and by their relation to the request, from prompt-response evidence. Requires
an individual completion lens; works on single-response lenses, where evidence is
scored on `z_a` alone and no paired response is shown.

```bash
prefscope interpret classify-role --lens-dir lenses/completion --corpus corpus.parquet \
    --names results/mylens/feature_fidelity.csv --out results/mylens/feature_roles.csv \
    --backend openai --model deepseek/deepseek-v3.2 --concurrency 8 --resume
```

| Flag | Type | Default | Meaning |
|------|------|---------|---------|
| `--names` | path | required | `feature_fidelity.csv`, or any `feature_id,concept` table |
| `--linkage` | path | — | `feature_prompt_linkage.csv`; combines statistical scope with the semantic label |
| `--all-named` | flag | off | include names without `fidelity_pass=True` |
| `--pole` | `positive` | — | required acknowledgement for a signed individual lens |
| `--n-top` | int | 6 | strongest unique-prompt examples per feature |
| `--n-random` | int | 2 | additional randomly drawn active examples |
| `--min-valid-examples` | int | 4 | labelled examples needed before a role is assigned |
| `--batch-size` | int | 4 | examples per LLM request |

Roles are `response_policy`, `presentation`, `reasoning_strategy`, `requested_task`,
`language`, `topic_content`, `mixed_or_unclear`; relations are `explicitly_requested`,
`elicited_or_implied`, `independently_chosen`, `unclear`. The common interpret flags
(`--backend`, `--model`, `--api-base`, `--api-key-env`, `--max-tokens`,
`--concurrency`, `--reasoning-effort`, `--resume`, `--fresh`, `--seed`, and
`--verify-frac`) apply. Use either `--corpus` or `--annotations`; `--features` limits the
feature IDs.

### `select-lens`

Picks a width and sparsity from a sweep of `sae-metrics` rows. Reconstruction improves
monotonically with capacity, so configurations are first screened for dead features,
duplicated decoder directions, realised sparsity against the target, and rows per
feature; the best fit among the survivors wins.

```bash
for M in 128 512 2048; do
  prefscope sae-metrics --lens-dir lenses/sweep_m$M --out sweep.csv
done
prefscope select-lens --sweep sweep.csv --n-rows 110000 --input-dim 4096
```

| Flag | Type | Default | Meaning |
|------|------|---------|---------|
| `--sweep` | path | required | CSV accumulated by repeated `sae-metrics --out` |
| `--n-rows` | int | — | training rows, to bound width by rows-per-feature |
| `--input-dim` | int | — | representation dimension, to report the expansion ratio |
| `--out` | path | — | annotated sweep table |

`--n-rows` matters for document-embedding lenses, which train on one vector per
document rather than one per token and so support far narrower dictionaries than
token-level SAEs.

### `name-prompts`
Compatibility spelling for `interpret name --lens-kind prompt`. It uses the same
per-feature checkpoint, strict resume signature, and cumulative usage ledger.

```
prefscope name-prompts --lens-dir DIR --corpus PARQUET --out CSV [...]
```

| flag | type / choices | default | meaning |
|------|----------------|---------|---------|
| `--lens-dir` | str | required | prompt lens dir (`z_prompt.npy`) |
| `--corpus` | str | required | corpus parquet (prompt text by `battle_id`) |
| `--out` | str | required | output `prompt_feature_names.csv` |
| `--features` | int* | `None` | subset of feature ids |
| `--n-active` | int | `10` | top-active prompts per feature |
| `--n-zero` | int | `10` | zero/inactive prompts per feature |
| `--negatives` | random \| close | `random` | random or similar silent prompt controls |
| `--verify-frac` | float | `0.2` | held-out fraction |
| `--seed` | int | `0` | random seed |
| `--debug-responses` | path | `None` | save raw LLM replies for debugging |
| `--backend` | openai \| claude-cli \| codex-cli | `openai` | LLM backend |
| `--model` | str | `deepseek/deepseek-v3.2` | LLM model |
| `--api-base` | str | `https://openrouter.ai/api/v1` | base URL |
| `--api-key-env` | str | `OPENROUTER_API_KEY` | API key env |
| `--concurrency` | int | `1` | parallel features |
| `--max-tokens` | int | `2000` | output-token budget for each naming call |
| `--reasoning-effort` | none \| minimal \| low \| medium \| high | `None` | optional reasoning control; `none` disables reasoning |
| `--pole` | positive \| negative | `None` | pole to name on a signed prompt lens |
| `--resume` | flag | enabled | resume matching completed features (default) |
| `--fresh` | flag | disabled | replace output/checkpoint/usage and start over |

The output is accompanied by `<stem>.resume.json`, `<stem>.usage.json`, and
`<stem>.usage.jsonl`, just like the canonical `interpret name` command.

### `feature-relations`

Find duplicate, specialized, coactive, same-name, and decoder-aligned feature pairs
without merging their identities.

```bash
prefscope feature-relations --lens-dir DIR --out feature_relations.csv [...]
```

| flag | type / choices | default | meaning |
|------|----------------|---------|---------|
| `--lens-dir` | str | required | lens directory containing the selected code arrays |
| `--names` | str | `None` | names/fidelity CSV; enables name-collision diagnostics |
| `--lens-kind` | completion \| prompt | `completion` | use completion or prompt codes |
| `--cluster-on` | difference \| individual | `individual` | completion relationship space |
| `--cofire-pole` | positive \| negative \| nonzero | `positive` | activation pole counted as firing |
| `--min-cooccur` | int | `30` | minimum shared activations |
| `--min-jaccard` | float | `0.05` | minimum Jaccard overlap |
| `--min-containment` | float | `0.50` | minimum directional containment |
| `--min-phi` | float | `0.05` | minimum phi association |
| `--min-lift` | float | `1.50` | minimum coactivation lift |
| `--min-name-similarity` | float | `0.80` | minimum normalized name similarity |
| `--min-decoder-cosine` | float | `0.70` | minimum decoder-direction cosine |
| `--fidelity-only` | flag | `False` | keep only fidelity-passing axes |
| `--no-decoder` | flag | `False` | skip decoder similarity and its torch requirement |
| `--out` | str | required | relationship CSV; also writes a `_summary.csv` sibling |

### `cluster-features`
Group co-activating SAE features into corpus-specific feature groups.

```
prefscope cluster-features --lens-dir DIR --out CSV [...]
```

| flag | type / choices | default | meaning |
|------|----------------|---------|---------|
| `--lens-dir` | str | required | lens dir with `z_diff.npy` |
| `--names` | str | `None` | `feature_fidelity`/`feature_names` CSV (concepts + fidelity) |
| `--n-clusters` | int | `10` | target clusters (k-means clusterers) |
| `--method` | str | `spherical-kmeans` | clusterer component: recommended `cofire-leiden`, legacy `mi-leiden`, `spherical-kmeans`, `agglomerative`, or any registered |
| `--resolution` | float | `1.0` | Leiden resolution (higher → more, smaller communities) |
| `--knn` | int | `0` | Leiden graph sparsity (use 6–20 for `cofire-leiden`) |
| `--knn-mode` | mutual \| union | `mutual` | mutual avoids weak one-sided bridges; union reproduces legacy graph construction |
| `--affinity-metric` | phi \| cosine \| npmi | `phi` | positive co-presence weight for `cofire-leiden` |
| `--cofire-pole` | positive \| negative \| nonzero | `positive` | interpreted signed pole counted as concept presence |
| `--min-cooccur` | int | `30` | minimum co-firing support for a graph edge |
| `--stability-runs` | int | `5` | Leiden seeds for adjusted-Rand stability diagnostics |
| `--min-cluster-size` | int | `1` | diagnostic threshold; small cofire communities remain distinct |
| `--small-community-policy` | preserve \| merge | `preserve` | `merge` exists only for legacy reproduction |
| `--super-resolution` | float | `None` | optionally group fine communities into hierarchical superclusters |
| `--super-knn` | int | `4` | mutual-kNN sparsity for optional superclusters |
| `--fidelity-only` | flag | `False` | cluster only fidelity-passing features |
| `--cluster-on` | difference \| individual | `difference` | co-firing space: `z_diff` or stacked `z_a`/`z_b` |
| `--lens-kind` | completion \| prompt | `completion` | `prompt` clusters `z_prompt.npy` |
| `--name-clusters` | flag | `False` | LLM-name each behavior from member concepts |
| `--backend` | openai \| claude-cli \| codex-cli | `openai` | LLM backend (for `--name-clusters`) |
| `--model` | str | `deepseek/deepseek-v3.2` | LLM model |
| `--api-base` | str | `https://openrouter.ai/api/v1` | base URL |
| `--api-key-env` | str | `OPENROUTER_API_KEY` | API key env |
| `--concurrency` | int | `1` | parallel cluster naming |
| `--out` | str | required | membership CSV; also writes `_summary.csv` and `_diagnostics.csv` siblings |

---

## Pipeline

### `run`
Run a config-driven pipeline (name/verify/cluster/win-relevance) from a YAML/JSON
file; every component is selected by name + params in the config. See
`config-schema.md`.

```
prefscope run --config FILE.yaml
```

| flag | type | default | meaning |
|------|------|---------|---------|
| `--config` | str | required | pipeline config (`.yaml` / `.yml` / `.json`) |

The runner writes cumulative LLM accounting to `out_dir/llm_usage.json` and an
append-only, interruption-safe event ledger to `out_dir/llm_usage.jsonl`, broken
down by `name`, `verify`, and LLM-assisted `cluster` stages and by served model.

---

## Analyze

### `win-relevance`
Report which signed response features are associated with human or judge preference.
The corpus must contain `human_pref`; this can come from `build-corpus --keep-labels`.
The result describes this dataset and does not show that a feature caused preference.

When prompt groups are available, every group receives the same total weight. PrefScope
does not report p-values or q-values unless there are at least ten independent groups and
at least five groups on each side of the tested split. Each row states the estimated
quantity, test, row count, and group count.

**Method details.** The preferred-side test uses a bounded group-mean test. The
length-controlled logistic model uses an HC1 group-clustered Wald test with a `G-1`
t reference. When every row is its own group, PrefScope keeps the historical row-level
tests.

```
prefscope win-relevance --lens-dir DIR --corpus PARQUET --out CSV [...]
```

| flag | type | default | meaning |
|------|------|---------|---------|
| `--lens-dir` | str | required | lens directory |
| `--corpus` | str | conditionally required | corpus parquet WITH `human_pref`; use with `--lens-dir` |
| `--encoded-dir` | str | conditionally required | applied code bundle alternative to lens + corpus |
| `--names` | str | `None` | feature_names/fidelity CSV to attach concepts + filter |
| `--all-features` | flag | `False` | score every feature, not just fidelity-passing |
| `--clusters` | str | `None` | `feature_clusters.csv` → also emit `<out>_clusters.csv` cluster-level win-relevance |
| `--group-col` | str | `None` | independent prompt group; defaults to `group_id` or normalized prompt hash |
| `--out` | str | required | output win-relevance CSV |

### `associate-outcomes`

Associate frozen sparse concepts with binary labels, probabilities/preferences,
continuous ratings, or several continuous attributes. This is descriptive and
post-training-dataset-specific; it is not a causal or objective good/bad judgment.

```bash
prefscope associate-outcomes --encoded-dir codes \
  --outcome-col helpfulness --outcome-col correctness \
  --outcome-kind multi_continuous --out outcome_associations.csv
```

| flag | type / choices | default | meaning |
|------|----------------|---------|---------|
| `--encoded-dir` | str | required | `encode-dataset` bundle with `meta.parquet` and codes |
| `--outcome-col` | str, repeatable | required | metadata outcome column; repeat for `multi_continuous` |
| `--outcome-kind` | binary \| probability \| preference \| continuous \| multi_continuous | required | validation and scale contract |
| `--normalization` | auto \| none \| zscore | `auto` | bounded scales stay natural; continuous scales z-score under `auto` |
| `--code-array` | auto \| z_a \| z_diff \| z_prompt | `auto` | sparse representation to associate; auto prefers `z_a` |
| `--names` | str | `None` | optional feature-name CSV |
| `--group-col` | str | `None` | independent group column; defaults to `group_id` or normalized prompt hash |
| `--no-grouping` | flag | `False` | explicitly request row-level descriptive associations |
| `--min-units` | int | `3` | minimum rows or independent groups per attribute |
| `--out` | str | required | long-form association CSV; also writes `_outcomes.json` normalization provenance |

Missing outcomes remain missing and are omitted separately per attribute. Grouped mode
first averages feature and outcome values within each prompt group, then gives groups
equal weight. Continuous normalization is recomputed across group means, so duplicating
rows cannot change grouped slopes. Fisher-exact range-midpoint p/q values run only when both feature and
outcome arms have at least five independent units; Pearson correlations and OLS slopes
remain descriptive effects, and thin cells retain descriptive
correlations with `inference_supported=False`. BH correction covers supported tests.

### `screen-confounds`

Screen each response feature for entanglement between preference association and the
A-minus-B word-count gap. All correlations for a feature use the same nonzero rows.
The `confound_entangled` flag also covers an unidentified residual under near-perfect
collinearity; it is a screening result, not a claim that a concept is spurious or
undesirable. Preference-association significance uses prompt-group-aware inference. The
legacy permutation null is refused when prompt groups repeat because row permutations
would violate the independent-unit contract.

```bash
prefscope screen-confounds \
  --lens-dir DIR --corpus PARQUET \
  --names feature_fidelity.csv --out bias_screen.csv
```

| flag | type | default | meaning |
|------|------|---------|---------|
| `--lens-dir` | str | required | completion lens containing `z_diff.npy` |
| `--corpus` | str | required | aligned corpus with `human_pref` and both completions |
| `--names` | str | `None` | optional feature annotation CSV to attach |
| `--confound-threshold` | float | `0.3` | minimum absolute feature/length correlation |
| `--collapse-fraction` | float | `0.5` | maximum residual/original outcome-correlation ratio |
| `--permute` | int | `0` | optional preference-label permutation-null repetitions |
| `--seed` | int | `0` | permutation seed |
| `--group-col` | str | `None` | independent prompt group for preference inference |
| `--out` | str | required | output CSV |

### `elicit`
Prompt-concept → response-concept co-activation lift (preference-independent):
which response concepts appear when a prompt concept is present. With repeated prompts,
row counts and lift remain descriptive while a distribution-free two-sample Hoeffding
bound operates on per-group response prevalence. Prompt membership must be constant within each group. Bonferroni
correction covers the complete attempted prompt × response family.

```
prefscope elicit --completion-lens DIR --prompt-lens DIR --out CSV [...]
```

| flag | type | default | meaning |
|------|------|---------|---------|
| `--completion-lens` | str | required | individual lens dir (`z_a.npy`; `z_b.npy` optional) |
| `--prompt-lens` | str | required | prompt lens dir (`z_prompt.npy`) |
| `--completion-names` | str | `None` | `feature_names.csv` (response concepts) |
| `--completion-fidelity` | str | `None` | `feature_fidelity.csv` → restrict to verified response axes |
| `--prompt-names` | str | `None` | `prompt_feature_names.csv` |
| `--prompt-fidelity` | str | `None` | `prompt_feature_fidelity.csv` → restrict to verified prompt axes |
| `--min-support` | int | `30` | minimum firing rows; with repeats, also minimum independent present groups for inference |
| `--min-cooccur` | int | `5` | min co-occurrences to test a cell |
| `--group-col` | str | `None` | independent group in completion-lens battle metadata; defaults to `group_id` or prompt hash |
| `--out` | str | required | output elicitation CSV |

### `conditional-delta`
Prompt-conditioned completion delta Δ_{k,f} (which response properties
distinguish the winner when each prompt concept is present) + optional
conditional δ_{f,k}. Prompt regions overlap: every positive concept above the
activation floor is retained rather than forcing one dominant `argmax` label. With
repeated prompts, Δ uses a two-sample Hoeffding bound over per-group mean signed
prevalence, conditional δ uses equal-group weighting and cluster-robust inference on the
exact decisive finite fit rows, and permutation moves whole prompt groups.

```
prefscope conditional-delta --completion-lens DIR --prompt-lens DIR --out CSV [...]
```

| flag | type | default | meaning |
|------|------|---------|---------|
| `--completion-lens` | str | required | paired completion lens dir (`z_diff.npy`); winner-oriented runs require `input_rep=individual` |
| `--prompt-lens` | str | required | prompt lens dir (`z_prompt.npy`) |
| `--corpus` | str | `None` | corpus WITH `human_pref` — orients an individual lens's antisymmetric `z_diff=f(e_a)-f(e_b)` toward the winner (required for `--conditional-out`); direct-difference lenses are rejected because their nonlinear codes cannot be reversed by negation |
| `--completion-names` | str | `None` | response concept names |
| `--prompt-names` | str | `None` | prompt concept names |
| `--prompt-clusters` | str | `None` | `prompt_feature_clusters.csv` → condition on prompt CLUSTERS |
| `--conditional-out` | str | `None` | also emit length-controlled conditional win-rate δ_{f,k} |
| `--completion-fidelity` | str | `None` | restrict conditional table to verified axes |
| `--prompt-fidelity` | str | `None` | restrict membership to verified prompt axes |
| `--min-prompt-activation` | float | `0` | positive activation required for membership |
| `--min-prompt-support` | int | `30` | minimum battles containing a prompt concept/cluster |
| `--seed` | int | `0` | seed |
| `--group-col` | str | `None` | independent group; defaults to `group_id` or normalized prompt hash |
| `--permute` | int | `0` | permutation null: shuffle whole prompt-membership groups N times |
| `--jobs` | int | `1` | parallelize permutation null across N processes |
| `--out` | str | required | output Δ_{k,f} CSV |

---

## Diagnose

### `report`

Write a human-readable Markdown report card for one model. Provide exactly one evidence
source through `--annotations` or `--corpus`; optional win-relevance, prompt-lens, or bank
artifacts add preference gaps, prompt-type sections, or pool-relative effects.

```bash
prefscope report --lens-dir DIR --model MODEL --corpus corpus.parquet \
  --names feature_fidelity.csv --win-relevance win_relevance.csv --out report.md
```

| flag | default | meaning |
|------|---------|---------|
| `--lens-dir`, `--model`, `--out` | required | lens, target model, Markdown output |
| `--annotations` / `--corpus` | `None` | alternative aligned evidence sources |
| `--names`, `--win-relevance`, `--bank` | `None` | optional annotation/preference/bank tables |
| `--prompt-lens`, `--prompt-names` | `None` | add prompt-type sections |
| `--top` | `15` | concepts per section |
| `--min-battles` | `20` | prompt-section support threshold |
| `--min-prompt-activation` | `0` | positive prompt membership floor |
| `--all-features` | false | include axes without a fidelity pass |
| embedding/cache/device flags | lens/config defaults | inference backend for raw corpus/annotations |

### `diagnose`
Aggregate a target model's contrast codes into per-feature tendencies.

```
prefscope diagnose --lens-dir DIR --annotations JSON... --model NAME --out CSV [...]
```

| flag | type / choices | default | meaning |
|------|----------------|---------|---------|
| `--lens-dir` | str | required | frozen lens directory |
| `--annotations` | str+ (nargs `+`) | required | OpenJury JSON(s) containing the target model |
| `--model` | str | required | target model name to diagnose |
| `--out` | str | required | output diagnosis CSV |
| `--battles-out` | str | `None` | optional parquet of per-battle evidence (for the viewer) |
| `--bank` | str | `None` | oriented-code bank dir (from `build-bank`); adds inside-vs-outside Welch contrast, sorts by `delta_vs_pool` |
| `--fidelity` | str | `None` | `feature_fidelity.csv`; attaches concepts and restricts to passing axes |
| `--win-relevance` | str | `None` | merge global `delta_win_rate` as the descriptive `helps_win` field |
| `--all-features` | flag | `False` | diagnose every feature, not just fidelity-passing |
| `--top` | int | `10` | how many over/under-expressed features to print |

`--embed-model-id` here defaults to `None` (falls back to the lens manifest's
`embed_model_id`; leave it unset unless you are auditing a deliberate override). The
other shared embedder flags apply **except** `--embed-model-revision`, which `diagnose`
does not accept.

### `build-bank`
Project every battle in BOTH orientations → pool baseline for `diagnose --bank`
and `validate-diagnosis`.

```
prefscope build-bank --lens-dir DIR --from-embeddings DIR --out DIR [...]
```

| flag | type / choices | default | meaning |
|------|----------------|---------|---------|
| `--lens-dir` | str | required | frozen lens directory |
| `--from-embeddings` | str | required | dumped embedding dir (`e_a.npy`/`e_b.npy`/`meta.parquet`) |
| `--label` | judge \| human | `judge` | orient outcomes by `y_judge` or human preference (needs `--corpus`) |
| `--corpus` | str | `None` | corpus parquet with `human_pref` (for `--label human`) |
| `--out` | str | required | output bank directory |
| `--device` | cuda \| mps \| cpu | `cpu` | device for SAE forward pass |

### `validate-diagnosis`
Does the diagnosed deficit predict actual win rate? (R² across models.)

```
prefscope validate-diagnosis --bank DIR --win-relevance CSV --out CSV [...]
```

| flag | type | default | meaning |
|------|------|---------|---------|
| `--bank` | str | required | oriented-code bank dir (`build-bank`) |
| `--win-relevance` | str | required | win-relevance CSV (feature reward weights) |
| `--out` | str | required | output per-model CSV |
| `--weight-col` | str | `delta_win_rate` | length-controlled AME column used to weight features |
| `--all-features` | flag | `False` | weight by every feature, not just significant |
| `--min-battles` | int | `20` | skip models with fewer oriented battles |
| `--seed` | int | `0` | random seed for bootstrap and permutation checks |
| `--loo` | flag | `False` | leave-one-model-out: refit reward weights excluding each model's battles |

---

## Token-SAE

### `extract-activations`
Extract layer-L token activations from any HF causal LM into a memmap cache.

```
prefscope extract-activations --corpus PARQUET --out DIR [...]
```

| flag | type / choices | default | meaning |
|------|----------------|---------|---------|
| `--corpus` | str | required | corpus parquet |
| `--out` | str | required | output cache dir |
| `--model-id` | str | `meta-llama/Llama-3.1-8B-Instruct` | HF causal LM |
| `--layer` | int | `24` | hidden layer to extract |
| `--n-battles` | int | `30000` | random subsample size; `0` = all |
| `--max-tokens` | int | `512` | max tokens per span |
| `--outlier-norm-mult` | float | `6.0` | outlier-norm clipping multiplier |
| `--device` | cuda \| cpu | `cuda` | device |
| `--dtype` | str | `bfloat16` | torch dtype |
| `--attn-implementation` | str | `sdpa` | HF attn backend (`eager` is the AMD/ROCm fallback) |
| `--seed` | int | `0` | subsample seed |

Extraction refuses a non-empty `--out` directory. It never truncates or merges an
existing activation cache; choose a new directory or remove the old cache explicitly.
The response span starts after the tokenizer's assistant-generation prefix, which is
kept exactly once in the assembled chat sequence.

### `train-token-sae`
Stream-train a BatchTopK SAE from an activation cache.

```
prefscope train-token-sae --cache DIR --out DIR [...]
```

| flag | type / choices | default | meaning |
|------|----------------|---------|---------|
| `--cache` | str | required | `extract-activations` cache dir |
| `--out` | str | required | output SAE dir |
| `--expansion` | int | `8` | `m_total = expansion * hidden_dim` (ignored if `--m-total` set) |
| `--m-total` | int | `0` | explicit feature count; overrides `--expansion` |
| `--k` | int | `64` | top-k active |
| `--matryoshka-prefix` | int+ | omitted (`[]`) | opt-in Matryoshka prefix lengths |
| `--val-frac` | float | `0.05` | validation fraction |
| `--max-train-tokens` | int | `40000000` | reservoir cap on training rows |
| `--epochs` | int | `2` | epochs |
| `--batch` | int | `4096` | train batch |
| `--seed` | int | `0` | seed |
| `--device` | cuda \| cpu | `cuda` | device |

### `summarize-activations`
Project cached activations through the SAE → per-span X^max / X^freq.

```
prefscope summarize-activations --cache DIR --sae DIR --out DIR [...]
```

| flag | type / choices | default | meaning |
|------|----------------|---------|---------|
| `--cache` | str | required | activation cache dir |
| `--sae` | str | required | `train-token-sae` output dir |
| `--out` | str | required | output summaries dir |
| `--batch` | int | `8192` | projection batch |
| `--device` | cuda \| cpu | `cuda` | device |
### `compare-responses`

Compute label-free, prompt-matched concept shifts between response A and response B from
an individual-lens `encode-dataset` bundle:

```bash
prefscope compare-responses \
  --encoded-dir response_codes \
  --features response_interpret \
  --prompt-encoded-dir prompt_codes \
  --prompt-features prompt_interpret \
  --side-a-name base --side-b-name adapted \
  --out comparison
```

| flag | default | meaning |
|------|---------|---------|
| `--encoded-dir` / `--features` | required | paired codes and response annotations |
| `--prompt-encoded-dir` / `--prompt-features` | `None` | optional aligned prompt codes/annotations |
| `--prompt-clusters` | `None` | condition on prompt clusters instead of raw concepts |
| `--side-a-name` / `--side-b-name` | `A` / `B` | display labels |
| `--presence-policy` / `--prompt-presence-policy` | `calibrated` | calibrated, positive_nonzero, or mixed |
| `--include-unverified` / `--include-unnamed` | false | widen the analyzed feature set explicitly |
| `--min-context-pairs` | `30` | minimum independent support per prompt context |
| `--group-col` | `None` | repeated-prompt group metadata column |
| `--examples-per-direction` | `3` | strongest examples saved per direction |
| `--confidence` | `0.95` | distribution-free interval confidence |
| `--out` | required | output directory |

The default `--presence-policy calibrated` excludes axes without passing semantic
thresholds. `mixed` and `positive_nonzero` are explicit exploratory alternatives.
Grouping defaults to canonical `group_id` or a stable normalized-prompt hash.
`--group-col` selects another repeated-generation identifier; preserve custom fields while
encoding with `--metadata-col`. Preference labels are never used by this command. See
[Compare response sets](../how-to/compare-response-sets.md).
