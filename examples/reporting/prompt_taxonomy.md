# Prompt-concept taxonomy (experimental recipe)

`prompt_taxonomy.py` classifies **the property represented by an existing named
prompt feature**, not every property present in its example prompts. It produces
caller-owned annotations, separate from the display-only `FeatureCatalog`.
It does not train a lens, verify feature names, compute distributions, sample
concept tuples, or change the Viewer.

## Inputs and meanings

Run from the repository root with its installed environment. Supply a saved
`FeatureBatch` and the exact name of its prompt view. That `FeatureMatrix` needs
`metadata.prompt` strings and `provenance.lens.feature_space_id`. Supply a CSV with:

| Column | Meaning |
|---|---|
| `feature_space_id` | Exact identity from the prompt matrix's lens provenance |
| `feature_id` | Integer coordinate ID from `matrix.feature_ids`, not column position |
| `pole` | `positive` or `negative`; opposite directions are separate rows |
| `concept_name` | Existing proposed name, unchanged; blank if unnamed |

Provide only the directions you want to classify. Duplicate feature/pole rows,
unknown feature IDs, wrong spaces and non-prompt matrices are rejected. A matrix
with nonnegative codes normally has only positive directions to name. Do not
invent negative names or convert separate pole-column IDs to signed coordinates
without an explicit source mapping.

The six dimensions are:

- **subject:** topic, such as astronomy or finance.
- **intent:** requested operation, such as explanation, comparison or translation.
- **audience:** intended recipient or expertise, such as children or beginners.
- **constraint:** output requirements, such as length, format, tone or method.
- **language:** natural-language identity, including a specified output language.
- **other:** a clear property outside the five dimensions, such as a greeting.

`assigned` means a taxonomy assignment only. It does **not** validate the concept
name or establish semantic presence. `mixed` and `insufficient_evidence` have an
empty dimension. They are not forced into `other`. Blank names and fewer than two
distinct activating prompts produce local `insufficient_evidence` without a call.
The categories do not make SAE features independent or guarantee coverage of
useful audience or constraint concepts. Review a small pilot before continuing.

## Dry run, pilot and resume

Choose the model yourself; the recipe has no default model. These commands use
illustrative paths. Dry run requires neither a key nor an API client and writes
nothing. It validates inputs and resume settings, then prints counts and total
request-text characters. Character counts are **not** token or cost estimates.

```bash
.venv/bin/python -m examples.reporting.prompt_taxonomy \
  --features artifacts/prompt-features --view z_prompt \
  --concepts artifacts/prompt-concepts.csv --out runs/prompt-taxonomy \
  --model YOUR_OPENROUTER_MODEL --limit 20 --dry-run
```

After inspecting the inputs, choose a model and set `OPENROUTER_API_KEY` in your
shell. **Removing `--dry-run` sends prompt text to the configured remote provider
and can incur charges.** Confirm that you may share that text. Names and examples
are delimited and escaped as untrusted data, but model judgments still need review.

```bash
# At most 20 new classifier calls. The output directory must not already exist.
.venv/bin/python -m examples.reporting.prompt_taxonomy \
  --features artifacts/prompt-features --view z_prompt \
  --concepts artifacts/prompt-concepts.csv --out runs/prompt-taxonomy \
  --model YOUR_OPENROUTER_MODEL --limit 20

# Same inputs/model/settings; omit limit to finish after reviewing the pilot.
.venv/bin/python -m examples.reporting.prompt_taxonomy \
  --features artifacts/prompt-features --view z_prompt \
  --concepts artifacts/prompt-concepts.csv --out runs/prompt-taxonomy \
  --model YOUR_OPENROUTER_MODEL --resume
```

`--limit` counts new classifier calls, not local abstentions or already saved
results. The existing client can retry provider requests, so it is **not** a hard
HTTP-request or monetary budget. Run only one process per output directory.
Changing the input values, row IDs, prompt text/identity, names, directions,
evidence settings, instructions or model settings rejects resume. Use a new
output directory for a changed experiment. API errors and invalid JSON raise;
they never become semantic abstentions. Already completed rows remain durable.
If a call finishes remotely but the process stops before saving its result,
resume may repeat that call. Usage is not an exactly-once billing ledger.

## Evidence and outputs

Defaults are six activating prompts and three non-activating contrasts per row,
seed 0 and at most 2,000 characters per prompt (head/tail truncation is marked).
The active sample takes the strongest half, then samples the remaining distinct
activating prompts. Contrasts exclude all identities active in that direction,
including active prompts not selected in the sample. Identity uses a supplied
`prompt_id`, then `instruction_id`, otherwise exact prompt text. Explicit IDs
remain authoritative even when different IDs have identical text. Blank prompt
text is not evidence. These are retrieval samples, not representative frequency
estimates. Non-activation is not proof that a named property is absent.

The run directory contains:

- `prompt_taxonomy.csv`: completed rows in input order, with exactly
  `feature_space_id,feature_id,pole,concept_name,dimension,status,reason`.
  Unfinished rows are absent; use the run record to distinguish partial runs.
- `run.json`: atomic authoritative results, full-input/request hashes, model and
  evidence settings, feature/pole identities, evidence row IDs, raw signed
  activations and truncation flags. It records completion state and counts.
  Retain the original feature batch and label CSV to reconstruct evidence text.
- `usage.jsonl`: the existing client's request ledger, including failed attempts.
- `usage.json`: cumulative usage summary, written even when a classification
  fails after the validated run starts. Provider-reported cost can be unavailable;
  it must not be interpreted as zero cost.

Resume regenerates the CSV from `run.json`, including after an interrupted CSV
write. Put human corrections in a **separate reviewed CSV**; do not edit the raw
CSV and expect resume to preserve those edits. The CLI records model settings,
not the API key. Do not put credentials in programmatic `model_settings`.
Treat outputs and usage logs as sensitive derived data.

For caller code, use `prepare_taxonomy(matrix, labels, ...)` and then
`run_taxonomy(plan, client, out_dir, model_settings={...}, limit=20)`. Supply an
existing OpenRouter-compatible `LLMClient` with a `UsageTracker`; `model_settings`
must accurately describe its model, endpoint and decoding options. Do not modify
the prepared plan. The CLI wires these pieces together with temperature 0 and
512 output tokens; all model/evidence options are listed by `--help`.
