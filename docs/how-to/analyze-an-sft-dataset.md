# Analyze an SFT dataset

Instruction-tuning data has one response per prompt and no preference labels. PrefScope
analyzes it end to end: what concepts the prompts contain, what concepts the responses
contain, and which response concepts each prompt concept elicits.

Analyses that compare two responses or read a preference — `win-relevance`,
`diagnose`, `report`, `build-bank`, `conditional-delta`, `screen-confounds` — do not
apply and refuse with a message naming the missing column.

## 1. Prepare the data

`--single` forces single-response mode instead of looking for a second response.
Structured chat columns are selected with role selectors:

```bash
prefscope prepare-dataset \
  --hf-dataset organization/dataset --split train \
  --prompt-col messages   --prompt-role   user:first \
  --response-col messages --response-role assistant:last \
  --language-col language --single \
  --out analysis/sft.parquet

prefscope inspect --corpus analysis/sft.parquet
```

`inspect` reports `"paired": false` for this data.

## 2. Build both lenses

The response lens must use `--input-rep individual`; a difference lens has no second
response to contrast against and is refused. Build both from the same table so their
rows stay aligned — `elicit` and `context-profile` require it.

```bash
prefscope build-lens --corpus analysis/sft.parquet \
    --input-rep individual --out lenses/completion \
    --embed-model-id Qwen/Qwen3-Embedding-8B --device cuda

prefscope embed-prompts --corpus analysis/sft.parquet --out embeddings/prompts \
    --embed-model-id Qwen/Qwen3-Embedding-8B --device cuda
prefscope build-prompt-lens --from-embeddings embeddings/prompts \
    --out lenses/prompt --device cuda
```

The response lens writes `z_a.npy` and records `dataset_mode: single`; no `z_b.npy` or
`z_diff.npy` is produced.

Choose `--m-total`/`--k` from a sweep rather than by hand — see
[`select-lens`](../reference/cli.md#select-lens). Width is bounded by rows per feature
here, because one vector per document is far fewer training vectors than a token-level
SAE sees.

## 3. Name, verify, and label

```bash
export OPENROUTER_API_KEY=...

prefscope interpret name --lens-dir lenses/completion --corpus analysis/sft.parquet \
    --out results/sft/feature_names.csv --lens-kind completion --resume
prefscope interpret verify --lens-dir lenses/completion --corpus analysis/sft.parquet \
    --names results/sft/feature_names.csv --out results/sft/feature_fidelity.csv \
    --lens-kind completion --resume
prefscope interpret classify-role --lens-dir lenses/completion \
    --corpus analysis/sft.parquet --names results/sft/feature_fidelity.csv \
    --out results/sft/feature_roles.csv --resume
```

`classify-role` separates response behavior (policy, presentation, reasoning) from
properties fixed by the prompt (requested task, language, topic). On instruction data
much of the dictionary is topic and language, so this is what makes the table legible.

For a one-shot config-driven run, the same three stages can be declared together:

```yaml
lens_dir: lenses/completion
out_dir: results/sft
stages: [name, verify, classify-role]
llm: {backend: openai, model: deepseek/deepseek-v3.2}
role_classifier: {n_top: 6, n_random: 2, concurrency: 4}
```

Run it with `prefscope run --config sft-interpret.yaml`. Keep the explicit commands above
for long remote runs where per-feature `--resume` checkpoints are important.

Repeat `name` and `verify` for the prompt lens with `--lens-kind prompt`.

## 4. Relate prompts to responses

`elicit` is preference-independent, so a single `(prompt, completion)` corpus is enough:

```bash
prefscope elicit --prompt-lens lenses/prompt --completion-lens lenses/completion \
    --prompt-names results/sft/prompt_feature_names.csv \
    --names results/sft/feature_fidelity.csv \
    --out results/sft/prompt_response_elicitation.csv
```

## 5. Export the viewer bundle

```bash
prefscope-export-viewer --lens-dir lenses/completion --analysis-dir results/sft \
    --corpus analysis/sft.parquet \
    --prompt-lens lenses/prompt --prompt-interpret-dir results/sft \
    --elicitation results/sft/prompt_response_elicitation.csv \
    --out viewer-data --joint-examples --feature-map --response-map
```

The export detects the single-response lens and skips the model and preference
artifacts, reporting each as skipped rather than writing an empty file. You get concept
distribution, all-axis co-activation, per-feature examples, the searchable SAE feature
atlas, the sampled response scatter, and the prompt→response relation.
