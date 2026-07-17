# PrefScope CLI Reference

Installing PrefScope provides the `prefscope` console entry point. Every subcommand is
invoked as:

```
prefscope <command> [args]
```

(`interpret` has nested subcommands: `prefscope interpret name ...`.)

Source of truth: `prefscope/__main__.py` (`build_parser`). Embedder/LLM defaults
come from `prefscope/config.py` (`CONFIG`) and `prefscope/interpret/llm.py`
(`DEFAULT_MODEL = "deepseek/deepseek-v3.2"`, `DEFAULT_API_BASE =
"https://openrouter.ai/api/v1"`). Config defaults: `embed_model_id =
"Qwen/Qwen3-Embedding-8B"`, `max_tokens = 4096`, `embed_batch_size = 32`,
`cache_dir = <PROJECT_ROOT>/data/cache`.

Commands at a glance:

| Group | Commands |
|-------|----------|
| Inspect | `inspect`, `sae-metrics` |
| Corpus / embed | `build-corpus`, `embed-corpus`, `embed-prompts` |
| Lens build | `build-lens`, `build-prompt-lens` |
| Interpret | `interpret name`, `interpret verify`, `name-prompts`, `cluster-features` |
| Pipeline | `run` |
| Analyze | `win-relevance`, `elicit`, `conditional-delta` |
| Diagnose | `diagnose`, `build-bank`, `validate-diagnosis` |
| Token-SAE | `extract-activations`, `train-token-sae`, `summarize-activations` |

A **shared embedder flag block** appears on `build-lens`, `embed-corpus`,
`embed-prompts`, and `diagnose`. It is documented once below and referenced.

### Shared embedder flags

| flag | type / choices | default | meaning |
|------|----------------|---------|---------|
| `--device` | cuda \| mps \| cpu | `cuda` | embedder device |
| `--embed-model-id` | str | `Qwen/Qwen3-Embedding-8B` (CONFIG) | embedding model id |
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
Build a merged label-free battle corpus from HuggingFace arenas.

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
| `--m-total` | int | `128` | SAE feature count M |
| `--k` | int | `16` | top-k active per row |
| `--matryoshka-prefix` | int+ (nargs `+`) | `[8]` | nested Matryoshka prefix lengths (m_total appended automatically) |
| `--whiten` | none \| standardize \| pca | `none` | input whitening (stored, re-applied at projection) |
| `--whiten-eps` | float | `1e-5` | whitening epsilon |
| `--input-rep` | difference \| individual | `difference` | SAE input: `e_a-e_b` (WIMHF) or pooled `[e_a; e_b]` |
| `--sae-type` | str (registered SAE) | `batchtopk` | SAE architecture; built-in `batchtopk`/`jumprelu`/`simple-topk` or any registered (`jumprelu`: pair with `--input-rep individual`) |
| `--sparsity-coef` | float | `1e-3` | jumprelu: L0 sparsity penalty λ |
| `--bandwidth` | float | `1e-3` | jumprelu: straight-through-estimator rectangle-kernel bandwidth ε |
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
| `--m-total` | int | `64` | SAE feature count |
| `--k` | int | `8` | top-k active |
| `--matryoshka-prefix` | int+ | `[8]` | Matryoshka prefix lengths |
| `--val-frac` | float | `0.1` | validation fraction |
| `--batch` | int | `512` | train batch |
| `--n-epochs` | int | `200` | epochs |
| `--seed` | int | `0` | seed |
| `--device` | cuda \| mps \| cpu | `cpu` | SAE device (default cpu, unlike build-lens) |
| `--embed-model-id` | str | `Qwen/Qwen3-Embedding-8B` | label only (recorded in manifest) |

Note: `build-prompt-lens` has no `--embed-model-id` embedding effect — it only
labels the manifest.

---

## Interpret

### `interpret name`
Label each feature from its top-activating pairs. Provide one of `--annotations`
/ `--corpus` (the lens's battle source).

```
prefscope interpret name --lens-dir DIR --out CSV (--annotations JSON... | --corpus PARQUET) [...]
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
| `--verify-frac` | float | `0.2` | held-out fraction (carried into naming/verify split) |
| `--seed` | int | `0` | seed |
| `--concurrency` | int | `1` | features sent to the LLM in parallel (thread pool) |

`name`-only flags:

| flag | type / choices | default | meaning |
|------|----------------|---------|---------|
| `--features` | int* (nargs `*`) | `None` | subset of feature ids (default all) |
| `--name-mode` | str | `auto` | interpreter strategy: `auto` picks individual/pairwise from manifest `input_rep`; or a registered name (`individual`, `pairwise`, `single-text`) |
| `--n-active` | int | `10` | top-active examples per feature |
| `--n-zero` | int | `10` | zero/inactive examples per feature |
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
| `--verify-mode` | str | `auto` | verifier strategy: `auto` picks individual/pairwise from `input_rep`; registered names: `individual`, `pairwise`, `prompt` |
| `--n-per-bucket` | int | `10` | examples per active/zero bucket |
| `--fidelity-threshold` | float | `0.3` | min \|correlation\| to pass (with Bonferroni p<0.05) |
| `--lens-kind` | completion \| prompt | `completion` | `prompt` verifies prompt-lens concepts on `z_prompt` + prompt text (needs `--corpus`) |
| `--negatives` | str | `random` | prompt verify: `random` silent prompts or `close` (needs `--embeddings`) |
| `--embeddings` | str | `None` | prompt verify: `e_prompt.npy` for `close` negatives |

### `name-prompts`
LLM-name prompt-lens features from their top-activating prompts.

```
prefscope name-prompts --lens-dir DIR --corpus PARQUET --out CSV [...]
```

| flag | type / choices | default | meaning |
|------|----------------|---------|---------|
| `--lens-dir` | str | required | prompt lens dir (`z_prompt.npy`) |
| `--corpus` | str | required | corpus parquet (prompt text by `battle_id`) |
| `--out` | str | required | output `prompt_feature_names.csv` |
| `--features` | int* | `None` | subset of feature ids |
| `--n-active` | int | `12` | top-active prompts per feature |
| `--n-zero` | int | `8` | zero/inactive prompts per feature |
| `--backend` | openai \| claude-cli \| codex-cli | `openai` | LLM backend |
| `--model` | str | `deepseek/deepseek-v3.2` | LLM model |
| `--api-base` | str | `https://openrouter.ai/api/v1` | base URL |
| `--api-key-env` | str | `OPENROUTER_API_KEY` | API key env |
| `--concurrency` | int | `1` | parallel features |

### `cluster-features`
Group co-activating SAE features into higher-level behaviors.

```
prefscope cluster-features --lens-dir DIR --out CSV [...]
```

| flag | type / choices | default | meaning |
|------|----------------|---------|---------|
| `--lens-dir` | str | required | lens dir with `z_diff.npy` |
| `--names` | str | `None` | `feature_fidelity`/`feature_names` CSV (concepts + fidelity) |
| `--n-clusters` | int | `10` | target clusters (k-means clusterers) |
| `--method` | str | `spherical-kmeans` | clusterer component: `mi-leiden`, `spherical-kmeans`, `agglomerative`, or any registered |
| `--resolution` | float | `1.0` | mi-leiden resolution (higher → more, smaller communities) |
| `--knn` | int | `0` | mi-leiden: sparsify to top-knn edges (0 = dense) |
| `--min-cluster-size` | int | `1` | mi-leiden: fold smaller communities into one bucket |
| `--fidelity-only` | flag | `False` | cluster only fidelity-passing features |
| `--cluster-on` | difference \| individual | `difference` | co-firing space: `z_diff` or stacked `z_a`/`z_b` |
| `--lens-kind` | completion \| prompt | `completion` | `prompt` clusters `z_prompt.npy` |
| `--name-clusters` | flag | `False` | LLM-name each behavior from member concepts |
| `--backend` | openai \| claude-cli \| codex-cli | `openai` | LLM backend (for `--name-clusters`) |
| `--model` | str | `deepseek/deepseek-v3.2` | LLM model |
| `--api-base` | str | `https://openrouter.ai/api/v1` | base URL |
| `--api-key-env` | str | `OPENROUTER_API_KEY` | API key env |
| `--concurrency` | int | `1` | parallel cluster naming |
| `--out` | str | required | output `feature_clusters.csv` (also writes `<out>_summary.csv`) |

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

---

## Analyze

### `win-relevance`
Which features humans reward (activation vs `human_pref` on the corpus). Corpus
must carry `human_pref` (`build-corpus --keep-labels`).

```
prefscope win-relevance --lens-dir DIR --corpus PARQUET --out CSV [...]
```

| flag | type | default | meaning |
|------|------|---------|---------|
| `--lens-dir` | str | required | lens directory |
| `--corpus` | str | required | corpus parquet WITH `human_pref` |
| `--names` | str | `None` | feature_names/fidelity CSV to attach concepts + filter |
| `--all-features` | flag | `False` | score every feature, not just fidelity-passing |
| `--clusters` | str | `None` | `feature_clusters.csv` → also emit `<out>_clusters.csv` cluster-level win-relevance |
| `--out` | str | required | output win-relevance CSV |

### `elicit`
Prompt-concept → response-concept co-activation lift (preference-independent):
which response concepts appear when a prompt concept is present.

```
prefscope elicit --completion-lens DIR --prompt-lens DIR --out CSV [...]
```

| flag | type | default | meaning |
|------|------|---------|---------|
| `--completion-lens` | str | required | individual lens dir (`z_a.npy` + `z_b.npy`) |
| `--prompt-lens` | str | required | prompt lens dir (`z_prompt.npy`) |
| `--completion-names` | str | `None` | `feature_names.csv` (response concepts) |
| `--completion-fidelity` | str | `None` | `feature_fidelity.csv` → restrict to verified response axes |
| `--prompt-names` | str | `None` | `prompt_feature_names.csv` |
| `--prompt-fidelity` | str | `None` | `prompt_feature_fidelity.csv` → restrict to verified prompt axes |
| `--min-support` | int | `30` | min responses where prompt feature fires to test a cell |
| `--min-cooccur` | int | `5` | min co-occurrences to test a cell |
| `--out` | str | required | output elicitation CSV |

### `conditional-delta`
Prompt-conditioned completion delta Δ_{k,f} (which response properties
distinguish the winner per prompt type) + optional conditional δ_{f,k}.

```
prefscope conditional-delta --completion-lens DIR --prompt-lens DIR --out CSV [...]
```

| flag | type | default | meaning |
|------|------|---------|---------|
| `--completion-lens` | str | required | completion lens dir (`z_diff.npy`) |
| `--prompt-lens` | str | required | prompt lens dir (`z_prompt.npy`) |
| `--corpus` | str | `None` | corpus WITH `human_pref` — orients `z_diff` toward winner (required for `--conditional-out`) |
| `--completion-names` | str | `None` | response concept names |
| `--prompt-names` | str | `None` | prompt concept names |
| `--prompt-clusters` | str | `None` | `prompt_feature_clusters.csv` → condition on prompt CLUSTERS |
| `--conditional-out` | str | `None` | also emit length-controlled conditional win-rate δ_{f,k} |
| `--completion-fidelity` | str | `None` | restrict conditional table to verified axes |
| `--seed` | int | `0` | seed |
| `--permute` | int | `0` | label-permutation null: shuffle prompt-concept labels N times |
| `--jobs` | int | `1` | parallelize permutation null across N processes |
| `--out` | str | required | output Δ_{k,f} CSV |

---

## Diagnose

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
| `--all-features` | flag | `False` | diagnose every feature, not just fidelity-passing |
| `--top` | int | `10` | how many over/under-expressed features to print |

`--embed-model-id` here defaults to `None` (falls back to the lens manifest's
`embed_model_id`; recommended to leave unset). Otherwise the same
[shared embedder flags](#shared-embedder-flags) apply.

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
| `--weight-col` | str | `win_assoc` | win-relevance column to weight features by |
| `--all-features` | flag | `False` | weight by every feature, not just significant |
| `--min-battles` | int | `20` | skip models with fewer oriented battles |
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
| `--matryoshka-prefix` | int+ | `[8]` | Matryoshka prefix lengths |
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
