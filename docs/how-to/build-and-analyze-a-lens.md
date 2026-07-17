# Build a lens and analyze a dataset by concept

Goal: take a set of pairwise battles, train a lens over it, and produce the four
concept tables (names, fidelity, clusters, win-relevance) that tell you *what
concepts the data contains and which ones humans/judges reward*.

## 0. Prerequisites

```bash
uv sync --extra arena          # HuggingFace arena loaders (only needed for build-corpus)
uv sync --extra cluster        # igraph + leidenalg for the default mi-leiden clusterer
# GPU torch for embedding + SAE training, e.g.:
uv sync --extra cu121          # NVIDIA   (or --extra rocm / --extra cpu)
export OPENROUTER_API_KEY=...  # for the naming/verification LLM calls
```

## 1. Build (or bring) a corpus

A corpus is a normalized parquet of battles. `build-corpus` pulls and merges
public arenas; dedup is on content hash, so overlapping dumps collapse to one row.

```bash
prefscope build-corpus \
    --source lmarena-100k lmarena-140k \
    --out corpus.parquet \
    --keep-labels                       # keep human_pref — needed for win-relevance
```

Schema: `battle_id · source · language · prompt · model_a · model_b ·
completion_a · completion_b` (+ `human_pref` with `--keep-labels`). Already have a
battle table? Skip this step — any parquet with those columns works.

Sanity-check it first (fast, no model load):

```bash
prefscope inspect --corpus corpus.parquet
```

## 2. Build the lens (this is the embedding step)

`build-lens` embeds every completion (vectors are cached and reusable) and trains
a BatchTopK SAE. The lens is **unsupervised** — no labels needed to train it.

```bash
prefscope build-lens \
    --corpus corpus.parquet \
    --input-rep individual \            # encoder applies to any single response
    --out lenses/mylens \
    --m-total 128 --k 16 \              # 128 features, 16 active per row
    --dump-embeddings emb/ \            # cache vectors so re-fits skip re-embedding
    --device cuda
```

`--input-rep`:
- **`individual`** — trains on pooled `[e_a; e_b]`; writes `z_a/z_b/z_diff`. The
  encoder works on a lone response, which is what diagnosis/inference need. Prefer this.
- **`difference`** — trains on `e_a − e_b`; writes only `z_diff`. A pure contrast
  lens (can't score a single response).

To retrain a different size without re-embedding:

```bash
prefscope build-lens --from-embeddings emb/ \
    --input-rep individual --out lenses/mylens_m256 --m-total 256 --k 16
```

The lens dir now holds `sae_model.pt`, `z_*.npy`, `battles.parquet`, and
`manifest.json` (records `input_rep`, `embed_model_id`, dims — read downstream,
never hardcoded).

## 3. Analyze by concept — one config

Write a `pipeline.yaml` (copy [`examples/pipeline.yaml`](../../examples/pipeline.yaml)):

```yaml
lens_dir: lenses/mylens
corpus: corpus.parquet                  # needed by win-relevance (human_pref)
out_dir: results/mylens
stages: [name, verify, cluster, win-relevance]
llm: {backend: openai, model: deepseek/deepseek-v3.2}
interpreter: {name: auto, n_active: 12}
verifier:    {name: auto, n_per_bucket: 12}
clusterer:   {name: mi-leiden, resolution: 1.2, knn: 6}
win_relevance: {all_features: false}    # restrict to fidelity-passing axes
```

```bash
prefscope run --config pipeline.yaml
```

This runs `name → verify → cluster → win-relevance`, resolving each component
through the registry and threading outputs. Equivalent to running the four
subcommands by hand (`interpret name`, `interpret verify`, `cluster-features`,
`win-relevance`) — the config is just the declarative front-end.

### What you get (under `out_dir`)

| File | Column to read | Meaning |
|------|----------------|---------|
| `feature_names.csv` | `concept` | the LLM's label for each SAE axis |
| `feature_fidelity.csv` | `fidelity_pass` | did the label survive a held-out falsification test |
| `feature_clusters.csv` | `cluster_id`, `behavior` | co-firing concepts grouped into behaviors |
| `win_relevance.csv` | `win_assoc`, `significant` | which concepts go with humans/judges preferring a response |

`cluster` also writes `feature_clusters_summary.csv`; with clusters present,
`win-relevance` adds `win_relevance_clusters.csv` (per-behavior Δwin-rate).

### Swapping a component

One-line edits, validated up front (a typo prints `config error: … available: …`):

```yaml
verifier:  {name: pairwise}                          # different verification strategy
clusterer: {name: spherical-kmeans, n_clusters: 20}  # different clustering algorithm
llm:       {backend: openai, model: Qwen/Qwen2.5-72B-Instruct,
            api_base: http://localhost:8000/v1}      # run the LLM step offline
```

Run a subset by listing fewer `stages` (e.g. `[name, verify]`).

## 4. Browse it

```bash
uv sync --extra viewer
uv run --extra viewer streamlit run prefscope/viewer/app.py -- \
    --lens-dir lenses/mylens
```

`feature_names.csv` / `feature_fidelity.csv` in the lens dir are picked up
automatically. The lens dir is small — sync it to a laptop and run the viewer locally.

## Prompt lenses

To analyze the **questions** instead of the answers, build a prompt lens and set
`lens_kind: prompt` in the config:

```bash
prefscope embed-prompts --corpus corpus.parquet --out pemb/
prefscope build-prompt-lens --from-embeddings pemb/ --out lenses/promptlens
```

```yaml
lens_dir: lenses/promptlens
corpus: corpus.parquet
out_dir: results/promptlens
lens_kind: prompt
stages: [name, verify, cluster]         # win-relevance is completion-only
```

Outputs use the `prompt_feature_*` filenames. To relate the two — *which prompt
concepts elicit which response concepts* — see the `elicit` and `conditional-delta`
subcommands (`prefscope elicit --help`).
