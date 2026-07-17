# Your first lens

This tutorial takes you end to end on the smallest possible scale: a tiny battle
table → a trained lens → named, verified concept axes. Once it runs, you will
understand what each artifact is and where to read the math.

Prerequisite: a working install ([Getting started](getting-started.md)).

**Prerequisites (be aware before you start):** unlike `inspect`, this flow is *not*
instant on a laptop CPU. `build-lens` embeds every completion; this tutorial overrides
the research default (`Qwen/Qwen3-Embedding-8B`) with the smaller
`Qwen/Qwen3-Embedding-0.6B`. The naming/verify stages need an LLM: either an
`OPENROUTER_API_KEY` or a local OpenAI-compatible endpoint (e.g. vLLM). Only `inspect`
from the previous tutorial is truly no-model.

## The mental model in one paragraph

A **lens** is a frozen encoder `f` that maps a response representation to a *signed
sparse code* `z`: a short list of concept activations where the **sign** says which
side of the battle expresses each concept more. You build a lens once, then reuse it
to inspect a battle, diagnose a model, or score which concepts drive preference. The
three durable artifacts, in order, are **corpus → lens → concept tables**. See
[the architecture explanation](../explanation/architecture.md) for the full picture.

## 1. The sample corpus

A corpus is just a table of battles. The minimum columns are a `battle_id`, a
`source`, a `language`, a prompt, and two completions; a `human_pref` column
(preference for A, in `{0, 0.5, 1}`) is optional but lets the later analysis ground
concepts in real preference.

The repo ships a small ready-to-use corpus at `examples/sample_corpus.parquet` (60
battles, with `human_pref` and all required columns), so you can run this tutorial
without building one. Inspect it first:

```bash
prefscope inspect --corpus examples/sample_corpus.parquet
```

In practice you would use far more battles (the SAE needs enough rows to learn
directions), but the shape is exactly this. To pull a large real corpus instead, see
`build-corpus` with `uv sync --extra arena`.

## 2. Build the lens

`build-lens` embeds every completion and trains a sparse autoencoder over the result.
Keep it small so it finishes quickly on CPU — a tiny dictionary (`--m-total`) and a
low active count (`--k`):

```bash
prefscope build-lens \
    --corpus examples/sample_corpus.parquet \
    --out lenses/sample \
    --m-total 16 --k 4 \
    --input-rep individual \
    --embed-model-id Qwen/Qwen3-Embedding-0.6B \
    --device cpu
```

What the flags mean:

- `--m-total 16` — the dictionary has 16 concept axes (features).
- `--k 4` — roughly 4 of them fire on any one example (sparsity).
- `--input-rep individual` — the encoder is trained on each completion's embedding
  so it can also score a lone response later. The default is `difference` (trained on
  the contrast `e_a − e_b`). These are **co-equal choices**, not method-vs-ablation;
  [the representations explanation](../explanation/representations.md) covers when to
  pick which.
- `--device cpu` — fine for a tiny table; large corpora want a GPU.
- `--embed-model-id Qwen/Qwen3-Embedding-0.6B` — a smaller member of the same
  last-token-pooled embedding family, suitable for learning the workflow. Use the
  configured 8B default for serious experiments when resources permit.

The result is a **lens directory** under `lenses/sample/`: the frozen encoder
(`sae_model.pt`), the cached codes (`z_a.npy`, `z_b.npy`, `z_diff.npy`), the
row-aligned `battles.parquet`, a training log, and a `manifest.json` recording the
embed model, dims, and `input_rep`. That `manifest.json` is the source of truth
downstream — nothing hardcodes the model id or dims.

### What just happened

`build-lens` did two things in sequence — embed, then train — and cached both. The
embedding step wraps each (prompt, completion) in an instruction format and encodes
it to a normalized vector; the SAE step learns a small dictionary of directions over
those vectors, keeping only a few active per example. The encoder, frozen, is your
lens `f`. The lens-as-contract view and the SAE details are in
[the lens explanation](../explanation/the-lens.md) and [the SAE](../explanation/sae.md).

## 3. Name and verify the axes

A raw axis is just a direction; to read it you name it, then check the name holds up.
The whole chain — name → verify → cluster → win-relevance — runs from one config.

Write `sample.yaml`:

```yaml
lens_dir: lenses/sample
corpus: examples/sample_corpus.parquet
out_dir: results/sample
stages: [name, verify, cluster, win-relevance]
llm: {backend: openai, model: deepseek/deepseek-v3.2}
interpreter: {name: auto, n_active: 3}
verifier:    {name: auto, n_per_bucket: 3}
```

Then run it. Any OpenAI-compatible endpoint works — a hosted model via
`OPENROUTER_API_KEY`, or a **local** server (point `--api-base` / the config's
`api_base` at e.g. a vLLM endpoint) so the whole thing runs offline and free:

```bash
OPENROUTER_API_KEY=... prefscope run --config sample.yaml
```

`n_active` / `n_per_bucket` are tiny here only because the corpus is small; on a real
corpus raise them (more examples per feature = more reliable names and more
statistical power in verification).

## 4. Read the four concept tables

`results/sample/` now holds the four tables that are PrefScope's payoff:

| File | What it answers |
|------|-----------------|
| `feature_names.csv` | what each concept axis *is* (a short phrase per feature) |
| `feature_fidelity.csv` | which names *survive verification* — an independent LLM agrees the named concept tracks the axis (`correlation`, `fidelity_pass`) |
| `feature_clusters.csv` | how axes *group* into co-activating behaviors |
| `win_relevance.csv` | which concepts *correlate with being preferred* |

Naming is hypothesis generation; verification is the check. The split (name on 80% of
battles, verify on a disjoint 20%), the falsification gate, and the
statistical-power caveat are explained in
[naming and fidelity](../explanation/naming-and-fidelity.md). The
preference/diagnosis statistics behind `win_relevance.csv` are in
[the diagnosis math doc](../explanation/diagnosis-math.md).

## What you have now

- A real lens directory you can reuse — `LoadedLens.from_dir("lenses/sample")` loads it
  for inference, or `prefscope diagnose` / the Streamlit viewer point at it.
- The four concept tables describing your data *by concept*.

To go deeper: swap the input representation
([representations](../explanation/representations.md)), spot-check a single battle
(`scripts/analyze_battle.py`), or diagnose a specific model (the diagnose how-to).
