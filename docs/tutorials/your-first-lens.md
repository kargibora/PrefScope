# Your first lens

This tutorial builds a small lens and checks its concept names.

First complete [Getting started](getting-started.md). This tutorial is not instant on a
laptop CPU because `build-lens` embeds every response. It uses the smaller
`Qwen/Qwen3-Embedding-0.6B` model instead of the 8B default. The naming and verification
steps also need an LLM through `OPENROUTER_API_KEY` or a local OpenAI-compatible server.

## The main idea

A **lens** is a saved encoder that turns an embedding into a short list of active
features. Prompt and individual-response lenses use non-negative feature values by
default. When PrefScope subtracts two response codes, the sign says which response has
more of a feature. A direct difference lens instead learns from A-minus-B embeddings.
You build a lens once, then reuse it
to inspect a battle, diagnose a model, or score which concepts drive preference. The
three durable artifacts, in order, are **corpus → lens → concept tables**. See
[the architecture explanation](../explanation/architecture.md) for the full picture.

## 1. The sample corpus

A corpus is a table of prompts and responses. PrefScope needs a prompt and response A.
Response B is optional and makes the row paired. Source, language, model names, and a
stable row ID are useful but can be filled when missing. An optional `human_pref` column
stores the preference for A as `0`, `0.5`, or `1`.

Generate a small ready-to-use corpus and matching configuration (60 synthetic battles,
with `human_pref` and all required columns), then inspect it:

```bash
prefscope init-demo --out demo
prefscope inspect --corpus demo/sample_corpus.parquet
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
    --corpus demo/sample_corpus.parquet \
    --out demo/lens \
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
- `--sae-type auto` (implicit) — selects non-negative `batchtopk-relu` for this
  individual lens. A direct difference lens resolves to signed `batchtopk`.
- `--device cpu` — fine for a tiny table; large corpora want a GPU.
- `--embed-model-id Qwen/Qwen3-Embedding-0.6B` — a smaller member of the same
  last-token-pooled embedding family, suitable for learning the workflow. Use the
  configured 8B default for serious experiments when resources permit.

The result is a **lens directory** under `demo/lens/`: the frozen encoder
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

`init-demo` already wrote `demo/quickstart.yaml`; its essential content is:

```yaml
lens_dir: /absolute/path/to/demo/lens
corpus: /absolute/path/to/demo/sample_corpus.parquet
out_dir: /absolute/path/to/demo/results
stages: [name, verify, cluster, win-relevance]
llm: {backend: openai, model: deepseek/deepseek-v3.2}
interpreter: {name: auto, n_active: 3}
verifier:    {name: auto, n_per_bucket: 3}
```

Then run it. Any OpenAI-compatible endpoint works — a hosted model via
`OPENROUTER_API_KEY`, or a **local** server (point `--api-base` / the config's
`api_base` at e.g. a vLLM endpoint) so the whole thing runs offline and free:

```bash
OPENROUTER_API_KEY=... prefscope run --config demo/quickstart.yaml
```

`n_active` / `n_per_bucket` are tiny here only because the corpus is small; on a real
corpus raise them (more examples per feature = more reliable names and more
statistical power in verification).

## 4. Read the four concept tables

`demo/results/` now holds the four tables that are PrefScope's payoff:

| File | What it answers |
|------|-----------------|
| `feature_names.csv` | what each concept axis *is* (a short phrase per feature) |
| `feature_fidelity.csv` | which names *survive held-out verification* — the verifier judges whether the named concept tracks the axis (`correlation`, `fidelity_pass`) |
| `feature_clusters.csv` | how axes *group* into co-activating behaviors |
| `win_relevance.csv` | which concepts *correlate with being preferred* |

Naming is hypothesis generation; verification is the check. Naming and verification
use the same LLM by default but operate on disjoint data with different prompts; set
`verify_llm` in the config for cross-model review. The split (name on 80% of battles,
verify on a disjoint 20%), the falsification gate, and the statistical-power caveat are
explained in
[naming and fidelity](../explanation/naming-and-fidelity.md). The
preference/diagnosis statistics behind `win_relevance.csv` are in
[the diagnosis math doc](../explanation/diagnosis-math.md).

## What you have now

- A real lens directory you can reuse — `Lens.from_dir("demo/lens")` loads it
  for inference, or `prefscope diagnose` / the Streamlit viewer point at it.
- The four concept tables describing your data *by concept*.

To go deeper: swap the input representation
([representations](../explanation/representations.md)), inspect arbitrary texts with
`prefscope concepts`, or diagnose a specific model ([the diagnosis guide](../how-to/diagnose-a-model.md)).
