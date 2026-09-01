# The SAE

The lens encoder is a sparse autoencoder (SAE): it maps an embedding to a sparse
code `z`. The rest of the pipeline only depends on that — a frozen encoder producing
sparse codes ([the lens](the-lens.md)). PrefScope ships four SAE architectures and
lets you register your own.

## Choosing an architecture

`build-lens --sae-type <name>` selects the SAE; the choice is recorded in the lens
`manifest.json` and read back by naming, verification, and diagnosis.

| `--sae-type` | Description |
|--------------|-------------|
| `auto` (default) | Resolves to signed `batchtopk` for direct differences and non-negative `batchtopk-relu` for individual responses or prompts. |
| `batchtopk` / `signed-batchtopk` | Signed BatchTopK. Keeps the `K × batch_size` largest magnitudes and preserves their sign. `batchtopk` retains the meaning of every legacy checkpoint; use it for direct difference lenses. |
| `batchtopk-relu` | Non-negative BatchTopK. Applies ReLU before allocating the batch sparsity budget, so zero means absent and positive values represent feature presence. Default for individual and prompt lenses. |
| `jumprelu` | JumpReLU SAE ([Rajamanoharan et al. 2024](https://arxiv.org/abs/2407.14435)). ReLU pre-activations pass learned per-feature thresholds, trained with an L0 penalty (`--sparsity-coef λ`) and straight-through estimator (`--bandwidth ε`). `--sparsity-warmup-steps` can warm λ. Codes are non-negative. |
| `simple-topk` | Plain top-`K` SAE, a training-time ablation. As a frozen lens it selects the top-`K` features per example at inference (`_threshold_select` → per-example top-`K`), so it activates exactly `K` — deployable, though `batch-topk` remains the default. |

```bash
prefscope build-lens --corpus corpus.parquet --input-rep individual \
    --sae-type jumprelu --sparsity-coef 1e-3 --bandwidth 1e-3 --out lenses/jr
```

## Adding your own SAE

The SAE is a registry component (kind `sae`). Subclass `BatchTopKSAE`, register it,
and select it with `--sae-type <your-name>`. See
[add an SAE](../extending/add-an-sae.md).

## BatchTopK in detail

Both BatchTopK variants train the SAE in `prefscope/sae/model.py`:

- **BatchTopK sparsity** — sparsity is allocated across the whole batch: for a batch
  of `B`, keep the `K × B` pre-activations largest in absolute value and zero the
  rest. Signed BatchTopK ranks magnitudes and preserves signs; non-negative BatchTopK
  applies ReLU and ranks positive values. A bounded calibration pass fits the frozen
  inference threshold to an average L0 near `K` and records the achieved L0.
- **Optional Matryoshka nesting** — passing `--matryoshka-prefix` trains prefixes as
  valid smaller dictionaries. It is off by default so nested-width training can be
  evaluated independently rather than silently changing every lens.
- **Dead-feature handling** — an auxiliary loss revives features that stop firing.

The trainer uses Adam with zero weight decay, reports explained variance as
`1 - normalized_MSE`, and records both checkpoint-selection metrics and metrics from
the deployed thresholded encoder. JumpReLU does not support Matryoshka; combining the
two fails explicitly instead of ignoring the prefixes.

Embeddings default to Qwen3-Embedding-8B (`D = 4096`) and the input representation
to `difference`; both are recorded in `manifest.json` — see
[representations](representations.md).
