# The SAE

The lens encoder is a sparse autoencoder (SAE): it maps an embedding to a sparse
code `z`. The rest of the pipeline only depends on that — a frozen encoder producing
sparse codes ([the lens](the-lens.md)). PrefScope ships three SAE architectures and
lets you register your own.

## Choosing an architecture

`build-lens --sae-type <name>` selects the SAE; the choice is recorded in the lens
`manifest.json` and read back by naming, verification, and diagnosis.

| `--sae-type` | Description |
|--------------|-------------|
| `batchtopk` (default) | BatchTopK Matryoshka SAE. Keeps the top `K × batch_size` activations during training and learns a per-feature inference threshold. Signed codes; works with any `--input-rep`. |
| `jumprelu` | JumpReLU SAE ([Rajamanoharan et al. 2024](https://arxiv.org/abs/2407.14435)). A learned per-feature threshold `θ_i` gates each feature (`z_i = π_i` when `π_i > θ_i`, else 0), trained with an L0 penalty (`--sparsity-coef λ`) and a straight-through estimator (`--bandwidth ε`). Codes are one-sided non-negative — use `--input-rep individual`. |
| `simple-topk` | Plain top-`K` SAE, a training-time ablation. As a frozen lens it selects the top-`K` features per example at inference (`_threshold_select` → per-example top-`K`), so it activates exactly `K` — deployable, though `batch-topk` remains the default. |

```bash
prefscope build-lens --corpus corpus.parquet --input-rep individual \
    --sae-type jumprelu --sparsity-coef 1e-3 --bandwidth 1e-3 --out lenses/jr
```

## Adding your own SAE

The SAE is a registry component (kind `sae`). Subclass `BatchTopKSAE`, register it,
and select it with `--sae-type <your-name>`. See
[add an SAE](../extending/add-an-sae.md).

## The default (BatchTopK) in detail

`batchtopk` trains the SAE in `prefscope/sae/model.py`:

- **BatchTopK sparsity** — sparsity is allocated across the whole batch: for a batch
  of `B`, keep the `K × B` pre-activations largest in absolute value and zero the
  rest, then learn a per-feature magnitude threshold. At inference a feature fires
  iff its pre-activation clears that threshold (a clean frozen encoder).
- **Matryoshka nesting** — the dictionary is trained so prefixes of it
  (`--matryoshka-prefix`) are valid smaller dictionaries, which curbs feature
  splitting and absorption.
- **Dead-feature handling** — an auxiliary loss revives features that stop firing.

Embeddings default to Qwen3-Embedding-8B (`D = 4096`) and the input representation
to `difference`; both are recorded in `manifest.json` — see
[representations](representations.md).
