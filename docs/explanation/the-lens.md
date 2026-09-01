# The lens

The lens is PrefScope's central artifact. This page states what a lens *is* as a
contract first, then describes the specific SAE that ships as the default
instantiation. The distinction matters: the contract is the framework; the default
is one configured choice within it.

## The contract

A **lens** is a frozen encoder `f` that maps an input representation `x` to a
**sparse code** `z`:

$$
z = f(x) \in \mathbb{R}^M ,\qquad \mathbb{E}[\|z\|_0] \approx K.
$$

Each coordinate `z_f` is one feature. Two properties define the contract:

1. **Sparse.** Only a few of the `M` axes fire on a typical input — BatchTopK controls
   the average budget rather than imposing a strict per-row cap.
2. **Explicit semantics.** The manifest records whether codes are non-negative
   presences or signed axes. For an individual/prompt presence lens, `z_f ≥ 0` and
   zero means silent. For any paired contrast code, `sign(z_f)` says **which side
   expresses concept `f` more**:

   $$
   z_f > 0 \;\Leftrightarrow\; A \text{ expresses concept } f \text{ more than } B,
   \qquad z_f < 0 \;\Leftrightarrow\; B \text{ does.}
   $$

The manifest fields `activation_polarity` and `code_semantics` prevent those two
quantities from being conflated. The pair sign convention is load-bearing downstream:
diagnosis reads
`sign(z_f)` to decide whether a model over- or under-expresses a concept, and
verification checks that an independent reader agrees with that sign (see
[naming and fidelity](naming-and-fidelity.md)).

Once trained, `f` is fixed. You can reuse it to inspect new data. Every lens directory
contains the encoder and `manifest.json`. A build directory can also contain cached
training codes. Concept names are optional annotations added during interpretation or
packaging; a fresh lens build does not contain them. Downstream code reads the embedding
model and dimensions from the manifest.

### The non-linearity caveat

A lens encoder is generally **non-linear** (the default uses a magnitude threshold).
A direct consequence is that for a contrast lens

$$
f(e_A - e_B) \;\neq\; -\,f(e_B - e_A).
$$

For a direct difference lens, orientation must be applied **before** projection. The
diagnosis bank therefore projects both A-minus-B and B-minus-A instead of negating one
result. For an individual lens, PrefScope projects A and B once and forms both exact code
differences from `z_a` and `z_b`. See [Diagnosis math](diagnosis-math.md#3-the-oriented-code-bank-pool-baseline).

## The default instantiation

The CLI defaults to a **signed BatchTopK sparse autoencoder over
Qwen3-Embedding-8B (D = 4096)** because its default representation is `difference`.
With `input_rep=individual` or a prompt lens, `sae_type=auto` instead selects
non-negative BatchTopK. Matryoshka is opt-in. Each choice is configurable — see
[What's configurable](#whats-configurable) below.

### Default embedding and input

By default each (prompt, completion) is wrapped in an instruction format and encoded
by Qwen3-Embedding with last-token pooling and L2 normalization to a unit vector
`e ∈ ℝ^D`. The default `difference` representation feeds the SAE the contrast
`x = e_A − e_B`; the alternative `individual` representation feeds pooled single
embeddings. Both are co-equal — see [representations](representations.md). (The embed
model and `input_rep` are recorded in the manifest, never hardcoded.)

### Default SAE resolution: signed or non-negative BatchTopK

The default encoder is a single-hidden-layer SAE with `M` features of which `K` fire
(the default, not the framework — see [the SAE](sae.md) for how pinned this is):

- **Encoder pre-activations:** `a = W_enc (x − b_in) + b_neuron`.
- **BatchTopK sparsity (training):** sparsity is allocated *across the whole batch*.
  Signed `batchtopk` keeps the `K·B` largest magnitudes and preserves signs;
  `batchtopk-relu` applies ReLU first and keeps the `K·B` largest positive values.
- **Decoder / reconstruction:** `x̂ = W_dec z + b_in` with each decoder column
  (dictionary atom) kept unit-norm; the radial gradient component is projected out so
  the optimizer doesn't fight the constraint.
- **Normalized-MSE loss:** error relative to a predict-the-mean baseline, so 1 means
  "no better than the mean" and 0 means perfect.
- **Optional Matryoshka nesting:** when `--matryoshka-prefix` is supplied, reconstruct
  from nested prefixes and average the losses. It is disabled by default.
- **Dead-neuron auxiliary loss:** features that stop firing are revived by asking them
  to reconstruct the current residual.
- **Inference threshold (the frozen lens):** after checkpoint selection, a bounded
  calibration pass chooses the scalar threshold that gives mean L0 near `K`.
  Signed codes gate `|a_f|`; non-negative codes gate `ReLU(a_f)`. Deployment NMSE,
  explained variance, L0, and dead/rare counts are recorded separately.

Defaults for `M`/`K` and the prefix lengths live in the code; you set them per-build
with `--m-total` / `--k` (see the [CLI reference](../reference/cli.md)).

## What's configurable

- **Embedding model and dims** — configured at embed time, recorded in the manifest.
- **Input representation (`input_rep`)** — `difference` / `individual` / `prompt`; see
  [representations](representations.md).
- **SAE architecture (`--sae-type`)** — `auto`, signed `batchtopk`, non-negative
  `batchtopk-relu`, `jumprelu`, `simple-topk`, or a custom registration; see
  [the SAE](sae.md).
- **`M`, `K`, Matryoshka prefixes** — build flags (`--m-total`, `--k`,
  `--matryoshka-prefix`).

To add a different encoder, register an `sae` component and declare its polarity and
code semantics. See [add an SAE](../extending/add-an-sae.md).
