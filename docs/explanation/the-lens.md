# The lens

The lens is PrefScope's central artifact. This page states what a lens *is* as a
contract first, then describes the specific SAE that ships as the default
instantiation. The distinction matters: the contract is the framework; the default
is one configured choice within it.

## The contract

A **lens** is a frozen encoder `f` that maps an input representation `x` to a **signed
sparse code** `z`:

$$
z = f(x) \in \mathbb{R}^M ,\qquad z \text{ has at most } K \text{ nonzero entries.}
$$

Each coordinate `z_f` is one **concept axis**. Two properties define the contract:

1. **Sparse.** Only a few of the `M` axes fire on any one input — that is what makes
   the code interpretable, one concept at a time.
2. **Signed.** The sign carries the comparison. For a contrast lens over a battle,
   `sign(z_f)` says **which side expresses concept `f` more**:

   $$
   z_f > 0 \;\Leftrightarrow\; A \text{ expresses concept } f \text{ more than } B,
   \qquad z_f < 0 \;\Leftrightarrow\; B \text{ does.}
   $$

This sign convention is load-bearing for everything downstream: diagnosis reads
`sign(z_f)` to decide whether a model over- or under-expresses a concept, and
verification checks that an independent reader agrees with that sign (see
[naming and fidelity](naming-and-fidelity.md)).

Once trained, `f` is **frozen** — a fixed function you reuse to inspect a battle,
diagnose a model, or score preference. The lens directory is that frozen `f` plus its
cached codes, the interpreted concept names (`feature_names.csv`), and a
`manifest.json` recording how it was built. Nothing downstream hardcodes the embed
model or dims; everything reads the manifest.

### The non-linearity caveat

A lens encoder is generally **non-linear** (the default uses a magnitude threshold).
A direct consequence is that for a contrast lens

$$
f(e_A - e_B) \;\neq\; -\,f(e_B - e_A).
$$

So orientation must be applied **before** projection, not recovered by flipping a
sign afterward. This is why the diagnosis bank projects both orientations explicitly
(see [the diagnosis math doc](diagnosis-math.md) §3), and why `LensRep.oriented_codes`
runs two genuine forward passes rather than negating one.

## The default instantiation

The default lens is a **BatchTopK Matryoshka sparse autoencoder over
Qwen3-Embedding-8B (D = 4096)** with the `difference` representation. Each of those
choices is configurable — see [What's configurable](#whats-configurable) below.

### Default embedding and input

By default each (prompt, completion) is wrapped in an instruction format and encoded
by Qwen3-Embedding with last-token pooling and L2 normalization to a unit vector
`e ∈ ℝ^D`. The default `difference` representation feeds the SAE the contrast
`x = e_A − e_B`; the alternative `individual` representation feeds pooled single
embeddings. Both are co-equal — see [representations](representations.md). (The embed
model and `input_rep` are recorded in the manifest, never hardcoded.)

### Default SAE: BatchTopK Matryoshka

The default encoder is a single-hidden-layer SAE with `M` features of which `K` fire
(the default, not the framework — see [the SAE](sae.md) for how pinned this is):

- **Encoder pre-activations:** `a = W_enc (x − b_in) + b_neuron`.
- **BatchTopK sparsity (training):** sparsity is allocated *across the whole batch* —
  for a batch of `B`, keep the `K·B` pre-activations largest in **absolute** value
  (sign preserved), zero the rest. Easy examples use fewer features, hard ones more,
  averaging to `K`.
- **Decoder / reconstruction:** `x̂ = W_dec z + b_in` with each decoder column
  (dictionary atom) kept unit-norm; the radial gradient component is projected out so
  the optimizer doesn't fight the constraint.
- **Normalized-MSE loss:** error relative to a predict-the-mean baseline, so 1 means
  "no better than the mean" and 0 means perfect.
- **Matryoshka nesting:** reconstruct from nested prefixes of the code and average the
  losses, forcing low-index atoms to carry the most — a nested dictionary.
- **Dead-neuron auxiliary loss:** features that stop firing are revived by asking them
  to reconstruct the current residual.
- **Inference threshold (the frozen lens):** at inference there is no batch, so a
  learned magnitude threshold `θ` decides which axes fire: `z_f = a_f` if `|a_f| > θ`,
  else 0. This thresholded `z = f(x)` is exactly the contract above — and it is the
  source of the non-linearity caveat.

Defaults for `M`/`K` and the prefix lengths live in the code; you set them per-build
with `--m-total` / `--k` (see the [CLI reference](../reference/cli.md)).

## What's configurable

- **Embedding model and dims** — configured at embed time, recorded in the manifest.
- **Input representation (`input_rep`)** — `difference` / `individual` / `prompt`; see
  [representations](representations.md).
- **SAE architecture (`--sae-type`)** — `batchtopk` / `jumprelu` / `simple-topk`, or a
  custom one you register; see [the SAE](sae.md).
- **`M`, `K`, Matryoshka prefixes** — build flags (`--m-total`, `--k`,
  `--matryoshka-prefix`).

To add a genuinely different encoder today you would edit `prefscope/sae/`, not
register a component. For swapping the configurable pieces, see `docs/extending/`.
