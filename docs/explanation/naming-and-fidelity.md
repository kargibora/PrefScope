# Naming and fidelity

A raw lens axis is a direction, not an explanation. To read it you **name** it, then
**verify** the name holds up. PrefScope keeps these as two distinct stages with a
deliberate gap between them: naming is hypothesis generation; verification is
falsification on held-out data. This page explains the split, the gate, and the
caveat — de-pinned from any specific LLM (any OpenAI-compatible backend works, hosted
or local).

## Scope: what a passing name is (and is not)

A verified name is a **verified hypothesis**, not ground truth. Concretely, a `fidelity_pass`
means: on a held-out, deliberately **case-control** sample (top activators vs silent
controls), an independent LLM's presence judgments correlate positively and significantly
with the axis firing. With the default sampler, that is *correlational detection fidelity
at the extremes*; the stratified-random mode covers the wider activation range. It does
**not** establish that:

- the name is the axis's complete or ground-truth meaning (naming reads only the positive
  pole; the negative pole of a signed axis is a separate, un-named concept);
- the feature is fully monosemantic, or detects *every* occurrence of the behaviour
  (the default top-only sample biases toward specificity over sensitivity; feature
  absorption can also hide true instances);
- the reported precision/agreement reflects **corpus prevalence** — it is measured on the
  balanced selected sample, not the population;
- the feature *causes* anything in the generating model. The SAE runs on **output
  embeddings**, so this is post-training data/representation analysis, not mechanistic
  interpretation of the model.

Treat names as auditable leads, and read the examples yourself before making a claim.
For stronger reporting, the config can generate several independently sampled naming
proposals and synthesize them (`n_candidates`), then verify over a larger
`stratified-random` held-out sample (`n_examples`). This reduces dependence on one
extreme evidence view, but it does not turn an observational output-space feature into
a causal or mechanistic claim.

## The split: name on one pool, verify on a disjoint one

Battles are split once — deterministically, by hashing the instruction id — into a
**name pool** (≈80%) and a disjoint **verify pool** (≈20%). Naming never touches the
verify pool, so verification is a genuine held-out check, not a re-test of the same
examples.

## Naming (hypothesis generation)

For each feature `f`, within the name pool, take the battles where the axis fires
hardest (its positive pole) plus some random *silent* battles (`z_f = 0`). Show an LLM
each example's prompt, response A, response B, and the signed activation `z_f`, and
ask for a short concept phrase `c_f` describing what distinguishes the
high-activation responses:

$$
c_f = \mathrm{LLM}_{\text{name}}\big(\{(\text{prompt}_i, A_i, B_i, z_{i,f})\}\big).
$$

These names are **unverified hypotheses**. With `n_candidates > 1`, PrefScope samples
several views from the strong-activation pool, generates independent proposals, and asks
for one atomic synthesis; `candidate_concepts` retains every proposal for audit. A name
that sounds plausible may still not track the axis — that is what the next stage checks.

## Verification (the falsification gate)

A name is trustworthy only if an **independent** LLM, shown a response pair and *just
the concept name* (not the activation), agrees with the SAE about which side expresses
it. This runs on the held-out verify pool.

For feature `f`, sample three buckets by activation — **pos** (top `n` among
`z_f > 0`), **neg** (top `n` among `z_f < 0`), and **tie** (`n` random among
`z_f = 0`). For each sampled battle `i`:

- the **SAE label** is `s_i = sign(z_{i,f}) ∈ {−1, 0, +1}`;
- the **LLM label** `ℓ_i ∈ {+1 (A), −1 (B), 0 (tie)}` answers, given only
  `(prompt, A, B, c_f)`, which response exhibits the concept more.

The **opposite-pole** design is what makes the pairwise test a real falsification test:
the `neg` bucket presents pairs where the *opposite* side should express the concept, so
a name that merely sounds right but does not track the axis gets caught. For individual
responses, `negatives: close` instead chooses silent controls that resemble activators in
the other SAE features.

The default `sampling: extremes` tests detection at the poles. For broader held-out
coverage, `sampling: stratified-random` samples uniformly inside the positive, negative,
and silent buckets; `n_examples: 300` divides a total 300-judgment budget across them.
For an individual or prompt feature, the same setting splits the budget between positive
activations and silent controls.

### The fidelity correlation and the gate

Fidelity is the Pearson correlation between the two label vectors:

$$
r_f = \mathrm{corr}(s, \ell),\qquad p_f = \text{its two-sided } p\text{-value.}
$$

A high positive `r_f` means the SAE axis and the human-readable concept track each other:
the concept is more present as the axis fires more. The gate combines a **positive**
effect-size threshold with significance, and **Bonferroni**-corrects for testing all `M`
features at once:

$$
p_f^{\text{bonf}} = \min(1,\ p_f \cdot M),\qquad
\text{fidelity\_pass}_f = (r_f \ge \tau)\ \wedge\ (p_f^{\text{bonf}} < 0.05),
$$

with `τ = 0.3` by default. The gate requires a **positive** correlation, so a name that
describes the *opposite* (low-activation) pole — `sign(r_f) = −1` — **fails**: a passing
name always describes the positive pole, which keeps downstream "more of concept X"
correct. The `sign` column is still reported for diagnosis, but for a passing feature it
is always `+1`. Agreement, precision/recall/F1 are reported but are diagnostic, not gating.

## The statistical-power caveat

With `n` per bucket you get only ≈ `3n` test cases, and the Bonferroni factor `M` is
harsh. A *genuine* feature with `|r_f| ≈ 0.4` and raw `p ≈ 0.01` can fail after
correction at small `n`. So a failed gate at small `n` is **not** proof the feature is
bad — it may just be underpowered. If real features keep failing, raise
`--n-per-bucket` (more cases → more power) before concluding the axis is meaningless.

This is the practical reason naming and verification are decoupled: you can re-run
verification with more examples without re-naming, and tune the gate to your corpus
size rather than trusting a single threshold blindly.

## Where this sits

Verification produces `feature_fidelity.csv`. Downstream stages — clustering, and the
diagnosis in [the diagnosis math doc](diagnosis-math.md) — by default restrict to
fidelity-passing axes, so the falsification gate is what keeps the final concept story
honest. For the flags that tune naming and verification, see the
[CLI reference](../reference/cli.md).

To swap the naming or verification strategy (they are registered `Interpreter` /
`Verifier` components), see `docs/extending/add-a-verifier.md`. For the exact flags
(`--n-active`, `--n-per-bucket`, `--abbreviate`, `--model` / `--api-base`), see the
build-and-analyze how-to.
