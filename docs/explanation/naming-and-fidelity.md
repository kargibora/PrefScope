# Naming and fidelity

A raw lens feature is a numerical direction, not an explanation. Naming proposes a
label for it. Verification checks that label on separate examples. PrefScope keeps these
as two different steps. This page explains both steps. The process works with any
supported LLM endpoint, hosted or local.

## Scope: what a passing name is (and is not)

A passing name is still a checked proposal, not ground truth. `fidelity_pass=True`
means that the name worked on separate high-activation examples and did not also fit the
silent controls. The default check focuses on the strongest activations. The optional
`quantile-stratified` mode checks a wider activation range. A passing result does **not** establish that:

- the name is the complete meaning of the feature;
- the feature describes only one idea or detects every occurrence of that idea;
- precision on the selected examples equals prevalence or precision in the full corpus;
- the feature caused anything inside the model that produced the text.

A signed feature can also have a different meaning on its negative pole. The stored name
usually describes only the positive pole.

Treat names as auditable leads, and read the examples yourself before making a claim.
For stronger reporting, the config can generate several independently sampled naming
proposals and synthesize them (`n_candidates`), then verify over a larger
`quantile-stratified` held-out sample (`n_examples`). This reduces dependence on one
extreme evidence view, but it does not turn an observational output-space feature into
a causal or mechanistic claim.

New individual and prompt lenses default to non-negative `batchtopk-relu`, so their
positive-versus-zero evidence has presence semantics. Historical individual/prompt
`batchtopk` lenses are signed axes. PrefScope refuses to run their single-text namer or
verifier unless you pass `--pole positive` (pipeline config: `pole: positive`). This is
an explicit acknowledgement that the result labels only the positive pole; it does not
change or repair the old lens. Direct difference lenses remain signed and use the
pairwise positive/opposite-pole verifier.

## The split: name on one pool, verify on a disjoint one

Battles are split once — deterministically, by hashing a prompt-level group id — into a
**name pool** (≈80%) and a disjoint **verify pool** (≈20%). Naming never touches the
verify pool, so verification is a genuine held-out check, not a re-test of the same
examples.

## Naming (hypothesis generation)

For each feature, PrefScope selects strong activations and silent controls from the name
pool. What the LLM sees depends on the lens:

- A direct difference lens shows the prompt, response A, response B, and signed contrast.
- An individual lens shows one response and its prompt.
- A prompt lens shows the prompt text alone.

The LLM then proposes a short phrase for the positive feature direction. The exact
sampling method is recorded with the output.

These names are **unverified hypotheses**. The CSV's `confidence` is the namer's
self-reported confidence, not a calibrated probability or statistical interval. With
`n_candidates > 1`, PrefScope samples
several views from the strong-activation pool, generates independent proposals, and asks
for one atomic synthesis; `candidate_concepts` retains every proposal for audit. A name
that sounds plausible may still not track the axis — that is what the next stage checks.

### Individual-response proposal review

For an individual lens, responses remain ranked by positive feature activation, but the
sampler first collapses A/B by instruction and keeps the stronger completion. Thus three
displayed activators represent three distinct prompts rather than two answers to one prompt
plus a third example. Silent controls also use distinct, non-active instructions.

The proposal returns one boolean support judgment per example. A second LLM call then acts
as a reviewer: it may retain the candidate or rewrite an overly narrow, broad, compound, or
prompt-topic-based phrase into one atomic response property. The reviewer evaluates the
final wording against the same naming examples. Because these are not independent data,
their support vectors are diagnostics and a cost-saving triage screen, not verification.
PrefScope forwards a candidate only when a strict majority of displayed activators match it
and its active match rate exceeds its control match rate. It does **not** require unanimity.

The name CSV records `reviewed_concept`, `naming_active_support`,
`naming_active_total`, `naming_control_violations`, `naming_control_total`,
`naming_screen_pass`, and `naming_review_action`. The legacy `naming_audit_*` columns remain
as exact-separation diagnostics for compatibility, but no longer define the primary gate.
Raw proposals remain in `candidate_concepts`, and `--debug-responses` writes proposal and
`*_review.txt` responses. For multiple evidence views, the synthesized final concept is
reviewed over their instruction-deduplicated union. The exact reviewed wording is frozen
before held-out verification; it is never revised after observing verification outcomes.

## Verification (the falsification gate)

A name passes the automated gate only if an LLM, shown a response pair and *just
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
the other SAE features. Individual verification also permits only one response per
instruction and excludes active instructions from its control bucket, so correlated A/B
answers do not count as independent observations. This held-out test is distinct from the
same-evidence naming review above.

The default `sampling: extremes` tests detection at the poles. For broader held-out
coverage, `sampling: quantile-stratified` covers weak through strong activations within
the positive, negative, and silent buckets; `n_examples: 300` divides a total
300-judgment budget across them.
For an individual or prompt feature, the same setting splits the budget between positive
activations and silent controls.

The naming and verification clients use the same model by default, although their data
splits and prompts differ. Configure a distinct verifier model when cross-model review is
important; either way, automated verification remains model judgment rather than human
ground truth.

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

`fidelity_pass` validates a name on the verifier's sampled distribution; it does **not**
make every nonzero SAE code a semantic occurrence. For corpus-level prevalence and model
behavior reports, continue with [semantic presence and context](presence-and-context.md).

To change the naming or verification strategy, see
[Add an interpreter](../extending/add-an-interpreter.md) and
[Add a verifier](../extending/add-a-verifier.md). For exact flags, see the
[CLI reference](../reference/cli.md).
