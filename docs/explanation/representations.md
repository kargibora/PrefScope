# Representations: difference vs. individual

A contrast lens has to turn a *pair* of completion embeddings `(e_A, e_B)` into the
rows its SAE trains on, and into the code it produces at inference. There are two
built-in ways to do that — `difference` and `individual`. They are **co-equal
choices**, not a method and its ablation: each is a legitimate lens with different
properties. Pick the one that matches what you will do with the lens.

The choice is recorded in the lens `manifest.json` as `input_rep`, so everything
downstream reads it back rather than assuming one. In the code, each is a `LensRep`
registered under its name (`prefscope/pipeline/lens_rep.py`).

## `difference` — train on the contrast

```
training rows:  x = e_A − e_B          (one row per battle)
inference code: z = f(e_A − e_B)
```

The SAE sees the **contrast vector** directly. Subtraction removes the *common mode* —
whatever A and B share (topic, language, format both got right) — and keeps the
*contrast*: how A differs from B. So the features are **contrast directions**, and
the code is literally `project(e_A − e_B)`. Two consequences:

- Reconstruction explained variance is modest by design: most of each embedding's
  energy was common-mode that you deliberately removed, so there is less variance left
  to explain. A low-ish recon number here is expected, not a failure.
- `x` is **not** renormalized — its magnitude encodes how different the pair is.

This is the WIMHF-style default. It is the right choice when your unit of analysis is
always a *pair* and you want axes that are explicitly about A-vs-B difference.

## `individual` — train on pooled single responses

```
training rows:  [e_A ; e_B]            (two rows per battle, pooled)
inference code: z = f(e_A) − f(e_B)
```

Here the SAE trains on **single** completion embeddings, pooled from both sides. Its
encoder `f` therefore applies to *any lone response* — and the contrast code is formed
*after* projection, as `f(e_A) − f(e_B)`. A difference lens cannot do this: its
encoder only ever saw contrasts. The lens saves `z_a`, `z_b`, and the derived
`z_diff = z_a − z_b`.

This is the right choice when you need to score a **single** response on its own — any
inference-time use where you do not have a paired opponent (e.g. evaluating a draft,
or feature-conditioned analysis of one completion).

## The crucial inequality

Because the SAE is non-linear (a magnitude threshold; see
[the lens](the-lens.md#the-non-linearity-caveat)):

$$
f(e_A) - f(e_B) \;\neq\; f(e_A - e_B).
$$

The two representations are genuinely different functions, not two spellings of the
same thing. `difference` projects the contrast; `individual` contrasts the
projections. That is exactly why they are co-equal choices rather than interchangeable
implementations.

## The OOD pitfall

The one thing you must **not** do: project a *single* response through a
**difference-trained** SAE. That encoder only ever saw contrast vectors `e_A − e_B`
during training; a lone embedding `e_A` is out of its training distribution, and the
resulting code is not meaningful. PrefScope guards this by **not saving**
per-side codes for a difference lens — only `z_diff` is written. If you need to score
single responses, build an `individual` lens.

## Which to pick

| You want to… | Use |
|--------------|-----|
| Analyze A-vs-B contrasts; axes about *how a pair differs* | `difference` |
| Score a *single* response, or contrast projections post-hoc | `individual` |
| Analyze prompts (no A/B pair at all) | a prompt lens (`build-prompt-lens`) |

A prompt lens is a third, non-contrastive case: a plain SAE over single prompt
vectors, with no A/B pairing. Its `LensRep` is marked non-contrastive and the
contrast operations (bank / diagnose) refuse it with a clear message rather than
producing nonsense.

For the exact build flags (`--input-rep`, `--dump-embeddings` for cheap re-fits), see
the build-and-analyze how-to; to add a new representation as a registered `LensRep`,
see `docs/extending/`.
