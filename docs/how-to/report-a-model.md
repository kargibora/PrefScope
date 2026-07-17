# Report a model's concept profile

Goal: from a set of battles (prompts + answers) and a named, verified lens, get a
readable **concept report card** for one model — the concepts it most/least
distinguishes itself from opponents on, which preferred concepts it under-expresses,
and which prompt types it's strong or weak on. (The CLI report's per-feature rate is a
target-minus-opponent *contrast*, not absolute prevalence — the viewer's report card
shows absolute prevalence `P(z>0)` from an individual lens.)

This is a presentation layer over [diagnosis](diagnose-a-model.md): same per-feature
signals, assembled into one report.

## Prerequisites

- A built lens with concept names + fidelity ([build & analyze](build-and-analyze-a-lens.md)) — `feature_names.csv` / `feature_fidelity.csv`.
- Battles where your model played (a corpus parquet or OpenJury annotation JSON), with `model_a`/`model_b` so the model can be oriented as "self".
- *Optional:* a `win_relevance.csv` (to surface **rewarded gaps**) and a prompt lens (to add the **prompt-types** section).

## Run it

```bash
prefscope report \
    --lens-dir lenses/mylens --model my-model \
    --corpus battles.parquet \
    --names lenses/mylens/feature_fidelity.csv \
    --win-relevance results/mylens/win_relevance.csv \
    --prompt-lens lenses/promptlens \
    --prompt-names lenses/promptlens/prompt_feature_names.csv \
    --out report.md --device cuda
```

It prints the report, writes it to `report.md`, and writes the underlying per-feature
diagnosis to `report_features.csv` next to it.

## What you get

```
# my-model — concept report card
123 battles · win rate 47%

## Frequently distinguishes from opponents
- refuses — differs from opponent in 34% of battles
- very descriptive — differs from opponent in 28% of battles

## Rarely distinguishes from opponents
- gives worked examples — differs from opponent in 3% of battles

## Rewarded gaps
- worked examples — under-expressed, +0.12 Δwin (length-controlled)

## Strong / weak prompt types

Strongest:
- coding — win rate 58% (n=120)
- factual Q&A — win rate 55% (n=90)

Weakest:
- multi-step reasoning — win rate 38% (n=75)

## Prompt → Response

- coding ⇒ code blocks — +0.09 Δwin (n=120)
- multi-step reasoning ⇒ refuses — -0.06 Δwin (n=75)
```

The **Prompt → Response** section is per-model: within each prompt type, it contrasts
the model's win rate when it produces a response concept against when it doesn't, so a
positive Δwin means producing that concept *given that kind of prompt* helps this model
win. It needs the prompt lens (same `--prompt-lens` flag).

What each input adds:

| Input | Adds |
|-------|------|
| `--names` | concept names (so axes read as "refuses", not "feature 12"); restricts to fidelity-passing axes by default (`--all-features` to include all) |
| `--win-relevance` | the **Rewarded gaps** section (concepts the model under-expresses *and* humans reward) |
| `--prompt-lens` (+ `--prompt-names`) | the **Strong / weak prompt types** section (the model's win-rate per prompt concept) *and* the per-model **Prompt → Response** section (which prompt concepts elicit which response concepts, and whether that helps the model win) |
| `--bank` | measures under-expression vs the model *pool* (`delta_vs_pool`) instead of vs zero |

`--top` sets how many concepts each section lists; `--min-battles` filters prompt
types with too few battles. The embedder is read from the lens manifest — see the
[CLI reference](../reference/cli.md) for the shared embedder flags.

## Interactive version (web viewer)

The same report card is available, interactively, in the web viewer (the *Report card*
tab) when you export a bundle with `scripts/export_viewer_data.py`. It renders every
section as charts and makes the **prompt types clickable** — clicking a prompt type
shows sample battles of that model on that prompt type (its answer vs the opponent's,
with the outcome). Pass `--report-battles` to populate that drill-in:

```bash
python scripts/export_viewer_data.py --lens-dir lenses/mylens --corpus battles.parquet \
    --prompt-lens lenses/promptlens --completion-lens lenses/mylens \
    --prompt-interpret-dir results/promptlens \
    --out viewer_data --report-battles
```

The drill-in needs a built bank (for the per-model diagnosis) and a prompt lens. Without
`--report-battles` the prompt-type rows still show win rates and battle counts, just
without the click-through.
