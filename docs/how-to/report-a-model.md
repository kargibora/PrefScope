# Report a model's concept profile

Use this guide to make a readable concept report for one model. The report compares the
model with its opponents on matched prompts. It can show common differences, missing
preference-associated concepts, and prompt types where the pattern changes.

The report command shows model-minus-opponent differences. It does not show absolute
concept prevalence. The viewer can show absolute prevalence when you use an individual
lens.

This report is a presentation layer over [diagnosis](diagnose-a-model.md).

## Prerequisites

- A lens with concept names and `feature_fidelity.csv`. See
  [Build and analyze a lens](build-and-analyze-a-lens.md).
- A corpus or OpenJury annotation file with `model_a` and `model_b`.
- Optional: `win_relevance.csv` for preference-associated gaps.
- Optional: a prompt lens for the prompt-types section.

## Run it

```bash
prefscope report \
    --lens-dir lenses/mylens --model my-model \
    --corpus battles.parquet \
    --names results/mylens/feature_fidelity.csv \
    --win-relevance results/mylens/win_relevance.csv \
    --prompt-lens lenses/promptlens \
    --prompt-names results/promptlens/prompt_feature_names.csv \
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
tab) when you export a bundle with `prefscope-export-viewer`. It renders every
section as charts and makes the **prompt types clickable** — clicking a prompt type
shows sample battles of that model on that prompt type (its answer vs the opponent's,
with the outcome). Pass `--report-battles` to populate that drill-in:

```bash
prefscope-export-viewer --lens-dir lenses/mylens --corpus battles.parquet \
    --prompt-lens lenses/promptlens --completion-lens lenses/mylens \
    --prompt-interpret-dir results/promptlens \
    --out viewer-data --report-battles
```

The drill-in needs a built bank (for the per-model diagnosis) and a prompt lens. Without
`--report-battles` the prompt-type rows still show win rates and battle counts, just
without the click-through.
