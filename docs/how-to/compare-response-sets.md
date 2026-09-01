# Compare two response sets on shared prompts

Use this workflow for pretrained versus post-trained checkpoints, two adapters, two
decoding policies, or any other aligned A/B response sets. The comparison is descriptive
and does not need a preference label.

## Data contract

Prepare one row per shared prompt:

```text
prompt · completion_a · completion_b [· model_a · model_b] [· human_pref]
```

The comparison always reports B minus A. `human_pref`, when present, is ignored by this
stage; use `win-relevance` separately when you want a criterion-specific preference
association.

## Encode both sides with one frozen response lens

```bash
prefscope encode-dataset \
  --lens hf://owner/response-lens \
  --data responses.parquet \
  --response-col completion_a \
  --response-2-col completion_b \
  --model-col model_a --model-2-col model_b \
  --metadata-col prompt_group \
  --out analysis/response_codes \
  --device cuda
```

For prompt-conditioned results, encode the same rows with a prompt lens:

```bash
prefscope encode-dataset \
  --lens hf://owner/prompt-lens \
  --data responses.parquet \
  --out analysis/prompt_codes \
  --device cuda
```

The two bundles align by `battle_id`, `item_id`, or `row_id`; their physical row order
does not need to match.

## Calculate paired concept shifts

```bash
prefscope compare-responses \
  --encoded-dir analysis/response_codes \
  --features lenses/response \
  --prompt-encoded-dir analysis/prompt_codes \
  --prompt-features lenses/prompt \
  --side-a-name pretrained \
  --side-b-name posttrained \
  --out analysis/pretrained_vs_posttrained
```

`--features` and `--prompt-features` may be an interpretation directory containing the
name, fidelity, and calibration CSVs, or one already-merged annotation CSV.

The default `--presence-policy calibrated` includes only concepts with a passing learned
semantic threshold, which `prefscope interpret calibrate-presence` produces as
`feature_calibration.csv`. Without that file the run fails with `no features remain
under presence_policy='calibrated'`. For exploratory work, `--presence-policy mixed`
applies thresholds where available and explicitly marks the positive-nonzero fallback
elsewhere.

Concepts are also restricted to named, fidelity-passing features by default. Use
`--include-unnamed` and `--include-unverified` to widen that set.

If several rows are repeated generations of one prompt, preserve their shared identifier
with `encode-dataset --metadata-col prompt_group`, then pass
`compare-responses --group-col prompt_group`. `--metadata-col` is repeatable. Effects and
uncertainty then give each prompt group equal weight instead of treating generations as
independent; a separate unique row identifier still aligns the response and prompt bundles.

## Outputs

```text
comparison.json
concept_shift.parquet
concept_shift_by_context.parquet
response_scope.parquet
paired_examples.parquet
```

For feature $f$, the primary effect is

$$
\Delta_f = \frac{1}{N}\sum_i
\left(\mathbb{1}[f\text{ in }B_i]-\mathbb{1}[f\text{ in }A_i]\right).
$$

The effect is the average change from A to B. When prompts repeat, every prompt group
receives the same weight. `n_nonzero_groups` reports how many independent groups support
the change. Every row states the test, interval method, and presence rule.

**Method details.** With one row per prompt, PrefScope uses the exact McNemar test on
A-only and B-only pairs. With repeated prompt groups, it uses a two-sided Hoeffding bound
for the equal-group average. The confidence interval uses the matching distribution-free
bound, so identical observed changes do not produce a misleading zero-width interval.

`response_scope` separates:

- `general_tendency`: a supported, cross-context-stable response policy, presentation, or
  reasoning-strategy shift;
- `context_specific_tendency`: evidence appears only under particular prompt contexts;
- `prompt_content`: topic, requested task, or requested language rather than a general
  model tendency;
- `unclassified`: insufficient evidence.

These are scope labels, not value judgments. A tendency becomes desirable or undesirable
only relative to an explicit rubric, policy, or preference source.

## Python API

```python
from prefscope import compare_encoded_responses

result = compare_encoded_responses(
    "analysis/response_codes",
    features="lenses/response",
    prompt_dir="analysis/prompt_codes",
    prompt_features="lenses/prompt",
    side_a_name="pretrained",
    side_b_name="posttrained",
)
result.save("analysis/pretrained_vs_posttrained")
```

The lower-level array APIs are `concept_presence`, `paired_concept_shift`,
`paired_concept_shift_by_region`, and `summarize_response_scope` in
`prefscope.analysis`.

## Viewer export

Add the comparison to an ordinary viewer export:

```bash
prefscope-export-viewer \
  --lens-dir lenses/response \
  --comparison-dir analysis/pretrained_vs_posttrained \
  --out viewer-data
```

The viewer exposes it as **Response-set shifts**, keeping response prevalence, paired
evidence, prompt conditioning, and preference outcomes visually separate.

Return to the [documentation home](../index.md).
