# Semantic presence and context

A feature name may fit the strongest examples but fail on weaker activations. PrefScope
therefore keeps three claims separate:

1. **Extreme fidelity**: does the name fit separate examples with high activation and
   not fit silent controls? This is the default `interpret verify` check.
2. **Semantic presence**: above what activation is the name reliable enough? PrefScope
   chooses a threshold on one set of prompt groups and checks it on another set.
   `interpret calibrate-presence` writes both steps to `feature_calibration.csv`.
3. **Model tendency**: after applying the threshold, does one model choose the property
   more often than its opponent across several prompt types? `context-profile` writes
   this evidence.

This distinction matters for properties such as “predicts sports scores in a table.” It
may be a real, coherent response feature while still being almost entirely determined by
the requested task. It belongs in **prompt/content**, not in a model's list of general
tendencies.

## Calibrate ordinary activations

```bash
prefscope interpret calibrate-presence \
  --lens-dir lenses/completion \
  --corpus corpus.parquet \
  --names lenses/completion/feature_names.csv \
  --fidelity lenses/completion/feature_fidelity.csv \
  --out lenses/completion/feature_calibration.csv \
  --backend openai --model google/gemma-3-27b-it \
  --concurrency 4 --resume
```

PrefScope chooses a threshold on one group of prompts and checks it on a separate
group. It uses at most one response from each prompt group. A feature passes only when
enough checked examples match its name and silent examples rarely match. The first group
can choose the threshold but can never set `presence_pass=True`.

The exact default rule uses a 95% Wilson interval. It requires at least 20 labeled active
prompt groups. The lower confidence bound for precision must be at least 0.8, and the
upper bound for matches among silent controls must be at most 0.2. Missing or insufficient
controls do not pass. The settings and implementation hash are stored so `--resume` only
reuses a matching run.

## Profile prompt dependence

The completion and prompt lenses must contain the same battle IDs in the same order.

```bash
prefscope context-profile \
  --completion-lens lenses/completion \
  --prompt-lens lenses/prompt \
  --calibration lenses/completion/feature_calibration.csv \
  --names lenses/completion/feature_names.csv \
  --prompt-names lenses/prompt/prompt_feature_names.csv \
  --prompt-fidelity lenses/prompt/prompt_feature_fidelity.csv \
  --prompt-clusters lenses/prompt/prompt_feature_clusters.csv \
  --out lenses/completion/feature_context.csv \
  --model-out lenses/completion/model_feature_context.parquet
```

Prompts may belong to several verified concepts or clusters simultaneously. Context
breadth and model stability use that overlapping membership rather than reducing each
prompt to its strongest feature. If prompt calibration is available, pass it with
`--prompt-calibration` and select `--prompt-presence-policy calibrated`.

The feature table records semantic prevalence, prompt dependence, context change, paired
A/B choice rate, and context breadth. Prompt dependence uses normalized mutual
information (NMI). Each feature also receives one category:

- `general`: a response policy/style/reasoning property with broad prompt support and a
  meaningful paired choice rate;
- `context_specific`: a model choice, but not stable or broad enough to call general;
- `prompt_content`: requested task, language, or subject matter largely fixed by the prompt;
- `unclassified`: evidence is missing or does not satisfy a category rule.

The model-feature table adds an exact sign test for A/B differences and a
Benjamini–Hochberg (BH) correction for multiple tests within each model. It also reports
whether the direction is consistent across contexts. “Good” and “bad” are not categories:
preference association is a separate, dataset-specific analysis.

Return to the [documentation home](../index.md) or read the
[prompt-concept extraction guide](../how-to/extract-prompt-concepts.md).
