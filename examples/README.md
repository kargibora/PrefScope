# PrefScope examples

Run examples from the repository root with the project environment.

## Core flow

| Example | Purpose |
|---|---|
| `inference/single_item.py` | feature one `PairItem` |
| `inference/local_dataset.py` | feature a local dataset |
| `inference/huggingface_dataset.py` | feature a Hugging Face dataset |
| `analysis/inspect_local_features.py` | apply direct numerical helpers |
| `advanced/custom_analysis_api.py` | write an ordinary custom function, with no component base class |
| `reporting/compile_report.py` | collect caller-owned tables and metrics in `Report` |

## Lens construction

- `training/train_completion_lens.py`
- `assets/make_sample_corpus.py`

These can require optional model and Torch dependencies.

## Specialized research recipes

- `analysis/preference_relevance.py`
- `analysis/outcome_association.py`
- `advanced/compare_model_stages.py`
- `advanced/analyze_saelens_pairs.py`
- `advanced/saelens_prompt_concepts.py`
- `advanced/presentations/compare_completions.py`

Specialized examples explicitly import code from `prefscope.recipes` where needed. They are
starting points, not stable public API, automatic pipelines, or universal analysis
protocols. Review their assumptions before using them in a study.

Examples do not log private text automatically. The caller decides what input and output
artifacts are safe to retain.
