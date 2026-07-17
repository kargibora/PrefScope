# PrefScope pipeline examples

Runnable Python scripts that use the PrefScope **Python API** directly (not the
CLI), so they double as worked examples of how to call the pipeline from your own
code. Each mirrors a CLI stage. Run them with the project's interpreter (`$PY`).

Order (after embeddings + the embedding dump exist):

| # | script | API used | where | needs |
|---|--------|----------|-------|-------|
| 01 | `01_train_lenses.py` | `build_lens_from_embeddings`, `build_prompt_lens` | GPU or CPU | embedding dump |
| 02 | `02_interpret_lens.py` | `load_lens_battles`, `LLMClient`, `name_features`, `verify_features` | LLM access | lens + corpus |
| 03 | `03_cluster_and_winrelevance.py` | `cluster_features`, `summarize_clusters`, `name_clusters`, `win_relevance` | LLM access for naming | lens + corpus w/ `human_pref` |
| 04 | `04_name_prompts.py` | `name_prompt_features` | LLM access | prompt lens + corpus |
| 05 | `prefscope conditional-delta` | `region_behavior_contrast` | CPU | completion + prompt lens |

Notes:
- The CLI (`prefscope <subcommand>`) does the same things; these are the
  library-level equivalents for scripting or extending the pipeline.
- 02/03/04 use an LLM via OpenRouter by default. Set `OPENROUTER_API_KEY` or
  select a configured local backend.
- `win-relevance` (03) needs a corpus built with `build-corpus --keep-labels` so
  the `human_pref` column is present.
- `--input-rep individual` is the lens the inference-time / delta work needs; the
  features score a lone completion and the lens saves `z_diff = f(e_a) - f(e_b)`.
