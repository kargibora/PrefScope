# Analyze Config Schema (`prefscope analyze`)

Source of truth: `prefscope/pipeline/analyze_config.py` (`AnalyzeConfig.from_dict`). This schema
applies frozen published lenses to a new dataset. It never trains or renames a lens.
Unknown keys fail closed. Relative local paths resolve from the config file.

## Top level

| key | type | default | meaning |
|---|---|---|---|
| `version` | int | `1` | config schema version |
| `lenses` | mapping | required | completion/prompt lens sources and Hub options |
| `data` | mapping | required | local/Hub source and canonical column mapping |
| `out_dir` | path | required | dedicated managed analysis directory |
| `device` | str | `cpu` | projector device |
| `embedding` | mapping | `{}` | applied-encoding backend/cache controls |
| `concepts` | mapping | `{}` | presence and export controls |
| `analysis` | mapping | `{}` | optional relationship/comparison/preference/outcome stages |
| `viewer` | mapping | `{enabled: true}` | static viewer-bundle controls |

`analysis_state.json` binds the resolved config, local content fingerprints, immutable Hub
commits, and a workflow contract version. Changed semantics or inputs require `--fresh` or
a new output directory.

## `lenses`

Use either shared Hub fields or direct completion/prompt sources.

| key | meaning |
|---|---|
| `repo`, `revision` | shared `owner/repository` and requested Hub ref |
| `completion`, `prompt` | direct local path or `hf://owner/repo[/subfolder]` override |
| `completion_subfolder`, `prompt_subfolder` | subdirectories in the shared Hub repository |
| `completion_annotations`, `prompt_annotations` | optional external interpretation directories |
| `token_env`, `cache_dir`, `local_files_only` | authenticated/cache/offline Hub controls |

Mutable Hub refs resolve to commit SHAs before state is signed or files are loaded.
Offline Hub loading requires a commit SHA.

## `data`

```yaml
data:
  source:
    type: local                 # local | huggingface
    path: ratings.parquet      # local
    # dataset_id: owner/data   # Hugging Face alternative
    # name: config
    # split: train
    # revision: main
    # token_env: HF_TOKEN
    # streaming: false
    # limit: null
  columns:
    prompt: prompt
    response_a: response
    response_b: null
    label: null
    model_a: null
    model_b: null
    item_id: null
    group_id: null
    language: null
    metadata: [helpfulness, conversation_id]
  text:
    prompt_role: null
    response_a_role: null
    response_b_role: null
  label:
    mode: null                 # probability | winner | a-wins
    a_values: []
    b_values: []
    tie_values: []
  mode: single                # auto | single | paired
  drop_empty: true
```

`group_id` is the first-class independent-unit mapping. `metadata` retains other scalar
outcome/group columns in the canonical table and provenance hash. Winner labels require
explicit token mappings.

## `embedding` and `concepts`

| block.key | default | meaning |
|---|---:|---|
| `embedding.backend` | `hf` | embedding backend |
| `embedding.batch_size` | `null` | optional positive batch size |
| `embedding.cache_dir` | `null` | reusable embedding cache |
| `concepts.presence_policy` | `mixed` | `calibrated`, `positive_nonzero`, or `mixed` |
| `concepts.fidelity_only` | `true` | exclude fidelity failures |
| `concepts.named_only` | `true` | exclude unnamed axes |
| `concepts.top_k` | `null` | optional feature limit |
| `concepts.include_text` | `false` | include source text in long concept tables |
| `concepts.chunk_size` | `4096` | export chunk size |

## `analysis`

`relationships`, `comparison`, and `preference` accept `auto`, `true`, or `false`.
`auto` runs only when the needed prompt lens, paired responses, or labels exist.

| key | default | meaning |
|---|---:|---|
| `relationships` | `auto` | prompt→response elicitation table |
| `comparison` | `auto` | label-free paired response comparison |
| `preference` | `auto` | grouped win-relevance |
| `group_col` | `null` | retained independent-group column; otherwise `group_id` or prompt hash |
| `min_support` | `30` | elicitation support |
| `min_cooccur` | `5` | elicitation co-occurrence support |
| `min_context_pairs` | `30` | paired context support |
| `examples_per_direction` | `3` | paired comparison examples |
| `side_a_name`, `side_b_name` | `A`, `B` | response-set labels |
| `outcomes` | `null` | generic outcome-family mapping below |

Outcome mapping:

| key | default | meaning |
|---|---:|---|
| `columns` | required | one column, or several for `multi_continuous` |
| `kind` | required | `binary`, `probability`, `preference`, `continuous`, `multi_continuous` |
| `normalization` | `auto` | `auto`, `none`, or `zscore` |
| `code_array` | `z_a` | `z_a`, `z_diff`, or `z_prompt` |
| `min_units` | `3` | descriptive association floor; thin inferential arms remain unsupported |
| `output` | `outcome_associations.csv` | safe unreserved CSV filename |

Every outcome/group column not already canonical must also appear in
`data.columns.metadata`.

## `viewer`

| keys | defaults / meaning |
|---|---|
| `enabled`, `output_dir` | `true`, `<out_dir>/viewer-data` |
| `examples_per_feature`, `examples_per_group` | `12`, `2` |
| `examples_random`, `examples_boundary` | `4`, `4` |
| `prompt_examples_per_feature`, `prompt_examples_per_group` | `8`, `2` |
| `prompt_examples_random`, `prompt_examples_boundary` | `4`, `4` |
| `joint_examples` | `true` |
| `feature_map`, `prompt_feature_map`, `response_map` | all `true` |
| `map_sample`, `map_sample_mode` | `2500`, `hybrid` (`top-activating`, `random`, `hybrid`) |
| `coactivation_top_k`, `coactivation_max_pairs` | `20`, `20000` |

See [`examples/workflows/analyze-published-lenses.yaml`](../../examples/workflows/analyze-published-lenses.yaml)
for a runnable minimal example. This page, rather than that minimal file, is the complete
accepted-key reference.
