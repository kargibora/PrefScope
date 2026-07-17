# Pipeline Config Schema (`prefscope run`)

Source of truth: `prefscope/pipeline/run.py` (`PipelineConfig.from_dict`,
`StageConfig`, `LLMConfig`, `_COMPLETION_OUTPUTS` / `_PROMPT_OUTPUTS`,
`_CLUSTER_CONTROL`, `_WIN_RELEVANCE_KEYS`).

A config is a single YAML (`.yaml`/`.yml`) or JSON file passed to
`prefscope run --config FILE`. The root must be a mapping. Unknown
top-level keys are rejected.

## Top-level keys

| key | type | default | meaning |
|-----|------|---------|---------|
| `lens_dir` | str | **required** | frozen lens directory (must contain `manifest.json`) |
| `out_dir` | str | **required** | all stage artifacts written here |
| `corpus` | str | `None` | merged corpus parquet (re-attaches battle text / `human_pref`) |
| `annotations` | str or list | `None` | OpenJury annotation JSON(s); a bare string is wrapped to a list |
| `lens_kind` | `completion` \| `prompt` | `completion` | which stage set + components apply |
| `stages` | list | full chain for the lens kind | subset/reordering of stages (run order is always canonical regardless of listed order) |
| `llm` | mapping | see [llm](#llm-block) | fallback LLM client shared by name / verify / cluster naming |
| `name_llm` | mapping | `None` | optional naming-specific client (same keys as `llm`) |
| `verify_llm` | mapping | `None` | optional independent verification client |
| `cluster_llm` | mapping | `None` | optional client for behavior/cluster naming |
| `interpreter` | name or mapping | `auto` | `name`-stage component + params |
| `verifier` | name or mapping | `auto` | `verify`-stage component + params |
| `clusterer` | name or mapping | `spherical-kmeans` | `cluster`-stage component + params |
| `win_relevance` | mapping | `{}` | `win-relevance` stage control (only `all_features`) |

`lens_dir` and `out_dir` are the only required keys. `lens_kind` must be exactly
`completion` or `prompt`.

## Stages per `lens_kind`

Each stage maps to a fixed output filename per lens kind:

**`completion`** (`_COMPLETION_OUTPUTS`) — full chain, default
`[name, verify, cluster, win-relevance]`:

| stage | output file |
|-------|-------------|
| `name` | `feature_names.csv` |
| `verify` | `feature_fidelity.csv` |
| `cluster` | `feature_clusters.csv` (+ `feature_clusters_summary.csv`) |
| `win-relevance` | `win_relevance.csv` |

**`prompt`** (`_PROMPT_OUTPUTS`) — default `[name, verify, cluster]`
(`win-relevance` is completion-only and rejected):

| stage | output file |
|-------|-------------|
| `name` | `prompt_feature_names.csv` |
| `verify` | `prompt_feature_fidelity.csv` |
| `cluster` | `prompt_feature_clusters.csv` (+ `prompt_feature_clusters_summary.csv`) |

`stages` may be a subset and may be listed in any order; the runner executes them
in canonical order (`name → verify → cluster → win-relevance`) and skips any not
listed or not applicable. Listing a stage outside the lens kind's set is an error
(with a hint for the `win-relevance`-on-prompt case).

### Preflight requirements
- `lens_dir/manifest.json` must exist.
- paired `completion`: stages in `{name, verify, win-relevance}` need **exactly one** of
  `corpus` / `annotations`; `win-relevance` additionally requires `corpus` (with
  `human_pref`).
- single-response `completion` (`manifest.dataset_mode: single`): `name` / `verify`
  use text retained in the lens and need no external corpus; `win-relevance` is rejected.
- `prompt`: `name` / `verify` require `corpus` (to fetch prompt text).
- If `corpus` is set it must exist on disk.

## Component blocks (`interpreter`, `verifier`, `clusterer`)

Each accepts either a bare component name:

```yaml
verifier: pairwise
```

or a mapping with a `name` (or `component`) key plus constructor params:

```yaml
verifier: {name: pairwise, n_per_bucket: 12}
```

Omitting the block uses the dataclass default component (`auto` for
interpreter/verifier, `spherical-kmeans` for clusterer). Params are validated
against the resolved component's `__init__` signature (`_accepted_params` →
`_check_params`): a param the component does not declare is rejected up front with
the allowed set listed. `auto` resolves to the first registered strategy for the
contract check.

Registered components (from the registry):

| kind | names |
|------|-------|
| interpreter | `auto`, `pairwise`, `individual`, `single-text` |
| verifier | `auto`, `pairwise`, `individual`, `prompt` |
| clusterer | `mi-leiden`, `spherical-kmeans`, `agglomerative` |

> Component param **defaults** are each component's own `__init__` defaults, which
> differ from the `interpret name` CLI flag defaults (e.g. interpreter `n_active`
> defaults to 12 here vs 10 on the CLI). Only the keys you set are passed; the
> rest fall back to component defaults.

Important interpretation parameters:

| block | key | default | meaning |
|-------|-----|---------|---------|
| `interpreter` | `n_candidates` | `1` | independent evidence views/proposals per feature before synthesis |
| `interpreter` | `candidate_pool_factor` | `3` | sample each proposal from this multiple of `n_active` strong activators |
| `verifier` | `sampling` | `extremes` | `extremes` or activation-range `stratified-random` |
| `verifier` | `n_examples` | `None` | total held-out judgment budget per feature (split over sign/control buckets) |
| `verifier` | `min_success_rate` | `0.8` | minimum parse/API success fraction for a pass |
| `verifier` | `min_bucket` | `5` | minimum surviving positive and negative/control judgments |

### Cluster control keys
The clusterer block may also carry these runner-control keys (popped before the
clusterer is constructed; `_CLUSTER_CONTROL`):

| key | type | default | meaning |
|-----|------|---------|---------|
| `cluster_on` | `difference` \| `individual` | `difference` | co-firing space (`z_diff` vs stacked `z_a`/`z_b`) |
| `fidelity_only` | bool | `false` | cluster only fidelity-passing features |
| `name_clusters` | bool | `false` | LLM-name each behavior from member concepts |
| `concurrency` | int | `1` | parallel cluster naming |

### win_relevance keys
Only one key is allowed (`_WIN_RELEVANCE_KEYS`):

| key | type | default | meaning |
|-----|------|---------|---------|
| `all_features` | bool | `false` | score every feature; `false` restricts to fidelity-passing |

## LLM blocks

Validated against `_KEYS`; unknown keys are rejected.

| key | type | default | meaning |
|-----|------|---------|---------|
| `backend` | str | `openai` | LLM backend (`openai` / `claude-cli` / `codex-cli`) |
| `model` | str | `deepseek/deepseek-v3.2` | model id |
| `api_base` | str | `https://openrouter.ai/api/v1` | OpenAI-compatible base URL |
| `api_key_env` | str | `OPENROUTER_API_KEY` | env var holding the API key |

`name_llm`, `verify_llm`, and `cluster_llm` accept the same keys. If absent, that
stage uses the shared `llm`; if present, it builds its own client. This lets a run
record an independent verifier model rather than relying on an informal convention.
Clients are built lazily; configs with only non-LLM stages never construct one.

## Artifact threading
Within a run, downstream stages read upstream outputs from `out_dir`: `verify`
reads the `name` CSV; `cluster` and `win-relevance` prefer the `verify` fidelity
CSV (richer — carries `fidelity_pass` for filtering), falling back to the `name`
CSV, then nothing. A stage not run in this invocation falls back to the canonical
on-disk filename if present.

## Complete annotated example

(`examples/pipeline.yaml`, completion lens, full chain.)

```yaml
lens_dir: lenses/indiv_8b              # frozen completion lens (z_diff/z_a/z_b + manifest)
corpus: corpora/arena_merged.parquet  # win-relevance needs human_pref; also re-attaches text
out_dir: results/pipeline/indiv_8b    # all artifacts land here

stages: [name, verify, cluster, win-relevance]   # omit to run the full chain

llm:                                   # one client shared by name / verify / cluster naming
  backend: openai                      # openai | claude-cli | codex-cli
  model: deepseek/deepseek-v3.2
  api_base: https://openrouter.ai/api/v1
  api_key_env: OPENROUTER_API_KEY

interpreter:
  name: auto                           # auto -> individual/pairwise from manifest input_rep
  n_active: 12
  n_zero: 8

verifier:
  name: auto                           # auto -> individual/pairwise; or pairwise/individual
  n_per_bucket: 12
  fidelity_threshold: 0.3

clusterer:
  name: mi-leiden                      # mi-leiden | spherical-kmeans | agglomerative
  resolution: 1.2                      # mi-leiden: higher -> more, smaller behaviors
  knn: 6                               # mi-leiden: sparsify to each feature's top-k edges
  # cluster_on: difference             # difference (z_diff) | individual ([z_a; z_b])
  # fidelity_only: true                # cluster only fidelity-passing features
  # name_clusters: true                # LLM-name each behavior

win_relevance:
  all_features: false                  # false -> restrict to fidelity-passing features
```

For the higher-cost multi-candidate and 300-judgment protocol, including separate
naming and verification models, see [`examples/research.yaml`](../../examples/research.yaml).
