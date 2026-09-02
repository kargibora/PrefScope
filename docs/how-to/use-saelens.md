# Use a pretrained SAE through SAELens

PrefScope can use a compatible pretrained internal-activation SAE instead of training a
new embedding lens. The public dataset interface is the same as for other lenses:

```text
PairItem rows -> lens.featurize(...) -> FeatureBatch -> analysis tools
```

This is an optional, experimental backend.

## Install and load

```bash
uv sync --extra saelens
```

Load it directly:

```python
from prefscope import Lens

lens = Lens.from_saelens(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device="cpu",
)
```

Or use the same settings from YAML:

```yaml
version: 1
backend: saelens
release: gpt2-small-res-jb
sae_id: blocks.8.hook_resid_pre
device: cpu
text_batch_size: 8
long_text_policy: truncate
include_bos: false
```

```python
lens = Lens.from_config("lens.yaml")
```

The adapter uses SAELens 6.50 or later. It loads registered releases by default. An
unregistered repository requires `allow_unregistered_release=True`, which is an
explicit trust decision.

## Encode a paired preference dataset

```python
from prefscope import TableDataset, preference_relevance

items = TableDataset(
    "preferences.parquet",
    prompt="prompt",
    a="response_a",
    b="response_b",
    pref="preference",          # P(A preferred), including 0.5 ties
    id="pair_id",
    group_id="prompt_id",
)
features = lens.featurize(items)

z_prompt = features.matrix("z_prompt")
z_a = features.matrix("z_a")
z_b = features.matrix("z_b")
z_diff = features.matrix("z_diff")
win_rates = lens.preference_relevance(features)

from prefscope import save_feature_batch
save_feature_batch(features, "analysis/features")
```

One reader and one SAE produce every view in the same feature coordinates. The paired
quantity is

```text
z_diff = max_t f(h_A,t) - max_t f(h_B,t)
```

It is not `f(h_A-h_B)` and not `f(pool_t h)`. `z_a` and `z_b` are nonnegative for a
nonnegative SAE, while their derived difference is signed. PrefScope records these
semantics separately for each view.

The default text policy treats the prompt, response A, and response B as independent
documents. It records `text_context=independent_documents`, truncation policy, BOS and special-token exclusion policies, sequence-position
selection, hook, and reader coordinates. It does not silently
claim that response features are prompt-conditioned model states. Prompt-conditioned
and chat-template rendering need a separate explicit text policy and are not yet
provided by this backend.

For wide SAEs, select a feature subset:

```python
features = lens.featurize(items, feature_ids=(12, 41, 99))
```

Selected feature identities are preserved through `FeatureBatch`, saved bundles,
matrices, and `preference_relevance`.

The same flow is available as a readable example:

```bash
python examples/advanced/analyze_saelens_pairs.py \
  --lens-config examples/inference/saelens.yaml \
  --data preferences.parquet --out analysis/saelens-features \
  --item-id-col pair_id --group-id-col prompt_id
```

## Analyze with the typed API

A `FeatureBatch` can be analyzed directly, or one view can be selected:

```python
from prefscope import OutcomeSpec, analyze_dataset

preference = OutcomeSpec.from_feature_batch(features)
result = analyze_dataset(
    features.matrix("z_diff"),
    outcomes={"preference": preference},
    group_ids=features.metadata["group_id"],
)
associations = result.outcome_associations
```

`preference_relevance` returns one row per feature. Main fields include `win_assoc`
(`win_rate_a_more - win_rate_a_less`), `preferred_side_rate`, firing/support counts,
group counts, tests, and explicit orientation/estimand columns. It preserves ties as
neutral `0.5` values where the estimand permits them.

Post-SAE max pooling gives longer texts more opportunities to reach a large maximum.
The basic relevance table is not length-controlled. Inspect the recorded response
lengths and run the framework's length-confound screen before interpreting an
association as content-specific.

Preference associations remain descriptive and dataset/judge-specific. They do not say
that a feature is good, bad, or causal.

## Inspect one prompt

```bash
.venv/bin/python examples/advanced/saelens_prompt_concepts.py \
  --prompt "Explain why the sky appears blue during the day." --top 8
```

The script also retrieves available Neuronpedia descriptions. They are external
proposed labels, not PrefScope-verified semantic-presence claims.

## Advanced exact-activation path

Callers that already own exact hook activations can bypass the built-in text reader:

```python
features = lens.project_saelens_tokens(
    row_ids=tuple(item_ids),
    token_activations={
        "response_a": response_a_token_activations,
        "response_b": response_b_token_activations,
    },
    token_row_ids={
        "response_a": response_a_token_item_ids,
        "response_b": response_b_token_item_ids,
    },
    representation_contract=extraction_contract,
    feature_ids=selected_feature_ids,
    metadata={"group_id": tuple(prompt_group_ids)},
)
```

The extraction contract must be produced independently. Copying the expected contract
is only a caller assertion; PrefScope cannot prove that arbitrary arrays came from the
claimed model.

## Memory, shape, and artifact limits

- SAE encoding happens per token before item pooling.
- Dense outputs are bounded by `max_output_bytes`.
- Flat standard, gated, TopK, JumpReLU, and transcoder variants are supported.
- Structured hooks and temporal SAEs fail closed.
- External release names and SAE IDs are not immutable PrefScope artifacts.
- Reader models and pretrained checkpoints keep their own licenses and access rules.
- Raw activity is not semantic presence. Labels still require target-data verification
  and calibration for semantic claims.
