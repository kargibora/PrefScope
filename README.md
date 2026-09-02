<h1 align="center">
  <img src="https://raw.githubusercontent.com/kargibora/PrefScope/main/docs/assets/prefscope-logo.jpg" alt="PrefScope" width="280">
</h1>

A win rate can show that one response set wins more often. It does not show which
recurring prompt or response patterns co-occur with those wins. **PrefScope turns
post-training datasets into reusable concept-level artifacts for that analysis.**

A PrefScope **lens** maps prompts and responses into sparse features. A configured LLM
can propose feature names and check them on held-out examples. The frozen lens can then
be reused on new datasets. PrefScope produces separate artifacts for raw activity,
semantic presence, prompt context, preference association, model differences, and
possible confounds.

The main API is backend-neutral:

```text
PairItem rows → Lens.featurize(...) → FeatureBatch → analysis artifacts
```

The same analysis code works with native PrefScope embedding lenses, published lenses,
precomputed features, compatible pretrained SAELens checkpoints, custom backends
injected directly in Python, and explicitly registered config backends.

PrefScope reports what a dataset contains and what its labels are associated with. It
does not turn those associations into universal, causal, or objective “good versus bad”
claims. A feature is an axis in the chosen reader representation—not a ground-truth
ontology or evidence of a mechanism in the model that generated the response. Results
can vary with the reader, SAE and interpretation seeds, and concept coverage.

For example, suppose a paired dataset contains two answers and a winner label per prompt.
A frozen lens might expose an axis that an LLM proposes to call “citation use.” PrefScope
keeps the held-out evidence for that name separate from a dataset-level table showing
whether A-minus-B activity on the axis is associated with the winner. This is an
illustrative workflow, not a reported finding.

## The artifact lifecycle

PrefScope separates stages so that changing a late analysis does not rerun an expensive
early stage:

```text
raw post-training data
        │
        ▼
normalized PairItem rows
(prompt, response A, optional response B, labels, groups, metadata)
        │
        ├── build a native lens
        ├── load a published lens
        └── wrap a compatible external backend
        │
        ▼
reusable lens + explicit provenance
        │
        ├── propose and verify feature names
        ├── calibrate semantic-presence thresholds
        └── freeze and reuse on new data
        │
        ▼
aligned FeatureBatch
        │
        ├── concept inventory and context
        ├── prompt → response relationships
        ├── preference and outcome associations
        ├── paired model/checkpoint comparisons
        └── measured-confound screens
```

The durable lens artifact stores the feature encoder and its manifest. Interpretation
and analysis tables remain explicit artifacts rather than being collapsed into one
composite score.

## Why PrefScope

- **Reusable artifacts.** Build and interpret a lens once, then apply it to new datasets.
- **One analysis boundary.** Every backend returns an aligned, role-aware
  `FeatureBatch` with feature IDs, orientation, numerical semantics, metadata, and
  provenance.
- **Evidence stays separated.** Raw firing, proposed names, held-out name fidelity,
  calibrated semantic presence, context profiles, and preference associations are not
  treated as the same claim.
- **Pairs are explicit.** Response A, response B, and `z_A - z_B` have declared roles and
  orientations. Direct contrast projection is kept distinct from per-side encoding.
- **Groups are first-class.** In group-aware analyses, repeated versions of one prompt
  can share a group ID so they do not receive extra inferential weight.
- **Extension is deliberate.** Python callers can inject a backend directly. Config-driven
  custom backends require explicit trusted registration; PrefScope never scans installed
  packages for plug-ins.

Unlike per-row LLM tagging, PrefScope learns and freezes a reusable feature basis rather
than asking an LLM to recreate the vocabulary for every dataset. Unlike an SAE dashboard,
it carries that basis into grouped, dataset-level comparisons and outcome associations.
Unlike a one-off regression, it preserves the reader representation, proposed label,
semantic-presence evidence, and outcome association as separate artifacts.

## Choose a workflow

| If you want to… | Start here |
|---|---|
| Build and interpret a new lens | [Quickstart](#quickstart-build-a-lens) |
| Featurize one item or dataset with an existing lens | [Small Python examples](#self-contained-example-gallery) |
| Analyze a dataset with an already interpreted lens | [`prefscope analyze`](#analyze-with-published-lenses) |
| Use a pretrained internal-activation SAE | [Use SAELens](https://github.com/kargibora/PrefScope/blob/main/docs/how-to/use-saelens.md) |
| Add a custom lens backend | [Add a lens backend](https://github.com/kargibora/PrefScope/blob/main/docs/extending/add-a-lens-backend.md) |
| Inspect one prompt and response | `prefscope extract-concepts` |
| Export all active concepts across a dataset | `prefscope concepts` |

Most new researchers should build and inspect a lens first. Use `prefscope analyze` when
you already have compatible prompt and response lenses. The repository IDs in the
published-lens examples are templates, not first-party releases.

## Installation

The core package can inspect artifacts and call remote services without importing
PyTorch:

```bash
python -m pip install prefscope
```

Install only the capabilities you need:

```bash
python -m pip install "prefscope[cpu]"       # apply/build a lens on CPU or Apple MPS
python -m pip install "prefscope[cluster]"   # cofire/MI Leiden clustering
python -m pip install "prefscope[arena]"     # Hugging Face arena loaders
python -m pip install "prefscope[viewer]"    # Streamlit viewer
python -m pip install "prefscope[saelens]"   # experimental pretrained SAEs
```

For a source checkout:

```bash
git clone https://github.com/kargibora/PrefScope.git prefscope
cd prefscope
uv sync --extra cpu --extra cluster
source .venv/bin/activate       # macOS/Linux
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. PrefScope requires
Python 3.10 or newer. For GPU work, install the appropriate PyTorch build from the
[official selector](https://pytorch.org/get-started/locally/) before installing
PrefScope.

Feature naming and verification require an LLM. PrefScope supports an OpenAI-compatible
HTTP service, Claude CLI, or Codex CLI. A hosted OpenRouter setup uses
`OPENROUTER_API_KEY`; a local vLLM server uses `--api-base`.

## Quickstart: build a lens

The package can generate a complete 60-battle synthetic workspace. The small dataset is
only a smoke test, not scientific evidence.

```bash
# 1. Generate and inspect normalized paired data. No model or network is needed.
prefscope init-demo --out demo
prefscope inspect --corpus demo/sample_corpus.parquet

# 2. Embed responses and train a small individual-response lens.
prefscope build-lens --corpus demo/sample_corpus.parquet \
    --embed-model-id Qwen/Qwen3-Embedding-0.6B --device cpu \
    --input-rep individual --out demo/lens --m-total 16 --k 4

# 3. Propose names, verify them, cluster features, and measure preference association.
OPENROUTER_API_KEY=... prefscope run --config demo/quickstart.yaml
```

The individual representation is deliberate: it scores A and B separately, then forms
`f(A) - f(B)`. A direct difference lens instead learns axes from input contrasts and
cannot safely score a response by itself.

A successful smoke run creates the lens directory plus `feature_names.csv`,
`feature_fidelity.csv`, `feature_clusters.csv`, and `win_relevance.csv` under the
configured output directory. Step 2 downloads the embedding model, and its runtime is
hardware-dependent. Step 3 needs paid or local LLM access and records its usage.

`run` is the canonical name → verify → cluster → preference-association workflow. Its
fidelity table supports interpretation of extreme activations; it does not establish
that the named concept is present in ordinary rows. Before making semantic-presence
claims, run `prefscope interpret calibrate-presence`. See
[Presence and context](https://github.com/kargibora/PrefScope/blob/main/docs/explanation/presence-and-context.md) for the required
selection and disjoint-confirmation stages.

For a guided explanation, see [Your first lens](https://github.com/kargibora/PrefScope/blob/main/docs/tutorials/your-first-lens.md).
[`examples/workflows/research.yaml`](https://github.com/kargibora/PrefScope/blob/main/examples/workflows/research.yaml) shows a higher-cost research profile.

## Self-contained example gallery

Examples are grouped by capability under `examples/inference`, `examples/training`,
`examples/analysis`, `examples/workflows`, and `examples/advanced`. Basic scripts keep
editable constants at the top and run without command-line configuration:

```bash
.venv/bin/python examples/analysis/outcome_association.py
.venv/bin/python examples/inference/single_item.py
.venv/bin/python examples/inference/local_dataset.py
.venv/bin/python examples/analysis/inspect_local_features.py
.venv/bin/python examples/inference/huggingface_dataset.py
.venv/bin/python examples/training/train_completion_lens.py
```

Each runnable card uses `observe_run(..., pretty=True)` for compact progress and prints a
small result to stdout. The log remains separate from the scientific output. Inference
cards show `Lens.from_config → PairItem rows → Lens.featurize → save_feature_batch` and
can switch between the adjacent SAELens and native-lens YAML files by editing one
constant.

Feature bundles retain input text and labels, so they are local, private artifacts. Raw
SAE activity is not semantic presence, reward, or quality. `FeatureCatalog` keeps proposed
names/descriptions separate from numerical batches; `feature_activation_table(...)` joins
by feature ID, and `FeatureTableRenderer` prints the resulting table. See the
[`examples/README.md`](examples/README.md) capability matrix for status and prerequisites.

## Analyze with published lenses

When prompt and response lenses are already trained and interpreted, one config can
prepare the data, load the lenses, featurize the rows, and export the analysis artifacts.
From a source checkout, start with the repository template:

```bash
cp examples/workflows/analyze-published-lenses.yaml analysis.yaml
# Edit the lens repositories, dataset columns, and output directory.
prefscope analyze --config analysis.yaml
```

A wheel installation does not include the repository's `examples/` directory; download
or write the YAML config before running the same `prefscope analyze` command.

Common settings can be overridden without editing the file:

```bash
prefscope analyze --config analysis.yaml \
  --data another.parquet --out analysis/another --device cuda \
  --set data.columns.response_a=answer \
  --set data.source.limit=10000
```

Paired comparisons run only when response B exists. Preference analysis runs only when
labels exist. Naming and SAE training are not repeated. Resume keys include resolved
settings and local input fingerprints; `--fresh` refuses to replace an unsafe or
unrecognized directory.

`analyze` uses bundled calibration artifacts when available. Its default `mixed` presence
policy keeps an explicitly labeled `positive_nonzero` fallback for exploration; those
rows are not semantic-presence claims. For a fail-closed semantic analysis, set
`concepts.presence_policy=calibrated`.

See [Bring your own dataset](https://github.com/kargibora/PrefScope/blob/main/docs/how-to/bring-your-own-dataset.md) for local files, Hugging
Face datasets, single responses, response pairs, winner labels, ties, and custom column
mappings.

## Backend-neutral Python API

The recommended dataset operation is `Lens.featurize(...)`:

```python
from prefscope import Lens, TableDataset, save_feature_batch

lens = Lens.from_config("lens.yaml")
items = TableDataset(
    "preferences.parquet",
    prompt="prompt",
    a="response_a",
    b="response_b",
    pref="preference",       # always P(A preferred); ties may be 0.5
    id="pair_id",
    group_id="prompt_id",
)

features = lens.featurize(items)
# Supported paired backends can return z_prompt, z_a, z_b, and z_diff = z_a - z_b.
save_feature_batch(features, "analysis/features")
if "z_diff" in features.arrays:
    preference = lens.preference_relevance(features)
```

`FeatureBatch` is the interoperability boundary. Downstream analysis does not need to
know whether features came from embeddings, residual activations, a published lens,
precomputed arrays, SAELens, or a hosted custom system.

Published PrefScope lenses can also be loaded directly:

```python
from prefscope import Lens, PairItem

lens = Lens.from_pretrained("owner/repository", revision="v1", device="cpu")
items = [
    PairItem(
        id="row-1",
        x="Explain gravity",
        y_a="Response A",
        y_b="Response B",
        pref=1.0,
        model_a="candidate",
        model_b="baseline",
    )
]
features = lens.featurize(items)
```

Historical `encode`, `encode_items`, and `encode_pairs` methods remain available for
compatibility. New code should use `featurize` and typed analysis contracts. See the
[Python API reference](https://github.com/kargibora/PrefScope/blob/main/docs/reference/python-api.md),
[lens config schema](https://github.com/kargibora/PrefScope/blob/main/docs/reference/lens-config-schema.md), and
[API stability guide](https://github.com/kargibora/PrefScope/blob/main/docs/reference/api-stability.md).

## Keep the evidence layers separate

| Artifact | What it supports | What it does **not** establish |
|---|---|---|
| Raw feature activity | A numerical feature fired | The proposed concept is semantically present |
| Proposed feature name | An external description of an axis | Held-out fidelity or human agreement |
| Held-out fidelity | The configured verifier judges that the name fits separate extreme activations | Human agreement or ordinary-activation semantic presence |
| Calibrated presence | A threshold met target verifier precision and was independently confirmed | Human ground truth or a causal model mechanism |
| Context profile | A response tendency differs by detected prompt context | General behavior when prompt coverage is incomplete |
| Preference association | A dataset or judge prefers one side more often when a feature differs | Universal quality, causality, or an intervention effect |

Reader-model activations describe the reader representation. They do not by themselves
show a mechanism inside the model that generated the response.

## What PrefScope can produce

| Question | Typical artifact |
|---|---|
| What recurring axes exist, and what might they mean? | `feature_names.csv`, `feature_fidelity.csv` |
| Which named concepts are detected in ordinary prompts or responses? | `feature_calibration.csv`, concept activation tables |
| Which prompt concepts elicit which response concepts? | prompt/response relationship tables |
| Which concept differences are associated with preference or another outcome? | `win_relevance.csv`, `outcome_associations.csv` |
| How do two models, checkpoints, or response sets differ on matched prompts? | paired concept-shift artifacts |
| Does a preference-associated feature also co-vary with response length? | length-confound screen artifacts |
| How can the artifact be explored? | static viewer export or `prefscope-view` |

Use only the analyses required by the research question. PrefScope deliberately does not
combine them into one framework score.

## Extending PrefScope

Use the narrowest contract that fits the extension:

- implement `LensBackend` for token-level, hosted, or otherwise nonstandard lenses;
- implement `RepresentationSource` for fixed-width embeddings or pooled activations;
- implement `AnalysisComponent` for a reusable downstream analysis;
- register trusted config-driven components explicitly with the registry.

See [Add a lens backend](https://github.com/kargibora/PrefScope/blob/main/docs/extending/add-a-lens-backend.md),
[Extending PrefScope](https://github.com/kargibora/PrefScope/blob/main/docs/extending/the-registry.md), and
[`examples/advanced/custom_analysis_api.py`](https://github.com/kargibora/PrefScope/blob/main/examples/advanced/custom_analysis_api.py).

## Documentation

The [documentation home](https://github.com/kargibora/PrefScope/blob/main/docs/index.md) is organized by task:

- **[Tutorials](https://github.com/kargibora/PrefScope/blob/main/docs/index.md#tutorials):** learn the complete workflow.
- **[How-to guides](https://github.com/kargibora/PrefScope/blob/main/docs/index.md#how-to-guides):** complete a specific analysis.
- **[Explanation](https://github.com/kargibora/PrefScope/blob/main/docs/index.md#explanation):** understand presence, context, and design.
- **[Reference](https://github.com/kargibora/PrefScope/blob/main/docs/index.md#reference):** look up APIs, schemas, and CLI flags.
- **[Extending](https://github.com/kargibora/PrefScope/blob/main/docs/index.md#extending):** add backends and components.

The project is an alpha API. Production and experimental surfaces are listed in
[Project status](https://github.com/kargibora/PrefScope/blob/main/docs/reference/status.md). Release changes are recorded in
[CHANGELOG.md](https://github.com/kargibora/PrefScope/blob/main/CHANGELOG.md), and contributions are welcome through
[CONTRIBUTING.md](https://github.com/kargibora/PrefScope/blob/main/CONTRIBUTING.md).

## Scientific limits

Named sparse features and preference associations are descriptive. For research claims,
use prompt-grouped splits, human semantic audits, multiple SAE and interpretation seeds,
frozen-lens reuse on shifted data, matched baselines, and group-aware uncertainty. An
external replication dataset is still needed before treating a dataset-specific result
as general.
