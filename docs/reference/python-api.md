# Python API Reference

PrefScope is usable as a library. The core object is a **`Lens`** — a frozen SAE
encoder + interpreted concept names + manifest. Its lifecycle is
`train → save → load → encode → analyze`.

Top-level imports (`import prefscope`):

```python
from prefscope import (
    Lens, LoadedLens, load_lens,          # the lens object + loaders
    PairItem, Dataset,                    # data contracts
    SAEConfig, TrainConfig,               # training configuration
    diagnose, evaluate_preference, feature_preference_relevance,  # analyses
    registry,                             # plug-in registry
)
```

`import prefscope` is **torch-free**: the heavy `Embedder` / `SAEProjector` /
`build_lens` imports happen lazily inside `Lens` methods, so importing the package
never pulls in torch. `LoadedLens` is a back-compat alias for `Lens`.

Sources of truth: `prefscope/api/loaded_lens.py`, `prefscope/api/config.py`,
`prefscope/analysis/__init__.py`, `prefscope/encode/sae.py`,
`prefscope/pipeline/run.py`.

---

## The lens object — `prefscope.Lens`

Wraps a built lens directory (SAE projector + embedder + optional
`feature_names.csv`) as a reusable inference artifact. Convention: `y_a` = "self"
(model under study), `y_b` = "other"; pair codes are signed self-minus-other;
`pref` = P(self preferred).

### Loading

```python
Lens.load(lens_dir, *, device: str = "cpu") -> Lens     # alias: Lens.from_dir
load_lens(lens_dir, *, device: str = "cpu") -> Lens     # module-level shorthand
```

Builds the real torch `SAEProjector` + `Embedder` (embedder model id taken from
the manifest's `embed_model_id`), loading `feature_names.csv` if present, and
records the backing directory as `lens.lens_dir`. `Lens.from_dir` is kept as an
alias of `Lens.load`. `load`, `load_lens`, and `TrainConfig` default to `"cpu"` so
the simplest API path is safe on a laptop; opt into `"cuda"` or `"mps"` explicitly.

### `encode` / `encode_one` — per-response codes

```python
lens.encode(prompts, completions=None) -> np.ndarray   # (N, M)
lens.encode_one(prompt, completion=None) -> np.ndarray # (M,)
```

Concept codes for individual responses. For an **individual** lens, both prompt
and completion are embedded; for a **prompt** lens, the prompt alone is embedded
(completions ignored). A single `str` is accepted for either argument (wrapped to
length 1; `encode` still returns a 2-D array, `encode_one` returns 1-D). A
**difference** lens is contrast-only and raises `ValueError` — use `encode_pairs`
instead.

```python
lens = load_lens("lenses/indiv_8b")
codes = lens.encode(["Write a poem", "Explain gravity"],
                    ["Roses are red…", "Mass curves spacetime…"])   # (2, M)
one = lens.encode_one("Write a poem", "Roses are red…")             # (M,)
```

### `concept_names` / `top_concepts` — naming

```python
lens.concept_names                          # pd.Series feature_id -> name, or None
lens.top_concepts(codes, k=5) -> list[list[tuple[str, float]]]
```

`concept_names` is `None` unless the lens carries a named `feature_names.csv`.
`top_concepts` returns, per row, the `k` **named** features with the largest
`|code|` as `(concept, signed_value)` pairs sorted by `|value|` descending;
unnamed features are skipped.

```python
codes = lens.encode(prompts, completions)
for row in lens.top_concepts(codes, k=3):
    print(row)   # e.g. [('verbosity', 2.1), ('refusal', -1.4), ('code blocks', 0.9)]
```

### `encode_pairs` — paired contrast codes

```python
lens.encode_pairs(dataset, *, return_meta=True) -> (codes (N, M), meta)  # alias: project
```

Iterates the dataset (`PairItem`-like objects with `.x`, `.y_a`, `.y_b`, `.id`,
`.pref`, `.model_a`, `.model_b`), embeds both responses, forms the lens's contrast
(per `input_rep`), and projects through the SAE to signed self-minus-other codes.
`meta` carries `id`, `pref`, `model_a`, `model_b`. `return_meta=False` returns just
the codes. It raises on single-response items or a token-granularity lens.
`lens.project` is kept as an alias and still returns the
`(codes, meta)` tuple.

### `encode_items` — paired or single-response datasets

```python
lens.encode_items(dataset, *, return_meta=True) -> (codes, meta)
```

Accepts a homogeneous iterable: paired items delegate to `encode_pairs`; items with
`y_b=None` produce absolute per-response codes through an **individual** lens. Mixed
paired/single input and single input on a difference lens raise clearly. Preference
analyses below require the paired contrast form.

### Analyses on pair codes

```python
lens.diagnose(codes, meta, *, fidelity_only=False) -> pd.DataFrame
lens.feature_preference_relevance(codes, meta) -> pd.DataFrame
lens.evaluate_preference(codes, meta, **kwargs) -> dict
```

`diagnose` gives per-feature over/under-expression + outcome association (sorted by
`net_direction`, names attached, optionally restricted to fidelity-passing
features). `feature_preference_relevance` gives per-feature univariate preference
relevance. `evaluate_preference` returns a cross-validated logistic readout dict
(`n`, `accuracy`, `auc`, `baseline_accuracy`, `n_features`, `top_features`); kwargs
forward to `analysis.evaluate_preference` (`n_splits=5`, `seed=0`). All three
delegate to `prefscope.analysis`.

```python
codes, meta = lens.encode_pairs(my_dataset)
diag  = lens.diagnose(codes, meta, fidelity_only=True)
rel   = lens.feature_preference_relevance(codes, meta)
score = lens.evaluate_preference(codes, meta)   # {'accuracy': ..., 'auc': ..., ...}
```

### `save`

```python
lens.save(dest) -> Path
```

Copies the backing lens directory to `dest` (recursive, `dirs_exist_ok`). Raises
`ValueError` if the lens has no backing directory (constructed directly rather than
loaded). No-op when `dest` equals the backing directory.

---

## Training a lens — `Lens.train`

```python
Lens.train(data, config=TrainConfig(), *, out, columns=None) -> Lens
```

Trains and saves a fresh lens, then loads it. `data` is normalized to a battles
DataFrame by `pairs_to_battles` (below), embedded, and passed to `build_lens`; the
result directory at `out` is loaded into a `Lens`.

### Configuration — `SAEConfig` / `TrainConfig`

```python
@dataclass
class SAEConfig:            # architecture — defines the frozen lens
    m: int = 128
    k: int = 16
    input_rep: str = "individual"     # "individual" | "difference" | "prompt"
    matryoshka_prefix: tuple = (8,)

@dataclass
class TrainConfig:          # run-time
    sae: SAEConfig = SAEConfig()
    embed_model_id: str | None = None # None -> embedder/config default
    val_frac: float = 0.1
    device: str = "cpu"
    max_train_rows: int | None = None
    train_kwargs: dict = {}           # n_epochs/lr/sparsity_coef -> build_lens **train_kwargs
```

`SAEConfig` is the part that defines what the lens *is* (width, sparsity, input
representation). `TrainConfig` adds the run-time knobs and nests an `SAEConfig` in
`sae`; entries in `train_kwargs` forward to `build_lens` (e.g. `n_epochs`, `lr`,
`sparsity_coef`).

### `pairs_to_battles` — data normalization

```python
pairs_to_battles(data, columns=None) -> pd.DataFrame
```

Pure (no torch, no embedding). Accepts:

- a `Dataset` / iterable of `PairItem` — mapped
  `x→prompt`, `y_a→completion_a`, `y_b→completion_b`, `id→instruction_id`, plus
  `pref→human_pref`, `model_a`, `model_b` when present;
- a `pd.DataFrame` — the `columns` rename map is applied first, then the three
  required columns (`prompt`, `completion_a`, `instruction_id`) are validated;
  `completion_b` is optional for homogeneous single-response data;
- a `str` / `Path` — read as parquet, then treated as a DataFrame.

Raises `ValueError` if any required column is missing.

---

## Two flows

### 1. Use a trained lens on (prompt, completion) lists

```python
from prefscope import load_lens

lens = load_lens("lenses/indiv_8b", device="cpu")
codes = lens.encode(prompts, completions)          # (N, M)
for row in lens.top_concepts(codes, k=5):
    print(row)
```

### 2. Train from a custom `Dataset`

```python
from prefscope import Lens, Dataset, PairItem, TrainConfig, SAEConfig

class MyData(Dataset):
    def __iter__(self):
        yield PairItem(id="1", x="Write a haiku", y_a="…", y_b="…",
                       pref=1.0, model_a="A", model_b="B")
        # …

cfg = TrainConfig(sae=SAEConfig(m=128, k=16, input_rep="individual"),
                  device="cuda", train_kwargs={"n_epochs": 20})
lens = Lens.train(MyData(), cfg, out="lenses/my_lens")
codes, meta = lens.encode_pairs(MyData())
score = lens.evaluate_preference(codes, meta)
lens.save("releases/my_lens_v1")
```

For instruction-tuning rows with one output each, omit `y_b`, keep
`input_rep="individual"`, then call `lens.encode_items(data)`. The saved artifact has
`dataset_mode: single`, `z_a.npy`, and enough text in `battles.parquet` to run
`name → verify → cluster` without a separate corpus file.

`Lens.train` also accepts a DataFrame or a parquet path directly:

```python
lens = Lens.train("battles.parquet", TrainConfig(), out="lenses/from_parquet")
lens = Lens.train(df, TrainConfig(), out="lenses/from_df",
                  columns={"q": "prompt", "a": "completion_a",
                           "b": "completion_b", "iid": "instruction_id"})
```

---

## Format-agnostic analyses — `prefscope.analysis`

Operate on `(codes, meta)` or raw `z` matrices — independent of how the codes were
produced. `codes` are signed self-minus-other SAE codes `(N, M)`; `meta` must
carry a `pref` column = P(self preferred). `diagnose`, `evaluate_preference`, and
`feature_preference_relevance` are re-exported at the top level (`prefscope`).

Re-exported from `prefscope.analysis` (`__init__.py`):

| function | returns | summary |
|----------|---------|---------|
| `diagnose(codes, meta, *, names=None, fidelity_only=False)` | DataFrame | per-feature over/under-expression + outcome assoc, sorted by `net_direction` |
| `feature_preference_relevance(codes, meta, *, names=None)` | DataFrame | per-feature univariate preference relevance |
| `evaluate_preference(codes, meta, *, n_splits=5, seed=0, names=None)` | dict | CV logistic readout (accuracy/auc/top_features) |
| `inside_outside_contrast(inside, outside)` | dict | Welch two-sample contrast (mean/delta/welch_t/welch_p/cohens_d) |
| `dataset_reward(z)` | ndarray | per-feature reward summary over a dataset |
| `split_half_stable(z, effect_fn, *, seed=0)` | DataFrame | split-half stability of a feature effect |
| `spurious_share(z, undesirable, *, eps=1e-9)` | ndarray | share of activity attributable to an undesirable surrogate |
| `label_inconsistency(z, reward, undesirable)` | ndarray | per-feature label-inconsistency signal |
| `diagnose_dataset(z, undesirable, *, ids=None, names=None, seed=0)` | — | dataset-level diagnosis bundle |
| `symmetric_activity(z_a, z_b)` | ndarray | per-feature A/B activity symmetry |
| `region_behavior_contrast(z, cluster_ids, *, seed=0)` | DataFrame | per-cluster region/behavior contrast |
| `feature_confound_correlation(z, surrogate)` | DataFrame | per-feature correlation with a confound surrogate |
| `auto_undesirable(z, surrogate, *, threshold=0.3)` | list | feature ids auto-flagged undesirable by surrogate correlation |

```python
import numpy as np, pandas as pd
from prefscope import diagnose, evaluate_preference

codes = np.load("lenses/indiv_8b/z_diff.npy")    # (N, M)
meta = pd.DataFrame({"pref": ...})               # P(self preferred) per row
diag = diagnose(codes, meta)                     # per-feature tendencies
score = evaluate_preference(codes, meta)         # CV readout dict
```

---

## Plug-in registry — `prefscope.registry`

Strategies (interpreters, verifiers, clusterers) register under a `kind`. List
the names registered for a kind, then build one by name:

```python
from prefscope import registry

registry.available("interpreter")            # -> list of registered names
obj = registry.make("interpreter", name, ...)  # construct by name (kwargs forwarded)
```

---

## Config pipeline — `prefscope.pipeline.run`

### `PipelineConfig`
Typed, validated view of a pipeline config (see `config-schema.md`).

```python
PipelineConfig.from_dict(d: dict) -> PipelineConfig   # validate an in-memory mapping
PipelineConfig.load(path) -> PipelineConfig           # load .yaml/.yml/.json then from_dict
```

Fields: `lens_dir`, `out_dir`, `stages`, `corpus`, `annotations`, `lens_kind`,
`llm` (`LLMConfig`), `interpreter`/`verifier`/`clusterer` (`StageConfig`),
`win_relevance` (dict). Returns the dataclass; raises `ValueError` on a bad/unknown
key.

### `run_pipeline`
```python
run_pipeline(cfg: PipelineConfig, *, client=None, verbose: bool = True) -> dict
```
Executes `cfg.stages` in canonical order, threading artifacts under `out_dir`.
Returns `{stage_name: output_Path}`. `client` overrides the LLM client (tests
inject a fake); otherwise the config's `llm` block builds one lazily on the first
LLM stage. `preflight(cfg)` (also public) fails fast on a missing lens/corpus.

```python
from prefscope.pipeline.run import PipelineConfig, run_pipeline

cfg = PipelineConfig.load("examples/pipeline.yaml")
outputs = run_pipeline(cfg)            # runs name -> verify -> cluster -> win-relevance
print(outputs["win-relevance"])        # Path to win_relevance.csv under out_dir
```

---

## Raw SAE projection — `prefscope.encode.sae.SAEProjector`

Frozen BatchTopK SAE projector. Accepts a path to `sae_model.pt` or the lens dir
containing it; auto-applies the lens's `whiten.npz` if present.

```python
SAEProjector(model_path, device: str = "cpu")
  .project(x: np.ndarray (N, D)) -> np.ndarray (N, M)     # sparse signed codes
  .reconstruct(z: np.ndarray (N, M)) -> np.ndarray (N, D) # back to embedding space
  .residual_norm(x: np.ndarray (N, D)) -> np.ndarray (N,) # ||x - recon|| per row
```
Attributes: `m_total` (M), `input_dim` (D), `config`, `device`. `project` raises
`ValueError` if the input dim != `input_dim` (embedder/lens mismatch).

```python
import numpy as np
from prefscope.encode.sae import SAEProjector

proj = SAEProjector("lenses/indiv_8b", device="cpu")
e = np.load("emb/e_a.npy").astype(np.float32)   # (N, D) embeddings
z = proj.project(e)                              # (N, M) sparse codes
resid = proj.residual_norm(e)                    # off-dictionary signal per row
```
