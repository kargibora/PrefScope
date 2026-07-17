# Architecture

This is the conceptual spine of PrefScope: a pipeline of swappable stages that
produce three durable artifacts. Understand this and every command and config option
falls into place.

## The pipeline is stages over a registry seam

PrefScope is not a monolith. It is a sequence of stages, each defined by a small
abstract base class in `prefscope.core`, with concrete implementations registered
under string names. You select an implementation *by name* — in a `prefscope run`
config, in a `PipelineConfig`, or by constructing the registered class in Python. The
registry is the seam that makes the framework pluggable: a new dataset, a new SAE
input representation, a new naming or verification strategy is one class plus a
`@registry.register(...)` line, not edits scattered across the codebase.

The stages and their plug points:

| Stage | Plug point (ABC) | What it does |
|-------|------------------|--------------|
| Data | `Dataset` | yields `PairItem(id, x, y_a, y_b?, pref?)` — one battle each |
| Activations | `ActivationSource` | turns a response into a vector (or pooled per-token activations) |
| SAE input | `Representation` / `LensRep` | combines the A/B sides into SAE rows |
| Diagnosis | `Interpreter` / `Verifier` / `Clusterer` | name axes, filter them, group them |

Because the analysis side reads `PairItem`s rather than a file format, *bringing your
own data* is the same idea as any other swap: implement `Dataset.__iter__` to yield
`PairItem`s from wherever your data lives (see the bring-your-own-data how-to).

## Three durable artifacts

PrefScope produces three things, in order. Each is **reusable**: changing a late stage
never re-runs an early one, because the early output is cached on disk.

```
raw datasets ──build-corpus──▶ corpus.parquet
                                   │
                          embed + train SAE   (build-lens)
                                   ▼
                               a LENS dir      ◀── the durable artifact
                                   │   (frozen encoder f + z_*.npy codes + manifest.json)
                         interpret by concept  (prefscope run)
                                   ▼
            concept tables: names · fidelity · clusters · win-relevance
```

**Corpus.** A normalized parquet of battles (`prompt`, `model_a/b`,
`completion_a/b`, optional `human_pref`), one content-hash `battle_id` per row.
Built once; deduped so overlapping source dumps collapse to one row. Reusable across
every lens you train on it.

**Lens.** A frozen sparse autoencoder trained over embeddings of the corpus, saved as
a directory. This is *the* durable artifact — the rest of the framework loads it and
runs on CPU. Its encoder turns any response into a sparse signed code; the directory
caches those codes (`z_diff` / `z_a` / `z_b`, or `z_prompt` for a prompt lens) and a
`manifest.json` recording the embed model, dims, and `input_rep`. What a lens *is* as
a contract is the subject of [the lens explanation](the-lens.md). Crucially,
embedding is **not** a separate command you run for its own sake — it is the cached
front half of `build-lens`.

**Concept tables.** Four CSVs, the analysis payoff: what each concept means
(`feature_names.csv`), which survive verification (`feature_fidelity.csv`), how they
group (`feature_clusters.csv`), and which ones are rewarded (`win_relevance.csv`).
Naming and fidelity are explained in [naming and fidelity](naming-and-fidelity.md);
the preference and per-model statistics are in
[the diagnosis math doc](diagnosis-math.md).

## Why the layering pays off

The artifacts are cached at the expensive boundaries. Embedding (GPU, slow) and SAE
training are paid once per lens; with `--dump-embeddings` you can even re-fit
different dictionary sizes without re-embedding. The whole analysis chain — naming,
verification, clustering, preference relevance, diagnosis — runs on the cached code
matrices with no GPU and no re-embedding. So iterating on a clusterer, a verifier, or
a diagnosis is cheap, and the heavy stages stay frozen behind their artifacts.

## Build vs. analyze vs. use

Three ways you interact with a lens, with a clean split today:

- **Build** a completion lens (embed + train the SAE): use the CLI (`build-lens`) or
  Python (`Lens.train(data, config=..., out=...)`). Prompt-lens building currently
  uses the CLI (`build-prompt-lens`).
- **Analyze** a lens by concept (name / verify / cluster / win-relevance): both the
  CLI (`prefscope run --config`) and Python (`run_pipeline(PipelineConfig.from_dict(...))`)
  are real.
- **Use** a trained lens for inference (project, diagnose, predict preference): both
  Python (`LoadedLens.from_dir`) and the CLI (`diagnose`) are real.

For the exact flags of each command, see the how-to guides under `docs/how-to/`; for
adding a new registered component, see `docs/extending/`.
