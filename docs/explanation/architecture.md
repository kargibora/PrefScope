# Architecture

PrefScope runs a series of stages. Each stage produces an output that the next stage
can use. Expensive outputs are saved, so you do not need to repeat earlier work.

## The pipeline uses replaceable stages

Each stage has a default implementation. Developers can add another implementation by
registering a Python class under a name. A config or Python caller then selects that
name. This lets you add a dataset loader, naming method, verification method, or analysis
without changing unrelated code.

The stages and their plug points:

| Stage | Python interface | Defined in | What it does |
|-------|------------------|------------|--------------|
| Data | `Dataset` | `prefscope/core/dataset.py` | yields `PairItem(id, x, y_a, y_b?, pref?)` — one row each |
| Vector source | `RepresentationSource` | `prefscope/core/representation.py` | converts items to aligned fixed-width prompt/response embeddings or pooled activations |
| SAE input | `LensRep` | `prefscope/pipeline/lens_rep.py` | turns embeddings into SAE rows (`difference`, `individual`, `prompt`) |
| SAE | `SAE` | `prefscope/sae/model.py` | encodes a representation into sparse codes |
| Naming | `NameStrategy` | `prefscope/interpret/strategy.py` | proposes a concept name per axis |
| Verification | `VerifyStrategy` | `prefscope/interpret/strategy.py` | tests a name on held-out activations |
| Clustering | `Clusterer` | `prefscope/pipeline/cluster.py` | groups co-firing axes into behaviors |
| Task analysis | `AnalysisComponent` | `prefscope/api/analysis.py` | consumes aligned feature/outcome contracts and returns versioned estimand tables |

The public `analysis.py` and `loaded_lens.py` modules are compatibility facades. Their
implementations are split into contract, component, execution, projection, inspection,
and publication modules. This keeps old imports working while preventing one module or
the `Lens` class from owning every responsibility.

The Python analysis API is separate from the command-line workflow. An `AnalysisPlan`
combines built-in or user-defined analysis components. It checks row IDs, groups,
comparison direction, and outcome type before running. Each component returns its own
table. PrefScope does not combine behavior, ratings, preferences, and confounds into one
score.

Because the analysis side reads `PairItem`s rather than a file format, *bringing your
own data* is the same idea as any other swap: implement `Dataset.__iter__` to yield
`PairItem`s from wherever your data lives (see the bring-your-own-data how-to).
`RepresentationSource` is a separate seam: Qwen text embeddings, an embedding API,
and pooled residual activations can produce the same validated `RepresentationBatch`.
`LensRep` remains the built-in numerical policy for difference/individual/prompt lenses;
it is not a general third-party artifact plugin in the current manifest schema.

## Three durable artifacts

PrefScope produces three things, in order. Each is **reusable**: changing a late stage
never re-runs an early one, because the early output is cached on disk.

```
raw datasets ──build-corpus / prepare-dataset──▶ corpus.parquet
                                   │
                          embed + train SAE   (build-lens)
                                   ▼
                               a LENS dir      ◀── the durable artifact
                                   │   (frozen encoder f + z_*.npy codes + manifest.json)
                         interpret by concept  (prefscope run)
                                   ▼
            concept tables: names · fidelity · clusters · win-relevance
```

**Corpus.** A normalized parquet, one content-hash `battle_id` per row. Loading needs
only `prompt` and `completion_a`; `completion_b` marks a row as paired, `model_a/b` and
`human_pref` are optional and required only by the analyses that use them.
`build-corpus` emits the full battle schema and dedupes overlapping source dumps;
`prepare-dataset` maps an arbitrary local or Hub table into the same shape. Reusable
across every lens you train on it.

**Lens.** A saved sparse autoencoder trained on embeddings. A prompt lens scores
prompts. An individual lens scores single responses. A difference lens scores only an
A/B contrast. The manifest records which type you have, the embedding model, and the
array shapes. Build directories can also cache `z_diff`, `z_a`, `z_b`, or `z_prompt`.
Compact published lenses may omit these training codes. See [The lens](the-lens.md) for
the full file contract.

Embedding is normally the first part of `build-lens`. The separate `embed-corpus` and
`embed-prompts` commands exist for cache warming and large scheduled runs.

**Concept tables.** A standard `prefscope run` can write four main CSV files: names
(`feature_names.csv`), name checks (`feature_fidelity.csv`), feature groups
(`feature_clusters.csv`), and preference associations (`win_relevance.csv`). Other
analysis commands write their own separate tables.
Naming and fidelity are explained in [naming and fidelity](naming-and-fidelity.md);
the preference and per-model statistics are in
[the diagnosis math doc](diagnosis-math.md).

### Artifact integrity and provenance

PrefScope validates a complete output before replacing an older one. If publication
fails, it restores the older output. It also records content hashes and fixed Hub
revisions so a resumed run cannot silently use changed data.

In detail, `prepare-dataset` resolves Hugging Face branches or tags to commit SHAs and hashes
the ordered canonical retained table. Lens manifests add a row-order-sensitive
`dataset_hash` over canonical retained metadata plus canonical float32 source embeddings.
That hash binds what the SAE actually saw and stays stable across width sweeps built from
the same embedding dump.

Completion and prompt lens builders write into clean sibling staging directories. They
validate checkpoint/manifest agreement, declared shapes and alignment, the serialized
manifest, dataset hash, and an exact artifact allowlist before one-directory publication.
If publication fails, the prior destination is restored. Applied encoding and viewer-lens
materialization use the same clean staging/whole-directory contract and reject source/output
overlap. Publishers use per-destination locks and recover a sole interrupted-swap backup.
Downstream analysis state also
fingerprints resolved Hub commits, local lens contents, and configuration so resume cannot
silently accept a moving remote artifact.

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
  Python (`Lens.load`) and the CLI (`diagnose`) are real. `LoadedLens` remains only as
  a backward-compatible alias. For task-centered Python analysis, construct explicit
  `FeatureMatrix` and `OutcomeSpec` objects and call `analyze_dataset(...)`; custom
  `AnalysisComponent` instances compose with the built-in outcome association task.

For exact command flags, see the [CLI reference](../reference/cli.md). For practical
steps, return to the [how-to guides](../index.md#how-to-guides). To add a component, see
[Extending PrefScope](../extending/the-registry.md).
