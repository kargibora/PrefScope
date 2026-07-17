# Glossary

Terms used across PrefScope, in one place.

- **Battle** — one prompt with two completions `A`, `B` from models `m_A`, `m_B`.
  The unit of the corpus.
- **Corpus** — a normalized parquet of battles; columns `battle_id, source,
  language, prompt, model_a, model_b, completion_a, completion_b` (+ optional
  `human_pref`). `battle_id` is a content hash, so duplicate battles dedup.
- **Lens** — a frozen SAE saved as a directory: the encoder plus the per-battle
  codes and `manifest.json`. The reusable artifact. See [the lens](../explanation/the-lens.md).
- **Code `z`** — a feature-activation vector. `z ∈ ℝ^M`, mostly zero (sparse). For a
  contrast it is **signed**: `sign(z_f)` says which side expresses concept `f` more.
- **`z_diff` / `z_a` / `z_b` / `z_prompt`** — the saved code arrays: contrast codes,
  per-side codes (individual lenses), and prompt codes (prompt lenses).
- **`M` / `K`** — dictionary size (number of features) and sparsity (active features
  per example); set at `build-lens` via `--m-total` / `--k`.
- **`input_rep`** — the lens representation (`difference`, `individual`, `prompt`),
  recorded in the manifest; drives which strategy `auto` mode picks. See
  [representations](../explanation/representations.md).
- **Lens kind** — `completion` (concepts in responses) or `prompt` (concepts in the
  question). A prompt lens runs `name → verify → cluster`.
- **Concept / feature** — one SAE axis. A **concept name** is its LLM label
  (`feature_names.csv`).
- **Fidelity** — whether a concept name survives a held-out falsification check;
  `fidelity_pass` in `feature_fidelity.csv`. See [naming and fidelity](../explanation/naming-and-fidelity.md).
- **Behavior / cluster** — a group of co-firing features (`feature_clusters.csv`);
  fights SAE over-granularity by merging near-duplicate concepts.
- **Orientation (self / other)** — in diagnosis, `y_a` is the model under study
  ("self") and `y_b` the comparison ("other"); codes are self-minus-other and
  `pref` = P(self preferred).
- **`net_direction`** — in a diagnosis, how much more (or less) a model expresses a
  concept than its peers. See [diagnosis math](../explanation/diagnosis-math.md).
- **`outcome_assoc` / `win_assoc`** — association between a concept's activation and a
  response being preferred (`win_relevance.csv`). Positive = doing more of it goes
  with winning.
- **Score gap** — the rubric/score difference between A and B in a comparison;
  rich-gap judges carry more graded signal.
- **SAE (sparse autoencoder)** — the model that produces the codes. The default is a
  BatchTopK Matryoshka SAE; the lens contract is just "signed sparse codes," so this
  is a default, not a requirement. See [the SAE](../explanation/sae.md).
- **Registry** — the `(kind, name) → class` map components are selected and registered
  through. See [the registry](../extending/the-registry.md).
