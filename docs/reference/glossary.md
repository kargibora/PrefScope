# Glossary

Short meanings for terms used across PrefScope.

- **Activation** — the numerical value of one feature for one item. A nonzero activation
  does not by itself prove that the feature name is present.
- **Artifact** — a saved output that another step can load. Examples are a prepared
  dataset, a lens directory, and a concept table.
- **Battle** — one prompt with two responses, A and B. The responses may come from two
  models or two versions of one model.
- **BH correction** — the Benjamini–Hochberg method. It adjusts p-values when many tests
  are run together. The output is a q-value.
- **Code (`z`)** — the vector of feature activations produced by a lens. Most values are
  zero, so the code is sparse.
- **Concept name** — an LLM-generated label for one feature. The label is a claim that
  must be checked on separate examples.
- **Corpus** — a normalized table of prompts and responses. A paired corpus can also
  contain models and a preference for A.
- **Embedding** — a fixed-width numerical vector that represents a prompt or response.
- **Estimand** — the exact quantity an analysis tries to estimate. For example, it may
  be the average B-minus-A change with every prompt group weighted equally.
- **Feature** — one learned direction in a lens. A feature may receive a concept name.
- **Fidelity** — whether a concept name works on held-out examples with high activation
  and does not also match the controls. The result is stored as `fidelity_pass`.
- **Group ID** — the ID of an independent unit, usually a prompt. Rows with the same
  group ID share one total weight in grouped analysis.
- **Inference** — a statistical statement about uncertainty, such as a confidence
  interval or p-value. Some PrefScope outputs are descriptive and do not support one.
- **`input_rep`** — how a lens was trained: `difference`, `individual`, or `prompt`.
- **Lens** — a saved feature encoder plus a manifest. A build directory may also contain
  training codes and aligned metadata. A compact published lens does not need them.
- **Manifest** — `manifest.json` inside a lens or output bundle. It records the model,
  shapes, representation type, and other facts needed to load the artifact safely.
- **`M` / `K`** — `M` is the number of features. `K` is the target number of active
  features per item.
- **NMI** — normalized mutual information. PrefScope uses it to summarize how strongly
  response-feature presence depends on prompt context.
- **OOD (out of distribution)** — an input that differs from the data on which a model
  or lens was trained. A difference lens is OOD when given one response by itself.
- **Orientation** — the direction of a comparison, such as A-minus-B or B-minus-A.
- **Pole** — one direction of a signed feature. A stored concept name describes the
  positive pole unless the artifact says otherwise.
- **Presence** — see **semantic presence**. Raw activity and semantic presence are not
  the same claim.
- **Provenance** — portable information about where an artifact came from, such as a
  model revision, dataset revision, or content hash. It must not contain credentials.
- **Response scope** — whether evidence points to a general tendency, a context-specific
  tendency, prompt content, or insufficient evidence.
- **Response tendency** — a prompt-matched difference in semantic concept presence
  between two response sets. It is not automatically a general model behavior.
- **SAE (sparse autoencoder)** — the model that learns lens features from embeddings.
- **Semantic presence** — a feature is above a calibrated threshold where its name has
  enough confirmed precision. Raw `z != 0` is only numerical activity.
- **Table contract** — a versioned declaration of a result table's required columns,
  logical types, unique key, direction, and units.
- **Tie** — a preference value of `0.5`. Many descriptive preference tables retain it
  as neutral; binary logistic analysis drops it and says so.
- **`win_assoc`** — a dataset-specific association between a signed response feature and
  preference. Positive does not mean universally good, and the association is not causal.
- **`z_a` / `z_b`** — feature codes for response A and response B from an individual lens.
- **`z_diff`** — a contrast code. Its orientation is recorded in the artifact.
- **`z_prompt`** — feature codes from a prompt lens.

See [The lens](../explanation/the-lens.md),
[Representations](../explanation/representations.md), and
[Semantic presence and context](../explanation/presence-and-context.md) for more detail.
