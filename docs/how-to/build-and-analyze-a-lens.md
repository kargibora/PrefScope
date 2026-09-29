# Build and analyze a lens

Start with a question you can check against paired examples: *does one feature appear more strongly in the preferred answer?* The commands below run from a source checkout after `uv sync`. The first example uses **hand-written synthetic activations**, not a trained lens or a research result. It runs without PyTorch, downloads, or a published lens.

Save this as `investigate.py`, then run `uv run python investigate.py` from the repository root:

```python
from pathlib import Path

import numpy as np
import pandas as pd

from prefscope import (
    FeatureBatch, FeatureCatalog, PairItem, Report,
    feature_activation_table, pair_item_metadata,
)

facts = [
    ("Why is the sky blue?", "Short wavelengths scatter more strongly.", "The ocean paints it."),
    ("What is binary search?", "It halves a sorted search range.", "It checks every item."),
    ("Why does ice float?", "Solid water is less dense.", "Ice has no mass."),
]
items, a_codes, b_codes = [], [], []
for i, (prompt, accurate, inaccurate) in enumerate(facts):
    # Reversing A/B tests side assignment, but does not add an independent prompt.
    items.extend([
        PairItem(f"{i}-a", prompt, accurate, inaccurate, pref=1.0, meta={"group_id": str(i)}),
        PairItem(f"{i}-b", prompt, inaccurate, accurate, pref=0.0, meta={"group_id": str(i)}),
    ])
    high = (2.0, 1.5, 1.2)[i]
    a_codes.extend([[high], [0.1]])
    b_codes.extend([[0.1], [high]])

batch = FeatureBatch(
    row_ids=tuple(item.id for item in items),
    arrays={"z_a": np.array(a_codes), "z_b": np.array(b_codes)},
    roles={"z_a": "response", "z_b": "response"},
    orientations={"z_a": "side_a", "z_b": "side_b"},
    feature_ids=(7,), metadata=pair_item_metadata(items),
    code_semantics="hand_written_demo", activation_polarity="nonnegative",
)
# A name is an annotation, not a finding about what this feature detects.
catalog = FeatureCatalog.from_mapping({7: "candidate factual-answer pattern"}, column="name")
rows = pd.DataFrame({
    "row_id": batch.row_ids, "group_id": batch.metadata["group_id"],
    "prompt": batch.metadata["prompt"],
    "response_a": batch.metadata["response_a"],
    "response_b": batch.metadata["response_b"],
    "pref": batch.metadata["pref"],
    "a": batch.array("z_a")[:, 0], "b": batch.array("z_b")[:, 0],
})
# Symptom: inspect high-A rows with their matched B answers, not just a mean.
examples = (feature_activation_table(batch.matrix("z_a"), catalog=catalog)
            .sort_values("abs_activation", ascending=False).head(3)
            .merge(rows, on="row_id", validate="many_to_one"))
# Exact-pair comparison controls for the prompt; group the reversed rows once.
rows["preferred_minus_other"] = np.where(rows.pref == 1.0, rows.a - rows.b, rows.b - rows.a)
by_prompt = (rows.groupby("group_id", as_index=False)
             .agg(n_rows=("row_id", "size"), mean_difference=("preferred_minus_other", "mean")))
print(examples[["row_id", "name", "activation", "response_a", "response_b"]].to_string(index=False))
print(by_prompt.to_string(index=False))
Report(
    title="Synthetic paired feature check",
    metrics={"n_rows": len(rows), "n_independent_prompts": len(by_prompt)},
    tables={"examples": examples, "matched_rows": rows, "by_prompt": by_prompt},
    metadata={"feature_label_status": "proposed, unverified", "data": "hand-written synthetic"},
).save(Path("example-output/investigation/report"))
```

Use a **new output directory** on each run; `Report.save` does not replace an existing report. The two reversed rows for each prompt are not independent evidence. Here all three prompt-level differences are positive *by construction*. A real study needs more independent prompts, review of high and low examples, and checks for length, topic, model identity, or other plausible alternatives. This paired contrast is descriptive; it cannot establish a causal feature, a general preference effect, or a meaningful label. A preference probability other than 0 or 1 (or a tie) needs a study-specific outcome rule instead of the binary `np.where` above.

## Apply the same check to an actual lens

`prefscope init-demo --out demo` creates a corpus, **not** a lens. To train an example lens, see [`examples/training/train_completion_lens.py`](../../examples/training/train_completion_lens.py); it downloads an embedding model and needs optional PyTorch (`uv sync --extra cpu`). Alternatively use your own compatible lens directory. [`examples/inference/local_dataset.py`](../../examples/inference/local_dataset.py) shows how to map source columns with `TableDataset` and save a `FeatureBatch`. Replace the synthetic `batch` above with `Lens.load("path/to/lens").featurize(items, views=("response_a", "response_b"))`, then choose an actual `feature_id` and join annotations with `lens.feature_catalog` or another catalog bound to that feature space. Check the lens representation and view semantics: a difference-only or prompt lens cannot supply two separate response-side codes. Supply your own pair IDs, prompt-group key and outcome orientation (`pref` means P(A preferred)); do not assume reversed rows or labels exist in your data.

Keep numerical activations separate from proposed names. Read the examples and naming/verification evidence before calling a feature a concept; nonzero activity is not calibrated semantic presence. For statistical tests, uncertainty intervals, outcome adjustment, matched sampling and presence thresholds, write or adapt **study-specific code**; [`prefscope.recipes`](../recipes/specialized-analysis.md) is optional, not a stable automatic analysis pipeline. `Report` only stores the tables and metrics you supply. If you need an interactive share, [`export_viewer_bundle`](../reference/viewer-bundle.md) packages an existing `FeatureBatch` with an already-built compatible Viewer; it does not run this analysis or build the Viewer. FeatureBatch and Viewer export can retain raw prompt and answer text: check permission and redact before publishing. The synthetic rows here are safe to share, but that does not establish permission for your own corpus.
