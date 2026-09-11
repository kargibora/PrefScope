# Use a SAELens checkpoint

Install the optional integration:

```bash
pip install "prefscope[saelens]"
```

Create a normal `Lens` wrapper around a registered SAELens release:

```python
from prefscope import Lens, PairItem

lens = Lens.from_saelens(
    "release-name",
    "sae-id",
    input_rep="individual",
    device="cpu",
)
items = [PairItem(id="1", x="prompt", y_a="answer A", y_b="answer B")]
batch = lens.featurize(items, views=("response_a", "response_b"))
```

The base PrefScope import remains Torch-free. SAELens and Torch load only when this
integration is used.

The wrapper records checkpoint and coordinate identity when the upstream release exposes
enough information. It also preserves the backend's declared activation polarity and
code semantics. These declarations describe numerical coordinates; they do not establish
feature meaning.

Run analysis with the same direct functions used for any other lens:

```python
from prefscope import activation_summary, top_activating_rows

matrix = batch.matrix("z_a")
activity = activation_summary(matrix)
examples = top_activating_rows(matrix, k=10)
```

For specialized analyses, explicitly inspect and import the relevant module below
`prefscope.recipes`. PrefScope does not automatically choose an outcome model or report
schema for SAELens features.

Use `allow_unregistered_release=True` only when you intentionally accept an upstream
release outside PrefScope's tested registry. Pin upstream revisions for reproducible
research.
