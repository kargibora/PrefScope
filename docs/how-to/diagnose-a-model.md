# Diagnose a model

Goal: given a trained, named lens, find out what a specific model **over- or
under-expresses** relative to its peers, and whether those tendencies help or hurt
its win rate. Two ways: the `LoadedLens` Python API (good for notebooks / custom
data) and the `diagnose` CLI (good for a full annotation set).

The orientation convention: `y_a` is the model under study ("self"), `y_b` is its
opponent ("other"); codes are self-minus-other and `pref` = P(self preferred).

## A. Python — `LoadedLens`

`LoadedLens` is the reusable inference artifact: load the lens once, then project
any dataset and analyze it on CPU.

```python
from prefscope.api.loaded_lens import LoadedLens
from prefscope.core import PairItem

lens = LoadedLens.from_dir("lenses/mylens", device="cpu")   # or device="cuda" for project()

# each item: the studied model's answer (y_a) vs a baseline (y_b)
data = [
    PairItem(id="q1", x="Explain entropy to a 10-year-old.",
             y_a="Entropy is how messy things get over time…",
             y_b="Entropy is the logarithm of the number of microstates…",
             pref=0.8, model_a="my-model", model_b="baseline"),
    # …more battles…
]

codes, meta = lens.project(data)            # embed → self-minus-other → sparse codes (N×M)
diag = lens.diagnose(codes, meta)           # per-feature over/under-expression DataFrame
pref = lens.evaluate_preference(codes, meta)  # how well the features explain the choice
```

`project()` runs the embedder (GPU recommended). If you already have `codes` (e.g.
saved from a previous run), the format-agnostic functions in `prefscope.analysis`
(`diagnose`, `feature_preference_relevance`, `evaluate_preference`) take any
`(codes, meta)` pair where `meta` has a `pref` column — no GPU needed.

Reading `diagnose`:
- **`net_direction` > 0** → the model does MORE of this concept than peers; **< 0** → a gap.
- **`outcome_assoc` > 0** → doing more of it goes with winning.

Pairing them separates strengths from weaknesses: *over-express + helps* = a
strength; *under-express + helps* = a gap worth closing.

If your lens has `feature_fidelity.csv`, pass `fidelity_only=True` to
`lens.diagnose(...)` to restrict the report to verified axes.

## B. CLI — `diagnose`

For a full OpenJury-style annotation file, the CLI projects + aggregates in one go:

```bash
prefscope diagnose \
    --lens-dir lenses/mylens \
    --annotations /path/to/annotations.json \
    --model my-model \                          # the target to orient as "self"
    --fidelity lenses/mylens/feature_fidelity.csv \   # attach names; restrict to verified
    --out diagnosis.csv \
    --device cuda
```

Writes a per-feature CSV (`net_direction`, `fire_rate`, `self_more/less_rate`,
`win_rate_self_more/less`, `outcome_assoc`) and prints the most over- and
under-expressed concepts. Add `--all-features` to include unverified axes.

View it in the quadrant tab of the Streamlit app:

```bash
uv run --extra viewer streamlit run prefscope/viewer/app.py -- \
    --lens-dir lenses/mylens --diagnosis diagnosis.csv
```

## C. Validate the diagnosis (optional)

Does the diagnosed deficit actually predict win rate across models? Build a pooled
oriented-code bank, then regress:

```bash
prefscope build-bank --lens-dir lenses/mylens \
    --from-embeddings emb/ --out bank/
prefscope win-relevance --lens-dir lenses/mylens \
    --corpus corpus.parquet --out win_relevance.csv
prefscope validate-diagnosis --bank bank/ \
    --win-relevance win_relevance.csv --out validation.csv --loo
```

`--loo` refits the reward weights leaving each model out — an honest held-out R².
