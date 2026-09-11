# Add a dataset adapter

Most local tables need no plug-in. Use `TableDataset` with explicit column names or
normalize the table before constructing `PairItem` values.

For a reusable source, implement the lightweight `Dataset` iteration contract and yield
validated `PairItem` objects:

```python
from prefscope import Dataset, PairItem

class MyDataset(Dataset):
    def __iter__(self):
        yield PairItem(id="1", x="prompt", y_a="A", y_b="B")
```

Keep source-specific downloading and authentication in the adapter. Do not make a dataset
adapter run feature extraction or analysis. Callers pass the resulting items to
`Lens.featurize(...)` explicitly.

Add offline tests with local fixtures. Default tests must not require network access.
