# Getting started

For the unreleased `0.3` API, use a source checkout. The published PyPI package may
have a different API.

```bash
git clone https://github.com/kargibora/PrefScope.git
cd PrefScope
uv sync
source .venv/bin/activate  # macOS/Linux
python -c "import prefscope; print(prefscope.__version__)"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. The base package
imports without PyTorch.

Create a demo dataset:

```bash
prefscope init-demo --out demo
```

The demo contains a corpus, not a trained lens. To apply an existing native lens, install
PyTorch with `uv sync --extra cpu`. Keep any other needed extras in that same command.
Replace `path/to/lens` below with your lens directory, then featurize one pair:

```python
from prefscope import Lens, PairItem, activation_summary

lens = Lens.load("path/to/lens")
pair = PairItem(id="demo", x="prompt", y_a="answer A", y_b="answer B")
batch = lens.featurize([pair], views=("response_a", "response_b"))
print(activation_summary(batch.matrix("z_a")))
```

The summary is numerical activity. It is not a semantic or causal conclusion. See
[Python API](../reference/python-api.md) for the supported objects and functions.
