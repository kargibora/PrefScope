# Lens Config Schema

`Lens.from_config(...)` loads one backend-neutral lens from a strict YAML mapping.
Unknown keys fail closed. This file selects a lens; it does not configure an entire
managed `prefscope analyze` run.

## Published PrefScope lens

```yaml
version: 1
backend: prefscope
source: hf://owner/repository/completion
revision: <commit-or-ref>
device: cpu
# annotations: annotations/
# local_files_only: false
```

`source` can also be a local lens directory. Relative paths resolve from the YAML file.
Hub revisions are resolved by the ordinary PrefScope loader.

## Pretrained SAELens lens

```yaml
version: 1
backend: saelens
release: gpt2-small-res-jb
sae_id: blocks.8.hook_resid_pre
device: cpu
dtype: float32
sae_batch_size: 1024
text_batch_size: 8
max_output_bytes: 268435456
long_text_policy: truncate       # truncate | error
include_bos: false
allow_unregistered_release: false
```

The reader model, hook, preprocessing, context size, and sequence-position slice come
from checkpoint metadata. `long_text_policy=truncate` keeps the first declared context
window. Prompt and response views are independent documents. External SAE identifiers
remain unpinned.

## Registered custom backend

```yaml
version: 1
backend: my-backend
options:
  model: local-model
```

The trusted plug-in must be imported explicitly and register `my-backend` under the
`lens_backend` registry kind. An explicitly configured top-level `device` is passed to
the custom backend constructor; do not duplicate it in `options`. PrefScope never scans
installed packages.

A loaded lens supports:

```python
lens = Lens.from_config("lens.yaml")
features = lens.featurize(items, feature_ids=selected_ids)
```

See [Add a lens backend](../extending/add-a-lens-backend.md).
