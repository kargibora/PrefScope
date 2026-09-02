# Lens Config Schema

`Lens.from_config(...)` loads one backend-neutral lens from a strict YAML mapping.
Unknown keys fail closed. This file selects exactly one lens; it does not configure an
entire managed `prefscope analyze` run, a dataset, or several feature spaces.

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

`source` can also be a local native lens directory. Relative paths resolve from the YAML
file. Hub revisions are resolved by the ordinary PrefScope loader. A native prompt lens
and a native individual/completion lens are different feature spaces. Apply them in
separate `Lens.from_config(...)` calls or separate small featurization examples;
do not combine their arrays into one `FeatureBatch`.

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
`lens_backend` registry kind before `Lens.from_config(...)` runs. The repository example
accepts repeatable global `--plugin MODULE` options for this purpose. Import only code
you trust: plug-in registration executes ordinary Python imports. An explicitly configured top-level
`device` is passed to the custom backend constructor; do not duplicate it in `options`.
PrefScope never scans installed packages.

## Capabilities and views

A loaded lens supports:

```python
lens = Lens.from_config("lens.yaml")
print(lens.capabilities.views)
features = lens.featurize(items, views=("response_a",), feature_ids=selected_ids)
```

Request only views declared by `lens.capabilities`. Canonical view names map to saved
arrays as follows:

| requested view | `FeatureBatch` / bundle array |
|---|---|
| `prompt` | `z_prompt` |
| `response_a` | `z_a` |
| `response_b` | `z_b` |
| `response_difference` | `z_diff` |

The `lens.capabilities.difference` field declares how `response_difference` is formed.
`a_minus_b_after_encoding` means `f(e_A) - f(e_B)`. A
`direct_difference_projection` means `f(e_A - e_B)` and does not provide separate
response scores. These operations are not interchangeable.

See [Add a lens backend](../extending/add-a-lens-backend.md) and the
[Python API](python-api.md).
