# Lens representation policies

A lens representation (`LensRep`) defines the numerical relationship between
already computed A/B vectors and a frozen SAE. This is different from a
[representation source](add-a-representation-source.md), which produces the
vectors.

The supported artifact policies are:

| policy | training input | projected views |
|---|---|---|
| `difference` | `e_a - e_b` | `z_diff = f(e_a - e_b)` |
| `individual` | pooled `e_a`, `e_b` | `z_a`, `z_b`, `z_diff = f(e_a)-f(e_b)` |
| `prompt` | prompt vector | `z_prompt` |

For a nonlinear SAE,

```text
f(e_a) - f(e_b) != f(e_a - e_b)
```

so these policies cannot be interchanged after training.

`LensRep` is an internal, closed artifact policy. Native manifests and builders recognize
only `difference`, `individual`, and `prompt`. Defining a new Python class does not make a
new policy safely publishable or reloadable.

Use a custom `RepresentationSource` when replacing Qwen embeddings with another
fixed-width source. Adding a genuinely new lens representation requires a new
manifest capability contract and coordinated builder/loader changes; it is not a
one-class extension today.
