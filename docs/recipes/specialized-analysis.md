# Specialized analysis recipes

PrefScope retains specialized analysis implementations under `prefscope.recipes` so that
researchers can inspect, copy, and adapt useful code without making it framework policy.

Current recipe groups include:

- `prefscope.recipes.analysis` — preference, paired outcomes, contexts, distributions,
  prompt regions, graphs, grouping, and SAE metrics;
- `prefscope.recipes.pipeline` — multi-step historical workflows built from those analyses;
- `prefscope.recipes.viewer_export` — historical table and map preparation functions.

Import the exact module explicitly:

```python
from prefscope.recipes.analysis.paired_outcomes import paired_outcome_shift
```

Recipes are not imported at the package root, registered, dispatched from configuration,
or covered by stable API guarantees. Read the function source and choose its parameters
as part of the study design. Do not assume a recipe's default threshold, statistical test,
label rule, or table layout is universally appropriate.
