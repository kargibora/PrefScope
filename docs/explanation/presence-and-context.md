# Numerical activity and semantic presence

A nonzero feature value is numerical activity. It is not automatically evidence that a
human-interpretable concept is present.

The supported numerical coactivation functions therefore require a boolean
`FeatureMatrix`. The caller must define and document how that boolean matrix was produced.
PrefScope preserves row IDs, feature IDs, and provenance, but does not select thresholds or
calibration policy.

```python
from prefscope import FeatureMatrix, coactivation_pairs

presence = FeatureMatrix(
    values=caller_defined_presence,
    row_ids=activity.row_ids,
    feature_ids=activity.feature_ids,
    role="presence",
    orientation="absolute",
    activation_polarity="boolean",
    code_semantics="semantic_presence",
    provenance={"presence_basis": ["defined by the study"]},
)
pairs = coactivation_pairs(presence)
```

Historical threshold and context-profiling implementations are retained under
`prefscope.recipes` for inspection and adaptation. They are not package-root functions,
CLI commands, or universal inference policy.
