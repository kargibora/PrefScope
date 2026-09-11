# Naming and fidelity

Feature values and feature annotations are separate evidence layers.

`FeatureCatalog` can attach a proposed `name`, `description`, and related display fields to
an explicit `feature_id`. A label does not change the numerical coordinate and does not by
itself show that the feature detects the named concept.

Naming and verification tools may help researchers propose and inspect annotations, but
their outputs remain study artifacts. The supported numerical analysis functions do not:

- infer names from English text;
- merge features based on label similarity;
- treat a passed label as causal evidence;
- filter activity based on a universal fidelity threshold.

Keep annotations tied to the same feature-space identity as the activation matrix. Use
`feature_activation_table(...)` for an explicit `feature_id` join. Review proposed labels
and fidelity evidence before publication.

Specialized labeling, calibration, and context routines are retained as recipe modules.
They are not part of the stable numerical analysis API.
