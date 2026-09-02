"""Torch-free composition example with custom vectors, analyses, and paired outcomes.

This uses deterministic synthetic data. Replace ``precomputed_source`` with an embedding
or pooled-residual provider and replace ``LinearProjector`` with a loaded PrefScope lens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import prefscope
from prefscope.analysis.presence import PresenceMatrix


class LinearProjector:
    """Small duck-typed projector used only to keep this example self-contained."""

    activation_polarity = "signed"
    code_semantics = "axis"
    m_total = 2

    def project(self, values):
        matrix = np.asarray(values, dtype=float)
        return matrix[:, :2]


def precomputed_source(items):
    """Return aligned fixed-width response vectors from any user-controlled backend."""
    index = np.arange(len(items), dtype=float)
    response_a = np.column_stack([index / len(items), np.sin(index)])
    response_b = np.column_stack([(index + 1) / len(items), np.cos(index)])
    return prefscope.RepresentationBatch(
        row_ids=tuple(item.id for item in items),
        arrays={"response_a": response_a, "response_b": response_b},
        metadata={"group_id": tuple(item.meta["group_id"] for item in items)},
        provenance={"source_type": "synthetic-precomputed", "revision": "v1"},
    )


class FeatureMagnitude(prefscope.AnalysisComponent):
    """Example user plug-in: one descriptive row per feature set."""

    name = "feature_magnitude"

    def run(self, dataset):
        table = pd.DataFrame(
            [
                {
                    "feature_set": name,
                    "mean_l2": float(np.linalg.norm(matrix.values, axis=1).mean()),
                }
                for name, matrix in dataset.features.items()
            ]
        )
        return prefscope.AnalysisArtifact(
            name=self.name,
            table=table,
            estimand="row-weighted descriptive mean feature-vector norm",
            metadata={"inference": "none"},
        )


def run_example():
    row_ids = tuple(f"row-{index}" for index in range(20))
    groups = tuple(f"prompt-{index // 2}" for index in range(20))
    items = [
        prefscope.PairItem(
            id=row_id,
            x=f"prompt {group}",
            y_a="response before",
            y_b="response after",
            meta={"group_id": group},
        )
        for row_id, group in zip(row_ids, groups, strict=True)
    ]
    source = prefscope.CallableRepresentationSource(
        precomputed_source,
        name="precomputed",
        provenance={"provider": "user-function", "revision": "v1"},
    )
    lens = prefscope.Lens(LinearProjector(), representation_source=source)
    projected = lens.project_representations(source.encode(items))

    # Synthetic stand-in for a real confirmation-only semantic-presence artifact.
    prompt_values = np.array([[index // 2 < 5] for index in range(20)], dtype=bool)
    confirmed_prompt_presence = PresenceMatrix(
        values=prompt_values,
        feature_ids=np.array([0]),
        basis=np.array(["semantic_threshold"], dtype=object),
        thresholds=np.array([0.7]),
        calibrated=np.array([True]),
    )
    prompt_features = prefscope.FeatureMatrix.from_presence(
        confirmed_prompt_presence,
        row_ids=row_ids,
        role="prompt",
        metadata={"group_id": groups},
        provenance={"calibration_split": "confirmation"},
    )

    quality_before = np.full(20, 0.4)
    quality_after = np.array([0.8 if index // 2 < 5 else 0.3 for index in range(20)])
    plan = prefscope.AnalysisPlan(
        (
            prefscope.FeatureArtifactDiagnostics(),
            prefscope.OutcomeAssociations(feature_sets=("response_change",)),
            prefscope.PairedOutcomeShifts(),
            prefscope.PromptConditionedOutcomeShifts(prompt_features="prompt_concepts"),
            FeatureMagnitude(),
        )
    )
    return prefscope.analyze_dataset(
        {
            "response_change": projected.matrix("z_diff"),
            "prompt_concepts": prompt_features,
        },
        outcomes={
            "after_quality": prefscope.OutcomeSpec(
                quality_after,
                row_ids=row_ids,
                kind="probability",
            ),
        },
        paired_outcomes={
            "quality": prefscope.PairedOutcomeSpec(
                quality_before,
                quality_after,
                row_ids=row_ids,
                kind="probability",
                side_a="before",
                side_b="after",
                interpretation="higher values mean higher task quality",
            ),
        },
        group_ids=groups,
        plan=plan,
    )


if __name__ == "__main__":
    result = run_example()
    for name, artifact in result.artifacts.items():
        print(f"\n[{name}] {artifact.estimand}")
        print(artifact.table.head().to_string(index=False))
