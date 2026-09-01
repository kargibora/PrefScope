import json

import numpy as np
import pandas as pd

from prefscope.viewer_export.features import export_features, export_meta
from prefscope.viewer_export.tables import (
    export_conditional,
    export_delta,
    export_prompt_features,
)


def _lens(tmp_path):
    lens = tmp_path / "lens"
    lens.mkdir()
    (lens / "manifest.json").write_text(json.dumps({
        "input_rep": "individual",
        "m_total": 3,
        "output_arrays": ["z_a", "z_b", "z_diff"],
    }))
    return lens


def test_viewer_meta_counts_persisted_flags_strictly(tmp_path):
    lens = _lens(tmp_path)
    features = pd.DataFrame({
        "feature_id": [0, 1, 2],
        "fidelity_pass": ["True", "False", np.nan],
    })

    meta = export_meta(lens, validation=None, features=features)

    assert meta["n_verified"] == 1


def test_viewer_feature_export_handles_an_uninterpreted_lens(tmp_path):
    lens = _lens(tmp_path)

    features = export_features(lens)

    assert features["feature_id"].tolist() == [0, 1, 2]


def test_viewer_feature_export_reads_external_analysis_and_roles(tmp_path):
    lens = _lens(tmp_path)
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    pd.DataFrame({
        "feature_id": [0, 1],
        "concept": ["uses headings", "discusses finance"],
    }).to_csv(analysis / "feature_names.csv", index=False)
    pd.DataFrame({
        "feature_id": [0, 1],
        "concept": ["uses headings", "discusses finance"],
        "fidelity_pass": [True, False],
        "n": [60, 60],
    }).to_csv(analysis / "feature_fidelity.csv", index=False)
    pd.DataFrame({
        "feature_id": [0],
        "classification_status": ["ok"],
        "semantic_role": ["presentation"],
        "semantic_family": ["behavioral"],
        "behavior_scope": ["context_conditional_behavior"],
        "role_confidence": ["high"],
        "label_coverage": [1.0],
    }).to_csv(analysis / "feature_roles.csv", index=False)

    features = export_features(lens, analysis)

    first = features.set_index("feature_id").loc[0]
    assert first["concept"] == "uses headings"
    assert bool(first["fidelity_pass"]) is True
    assert first["fidelity_n"] == 60
    assert first["semantic_role"] == "presentation"
    assert first["behavior_category"] == "context_specific"
    assert first["label_coverage"] == 1.0


def test_viewer_relation_exports_do_not_promote_false_or_missing_flags(tmp_path):
    features = pd.DataFrame({
        "feature_id": [0],
        "concept": ["example"],
        "fidelity_pass": ["False"],
    })
    bias = tmp_path / "bias.csv"
    pd.DataFrame({
        "feature_id": [0], "confound_entangled": ["False"],
    }).to_csv(bias, index=False)
    delta = tmp_path / "delta.csv"
    pd.DataFrame({
        "prompt_concept": [0, 1],
        "completion_feature": [0, 0],
        "delta": [0.2, 0.1],
        "p_bonferroni": [0.01, 0.01],
        "stable": ["False", np.nan],
    }).to_csv(delta, index=False)

    exported = export_delta(delta, features, bias)

    assert exported["n_significant"] == 0
    assert all(not cell["stable"] for cell in exported["cells"])
    assert exported["completion_features"][0]["fidelity_pass"] is False
    assert exported["completion_features"][0]["confound_entangled"] is False

    conditional = tmp_path / "conditional.csv"
    pd.DataFrame({
        "prompt_concept": [0, 1],
        "feature_id": [0, 0],
        "delta_win_rate": [0.2, 0.1],
        "cond_significant": ["False", np.nan],
    }).to_csv(conditional, index=False)
    exported_conditional = export_conditional(conditional, features)
    assert exported_conditional["n_significant"] == 0
    assert all(not cell["sig"] for cell in exported_conditional["cells"])


def test_prompt_feature_export_keeps_named_axes_without_fidelity_rows(tmp_path):
    pd.DataFrame({
        "feature_id": [0, 1, 2],
        "concept": ["translation", "mathematics", "code"],
        "status": ["ok", "ok", "insufficient_evidence"],
    }).to_csv(tmp_path / "prompt_feature_names.csv", index=False)
    pd.DataFrame({
        "feature_id": [1],
        "fidelity_pass": [True],
        "agreement": [0.9],
    }).to_csv(tmp_path / "prompt_feature_fidelity.csv", index=False)

    exported = export_prompt_features(tmp_path)

    assert [row["feature_id"] for row in exported["features"]] == [0, 1, 2]
    by_id = {row["feature_id"]: row for row in exported["features"]}
    assert by_id[1]["fidelity_pass"] is True
    assert by_id[0]["fidelity_pass"] is None
