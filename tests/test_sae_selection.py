import pandas as pd
import pytest

from prefscope.analysis.sae_selection import (
    evaluate_sweep, expansion_ratio, recommend_config,
)


def _row(m, k, fvu, dead=0.01, cos=0.2, l0=None):
    return {"m_total": m, "k": k, "fvu": fvu, "dead_frac": dead,
            "decoder_cos_mean_max": cos, "l0_mean": k if l0 is None else l0}


def test_best_reconstruction_among_admissible_wins():
    rec = recommend_config([_row(256, 16, 0.30), _row(512, 32, 0.18), _row(1024, 32, 0.12)])
    assert (rec["m_total"], rec["k"]) == (1024, 32)
    assert rec["admissible"] and rec["n_admissible"] == 3


def test_dead_features_reject_a_width():
    rec = recommend_config([_row(256, 16, 0.30), _row(2048, 32, 0.10, dead=0.2)])
    assert rec["m_total"] == 256
    assert rec["n_admissible"] == 1


def test_duplicated_directions_reject_a_width():
    frame = evaluate_sweep([_row(512, 32, 0.10, cos=0.8), _row(256, 16, 0.25)])
    rejected = frame[frame["m_total"] == 512].iloc[0]
    assert not rejected["admissible"]
    assert "duplicated directions" in rejected["rejected_because"]


def test_rows_per_feature_bounds_width_for_document_embeddings():
    """110k documents cannot support a 16k-feature dictionary."""
    frame = evaluate_sweep([_row(16384, 32, 0.05), _row(2048, 32, 0.15)], n_rows=110_000)
    wide = frame[frame["m_total"] == 16384].iloc[0]
    assert not wide["admissible"]
    assert "memorisation risk" in wide["rejected_because"]
    assert frame[frame["m_total"] == 2048].iloc[0]["admissible"]


def test_realised_sparsity_far_below_target_is_rejected():
    frame = evaluate_sweep([_row(512, 64, 0.10, l0=20.0)])
    assert not frame.iloc[0]["admissible"]
    assert "far below k" in frame.iloc[0]["rejected_because"]


def test_falls_back_to_least_bad_and_says_so():
    rec = recommend_config([_row(512, 32, 0.10, dead=0.5), _row(256, 16, 0.20, dead=0.4)])
    assert not rec["admissible"]
    assert rec["n_admissible"] == 0
    assert rec["rejected_because"]


def test_missing_columns_and_empty_sweeps_are_refused():
    with pytest.raises(ValueError, match="missing"):
        evaluate_sweep([{"m_total": 512, "k": 32}])
    with pytest.raises(ValueError, match="empty"):
        evaluate_sweep(pd.DataFrame(columns=["m_total", "k", "fvu", "dead_frac", "l0_mean"]))


def test_expansion_ratio_flags_the_undercomplete_regime():
    assert expansion_ratio(512, 4096) == pytest.approx(0.125)
    assert expansion_ratio(16384, 4096) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        expansion_ratio(512, 0)
