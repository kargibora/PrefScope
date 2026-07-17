"""Synthetic tests for the oriented-code bank, Welch contrast, and validation.

No GPU / embeddings: a tiny fake projector and hand-built code matrices with a
planted signal, so each new statistic is checked against a known answer.
"""
import numpy as np
import pandas as pd

from prefscope.pipeline.diagnose import (diagnose_features, diagnose_from_bank)
from prefscope.pipeline.oriented_bank import (build_oriented_codes, save_bank,
                                              load_bank)
from prefscope.pipeline.validate import validate_diagnosis


class _IdProjector:
    """project(x) = x, so oriented codes equal the raw difference vectors."""
    def project(self, x):
        return np.asarray(x, dtype=np.float32)


def _battles(models_a, models_b, y, completions=None):
    df = pd.DataFrame({
        "instruction_id": [f"b{i}" for i in range(len(y))],
        "model_a": models_a, "model_b": models_b, "y_judge": y,
    })
    if completions is not None:
        df["completion_a"], df["completion_b"] = completions
    return df


def test_bank_length_orientation():
    # battle 0: a has 3 words, b has 1 -> a-as-self length +2, b-as-self -2
    e_a = np.zeros((1, 1), np.float32)
    e_b = np.zeros((1, 1), np.float32)
    battles = _battles(["X"], ["Y"], [1.0],
                       completions=(["w w w"], ["w"]))
    Z, meta = build_oriented_codes(e_a, e_b, battles, _IdProjector())
    assert "length" in meta.columns
    # row 0 = a-as-self (+2), row 1 = b-as-self (−2)
    np.testing.assert_allclose(meta["length"].to_numpy(), [2.0, -2.0])


def test_bank_length_zero_without_text():
    e_a = np.zeros((2, 1), np.float32)
    e_b = np.zeros((2, 1), np.float32)
    battles = _battles(["X", "Y"], ["Y", "X"], [1.0, 0.0])
    _, meta = build_oriented_codes(e_a, e_b, battles, _IdProjector())
    assert "length" in meta.columns
    np.testing.assert_allclose(meta["length"].to_numpy(), [0.0, 0.0, 0.0, 0.0])


# ---- diagnose_features Welch contrast -------------------------------------

def test_welch_contrast_columns_only_when_baseline():
    z = np.array([[1.0, -1.0], [1.0, 0.0]], dtype=np.float32)
    win = np.array([1.0, 0.0])
    base = diagnose_features(z, win)
    assert "delta_vs_pool" not in base.columns          # backward compatible
    withpool = diagnose_features(z, win, z_outside=np.zeros((4, 2), np.float32))
    for c in ("net_direction_pool", "delta_vs_pool", "welch_t", "welch_p",
              "welch_p_bonferroni", "cohens_d"):
        assert c in withpool.columns


def test_delta_vs_pool_equals_net_direction_gap():
    # inside mostly +1, pool mostly -1. A little variance on each side keeps Welch
    # well-defined (a perfectly-constant split has undefined variance -> nan p).
    inside = np.ones((20, 1), np.float32)
    inside[0, 0] = -1.0
    outside = -np.ones((50, 1), np.float32)
    outside[0, 0] = 1.0
    df = diagnose_features(inside, np.ones(20), z_outside=outside)
    r = df.iloc[0]
    assert r["net_direction"] > 0 and r["net_direction_pool"] < 0
    # the invariant this test asserts: delta_vs_pool == net_direction gap
    assert abs(r["delta_vs_pool"] - (r["net_direction"] - r["net_direction_pool"])) < 1e-6
    assert r["welch_p"] < 1e-6                            # clearly distinct


def test_diagnose_features_length_controlled_columns():
    # length present -> outcome_assoc_lc and length_confound added; raw kept
    rng = np.random.default_rng(0)
    n = 200
    z0 = rng.choice([-1.0, 1.0], size=n)
    length = rng.normal(size=n)
    win = (z0 > 0).astype(float)            # feature 0 helps win
    z = np.column_stack([z0, rng.choice([-1.0, 1.0], size=n)]).astype(np.float32)
    base = diagnose_features(z, win)
    assert "outcome_assoc_lc" not in base.columns      # back-compat: absent w/o length
    assert "outcome_assoc" in base.columns
    withlen = diagnose_features(z, win, length=length)
    for c in ("outcome_assoc", "outcome_assoc_lc", "length_confound"):
        assert c in withlen.columns
    # feature 0 (helps win) has a positive length-controlled AME
    assert withlen.iloc[0]["outcome_assoc_lc"] > 0


# ---- oriented-code bank ----------------------------------------------------

def test_bank_has_both_orientations_and_pool_baseline():
    e_a = np.array([[2.0], [0.0]], dtype=np.float32)
    e_b = np.array([[0.0], [1.0]], dtype=np.float32)
    battles = _battles(["X", "Y"], ["Y", "X"], [1.0, 0.0])
    Z, meta = build_oriented_codes(e_a, e_b, battles, _IdProjector())
    assert Z.shape == (4, 1)                              # 2 battles x 2 orientations
    # A-as-self codes are e_a - e_b; B-as-self are e_b - e_a (sign-flipped here
    # only because the projector is linear/identity)
    np.testing.assert_allclose(Z[:2, 0], [2.0, -1.0])
    np.testing.assert_allclose(Z[2:, 0], [-2.0, 1.0])
    # win is P(self preferred): A-rows = y, B-rows = 1 - y
    np.testing.assert_allclose(meta["win"].to_numpy(), [1.0, 0.0, 0.0, 1.0])
    assert list(meta["self_model"]) == ["X", "Y", "Y", "X"]
    assert set(meta["orientation"]) == {"a", "b"}


def test_bank_roundtrip(tmp_path):
    e_a = np.random.randn(5, 3).astype(np.float32)
    e_b = np.random.randn(5, 3).astype(np.float32)
    battles = _battles(["A"] * 5, ["B"] * 5, [1.0, 0.0, 0.5, 1.0, 0.0])
    Z, meta = build_oriented_codes(e_a, e_b, battles, _IdProjector())
    save_bank(tmp_path, Z, meta, lens_dir="lens", label_col="y_judge")
    Z2, meta2, man = load_bank(tmp_path)
    np.testing.assert_allclose(Z, Z2)
    assert man["n_battles"] == 5 and man["m_total"] == 3


def test_bank_rejects_label_out_of_range():
    import pytest
    e_a = np.zeros((2, 1), np.float32)
    e_b = np.zeros((2, 1), np.float32)
    battles = _battles(["A", "A"], ["B", "B"], [0.7, 1.0])   # 0.7 invalid
    with pytest.raises(ValueError):
        build_oriented_codes(e_a, e_b, battles, _IdProjector())


def test_diagnose_from_bank_picks_target_rows():
    # X over-expresses feature 0; the pool (Y, Z) does not
    e_a = np.array([[3.0], [3.0], [0.0], [0.0]], dtype=np.float32)
    e_b = np.zeros((4, 1), dtype=np.float32)
    battles = _battles(["X", "X", "Y", "Z"], ["Y", "Z", "Z", "Y"],
                       [1.0, 1.0, 0.5, 0.5])
    Z, meta = build_oriented_codes(e_a, e_b, battles, _IdProjector())
    df, summary = diagnose_from_bank(Z, meta, "X")
    assert summary["n_battles"] == 2 and summary["has_baseline"]
    r = df[df["feature_id"] == 0].iloc[0]
    assert r["net_direction"] > 0 and r["delta_vs_pool"] > 0


# ---- predictive validation -------------------------------------------------

def test_validate_recovers_planted_winrate_signal():
    # Construct 4 models whose net_direction on feature 0 is monotone in win rate.
    rng = np.random.default_rng(0)
    rows_Z, self_model, win, orient, other = [], [], [], [], []
    strengths = {"m0": -1.0, "m1": -0.3, "m2": 0.3, "m3": 1.0}
    for m, s in strengths.items():
        for _ in range(40):
            z0 = 1.0 if rng.random() < (s + 1) / 2 else -1.0   # P(+) rises with s
            rows_Z.append([z0, rng.choice([-1.0, 1.0])])       # feature 1 = noise
            self_model.append(m); orient.append("a"); other.append("pool")
            win.append(1.0 if rng.random() < (s + 1) / 2 else 0.0)
    Z = np.array(rows_Z, dtype=np.float32)
    meta = pd.DataFrame({"orientation": orient, "self_model": self_model,
                         "other_model": other, "win": win})
    wr = pd.DataFrame({"feature_id": [0, 1], "win_assoc": [1.0, 0.0],
                       "significant": [True, False]})
    df, summary = validate_diagnosis(Z, meta, wr, weight_col="win_assoc", min_battles=10)
    assert summary["n_models"] == 4
    # predicted deficit score should track actual win rate strongly & positively
    assert summary["insample_r"] > 0.8
    assert summary["insample_r2"] > 0.6
    # bootstrap CI + permutation null are reported and sane
    assert summary["insample_r2_ci_lo"] <= summary["insample_r2"] <= summary["insample_r2_ci_hi"]
    assert 0.0 < summary["insample_r2_perm_p"] <= 1.0


def test_validate_default_weight_is_delta_win_rate():
    import pytest
    rng = np.random.default_rng(0)
    rows_Z, self_model, win, orient, other = [], [], [], [], []
    strengths = {"m0": -1.0, "m1": -0.3, "m2": 0.3, "m3": 1.0}
    for m, s in strengths.items():
        for _ in range(40):
            z0 = 1.0 if rng.random() < (s + 1) / 2 else -1.0
            rows_Z.append([z0, rng.choice([-1.0, 1.0])])
            self_model.append(m); orient.append("a"); other.append("pool")
            win.append(1.0 if rng.random() < (s + 1) / 2 else 0.0)
    Z = np.array(rows_Z, dtype=np.float32)
    meta = pd.DataFrame({"orientation": orient, "self_model": self_model,
                         "other_model": other, "win": win})
    # default weight_col is now delta_win_rate; a frame lacking it must raise
    wr = pd.DataFrame({"feature_id": [0, 1], "win_assoc": [1.0, 0.0],
                       "significant": [True, False]})
    with pytest.raises(ValueError):
        validate_diagnosis(Z, meta, wr, min_battles=10)
    # supplying delta_win_rate works
    wr2 = wr.assign(delta_win_rate=[0.2, 0.0],
                    delta_win_significant=[True, False])
    df, summary = validate_diagnosis(Z, meta, wr2, min_battles=10)
    assert summary["weight_col"] == "delta_win_rate"
    assert summary["n_models"] == 4


def test_validate_loo_length_controlled_flag():
    rng = np.random.default_rng(1)
    rows_Z, self_model, win, orient, other, length = [], [], [], [], [], []
    strengths = {"m0": -1.0, "m1": -0.3, "m2": 0.3, "m3": 1.0}
    for m, s in strengths.items():
        for _ in range(40):
            z0 = 1.0 if rng.random() < (s + 1) / 2 else -1.0
            rows_Z.append([z0, rng.choice([-1.0, 1.0])])
            self_model.append(m); orient.append("a"); other.append("pool")
            win.append(1.0 if rng.random() < (s + 1) / 2 else 0.0)
            length.append(rng.normal())
    Z = np.array(rows_Z, dtype=np.float32)
    base = pd.DataFrame({"orientation": orient, "self_model": self_model,
                         "other_model": other, "win": win})
    wr = pd.DataFrame({"feature_id": [0, 1], "delta_win_rate": [0.2, 0.0],
                       "delta_win_significant": [True, False]})
    # no length column -> LOO falls back to raw, flag False
    _, s_nolen = validate_diagnosis(Z, base, wr, min_battles=10, loo=True)
    assert s_nolen["loo_length_controlled"] is False
    assert "loo_r2" in s_nolen
    # with a length column -> length-controlled LOO, flag True
    meta = base.assign(length=length)
    _, s_len = validate_diagnosis(Z, meta, wr, min_battles=10, loo=True)
    assert s_len["loo_length_controlled"] is True
