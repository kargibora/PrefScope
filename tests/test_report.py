"""PURE-function tests for the `prefscope report` presentation layer
(format_report + prompt_concept_winrates). No embedding / GPU / LLM."""
import numpy as np
import pandas as pd

from prefscope.artifacts import BATTLES, Z_PROMPT
from prefscope.pipeline.report import (
    format_report,
    prompt_concept_winrates,
    prompt_to_response_winrates,
)


def _diag():
    # 4 named features + 1 unnamed. helps_win + net_direction present.
    return pd.DataFrame({
        "feature_id": [0, 1, 2, 3, 4],
        "concept": ["chatty", "code blocks", "refuses", "citations", ""],
        "fire_rate": [0.90, 0.05, 0.40, 0.30, 0.99],
        "net_direction": [0.5, -0.2, 0.3, -0.4, 0.1],
        # rewarded gap = under-expressed (net<0) AND rewarded (helps_win>0): feat 3.
        # feat 1 is under-expressed but NOT rewarded; feat 0 is rewarded but OVER-expressed.
        "helps_win": [0.20, -0.10, 0.05, 0.15, 0.50],
    })


def test_format_report_core_sections():
    md = format_report(_diag(), model="ModelX", n_battles=120, win_rate=0.42, top=3)
    assert "# ModelX — concept report card" in md
    assert "120 battles · win rate 42%" in md
    for h in ("## Frequently distinguishes from opponents",
              "## Rarely distinguishes from opponents", "## Rewarded gaps"):
        assert h in md
    # prompt-types section omitted without prompt_winrates
    assert "Strong / weak prompt types" not in md


def test_does_a_lot_orders_by_fire_desc_and_skips_unnamed():
    md = format_report(_diag(), model="M", n_battles=10, win_rate=0.5, top=3)
    does_a_lot = (md.split("## Frequently distinguishes from opponents")[1]
                    .split("## Rarely distinguishes from opponents")[0])
    # highest-fire NAMED concept first (unnamed feat 4 fires 0.99 but is skipped)
    first_bullet = [l for l in does_a_lot.splitlines() if l.startswith("- ")][0]
    assert first_bullet == "- chatty — differs from opponent in 90% of battles"
    assert "99% of battles" not in md  # unnamed feature excluded everywhere


def test_prompt_concept_winrates_drops_silent_rows(tmp_path):
    """A silent (all-zero) prompt code has no dominant concept — it must be DROPPED, not
    assigned to feature 0 by a bare argmax (#4)."""
    import numpy as np
    import pandas as pd
    from prefscope.pipeline.report import prompt_concept_winrates

    plens = tmp_path / "plens"
    plens.mkdir()
    ids = ["b0", "b1", "b2", "b3"]
    # b0,b1,b2 express concept 0 (+pole); b3 is silent (all-zero) -> excluded, not concept 0
    np.save(plens / "z_prompt.npy",
            np.array([[3., 0.], [2., 0.], [1., 0.], [0., 0.]], np.float32))
    pd.DataFrame({"battle_id": ids}).to_parquet(plens / "battles.parquet")
    win = np.array([1., 1., 0., 0.])
    agg = prompt_concept_winrates(str(plens), ids, win, min_battles=1)
    assert list(agg["prompt_concept"]) == [0]        # only concept 0; the silent b3 is gone
    assert int(agg.iloc[0]["n"]) == 3                # not 4
    np.testing.assert_allclose(agg.iloc[0]["win_rate"], 2 / 3)  # mean(1,1,0), b3 excluded


def test_rewarded_gaps_picks_under_and_rewarded_only():
    md = format_report(_diag(), model="M", n_battles=10, win_rate=0.5, top=10)
    gaps = md.split("## Rewarded gaps")[1]
    # feat 3 (citations): net<0 AND helps_win>0 -> included
    assert "citations — under-expressed, +0.15 Δwin (length-controlled)" in gaps
    # feat 1 (code blocks): under-expressed but helps_win<0 -> excluded
    assert "code blocks" not in gaps
    # feat 0 (chatty): rewarded but OVER-expressed (net>0) -> excluded
    assert "chatty" not in gaps


def test_rewarded_gaps_uses_delta_vs_pool_when_present():
    df = _diag()
    # citations now over-expressed vs pool -> dropped; refuses under vs pool + rewarded
    df["delta_vs_pool"] = [0.3, -0.1, -0.2, 0.4, 0.0]
    md = format_report(df, model="M", n_battles=10, win_rate=0.5, top=10)
    gaps = md.split("## Rewarded gaps")[1]
    assert "refuses" in gaps          # delta_vs_pool<0 AND helps_win>0
    assert "citations" not in gaps    # delta_vs_pool>0 even though net_direction<0


def test_format_report_relations_section():
    rel = pd.DataFrame({"prompt_concept": ["coding", "reasoning"],
                        "response_concept": ["code blocks", "refuses"],
                        "delta_win": [0.09, -0.06], "n": [120, 80]})
    md = format_report(_diag(), model="M", n_battles=10, win_rate=0.5, relations=rel)
    assert "## Prompt → Response" in md
    sec = md.split("## Prompt → Response")[1]
    # strongest |Δwin| first; signed delta + support shown
    assert "coding ⇒ code blocks — +0.09 Δwin (n=120)" in sec
    assert "reasoning ⇒ refuses — -0.06 Δwin (n=80)" in sec
    # section omitted when not supplied
    assert "Prompt → Response" not in format_report(_diag(), model="M",
                                                    n_battles=10, win_rate=0.5)


def _prompt_lens_one_concept(tmp_path, ids):
    plens = tmp_path / "plens"
    plens.mkdir()
    z = np.zeros((len(ids), 2), dtype=np.float32)
    z[:, 0] = 5.0  # every battle -> prompt concept 0
    np.save(plens / Z_PROMPT, z)
    pd.DataFrame({"battle_id": ids}).to_parquet(plens / BATTLES)
    return plens


def test_prompt_to_response_winrates_within_prompt_contrast(tmp_path):
    ids = [f"b{i}" for i in range(8)]
    plens = _prompt_lens_one_concept(tmp_path, ids)
    # response feature (col 0 -> feature id 7) fires on b0-b3, not on b4-b7;
    # the model wins exactly when it fires => within-prompt delta = +1.0
    codes = np.zeros((8, 1), dtype=np.float32)
    codes[:4, 0] = 2.0
    win = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
    out = prompt_to_response_winrates(plens, ids, codes, [7], win,
                                      response_names={7: "code blocks"},
                                      prompt_names={0: "coding"}, min_support=3)
    assert len(out) == 1
    r = out.iloc[0]
    assert r["prompt_concept"] == "coding"
    assert r["response_concept"] == "code blocks"
    np.testing.assert_allclose(r["delta_win"], 1.0)
    assert int(r["n"]) == 4


def test_prompt_to_response_winrates_min_support_drops_thin_edges(tmp_path):
    ids = [f"b{i}" for i in range(6)]
    plens = _prompt_lens_one_concept(tmp_path, ids)
    # feature fires on only 2 battles -> fired side below min_support=3 -> dropped
    codes = np.zeros((6, 1), dtype=np.float32)
    codes[:2, 0] = 2.0
    win = np.array([1, 1, 0, 0, 0, 0], dtype=float)
    out = prompt_to_response_winrates(plens, ids, codes, [7], win, min_support=3)
    assert out.empty
    assert list(out.columns) == ["prompt_concept", "response_concept", "delta_win", "n"]


def test_prompt_to_response_winrates_drops_silent_and_negative_prompt_rows(tmp_path):
    """A silent/all-negative prompt code has no dominant concept — a bare argmax would
    put it on feature 0 (silent) or the least-negative feature (negative pole) and forge a
    spurious prompt→response edge. Those rows must be dropped (positive-max gate)."""
    plens = tmp_path / "plens"
    plens.mkdir()
    ids = [f"b{i}" for i in range(10)]
    zp = np.zeros((10, 2), dtype=np.float32)
    zp[:6, 0] = 5.0                       # b0..b5 -> real +pole concept 0
    zp[6:, 0], zp[6:, 1] = -2.0, -1.0     # b6..b9 all-negative; bare argmax -> concept 1
    np.save(plens / Z_PROMPT, zp)
    pd.DataFrame({"battle_id": ids}).to_parquet(plens / BATTLES)
    codes = np.zeros((10, 1), dtype=np.float32)
    codes[[0, 1, 2, 6, 7], 0] = 2.0       # fires within both groups
    win = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0], dtype=float)
    out = prompt_to_response_winrates(plens, ids, codes, [7], win, min_support=2)
    # only the genuine concept-0 group survives; the negative rows never form concept 1
    assert set(out["prompt_concept"]) == {0}


def test_prompt_to_response_winrates_drops_battles_absent_from_prompt_lens(tmp_path):
    ids = [f"b{i}" for i in range(4)]
    plens = _prompt_lens_one_concept(tmp_path, ids)
    # ask about battles the prompt lens never saw -> empty, no crash
    codes = np.ones((2, 1), dtype=np.float32)
    out = prompt_to_response_winrates(plens, ["x0", "x1"], codes, [7],
                                      np.array([1.0, 0.0]), min_support=1)
    assert out.empty


def test_no_win_relevance_emits_hint():
    df = _diag().drop(columns=["helps_win"])
    md = format_report(df, model="M", n_battles=10, win_rate=0.5)
    assert "(pass --win-relevance to surface rewarded gaps)" in md


def test_outcome_assoc_lc_fallback_reward():
    df = _diag().drop(columns=["helps_win"])
    df["outcome_assoc_lc"] = [0.2, -0.1, 0.05, 0.15, 0.5]
    md = format_report(df, model="M", n_battles=10, win_rate=0.5, top=10)
    gaps = md.split("## Rewarded gaps")[1]
    assert "citations — under-expressed, +0.15 Δwin (length-controlled)" in gaps


def test_prompt_types_section_when_supplied():
    pw = pd.DataFrame({"prompt_concept": ["coding", "chitchat", "math"],
                       "win_rate": [0.7, 0.2, 0.5], "n": [40, 30, 25]})
    md = format_report(_diag(), model="M", n_battles=10, win_rate=0.5,
                       prompt_winrates=pw, top=2)
    assert "## Strong / weak prompt types" in md
    sec = md.split("## Strong / weak prompt types")[1]
    strong = sec.split("Weakest:")[0]
    weak = sec.split("Weakest:")[1]
    assert "coding — win rate 70% (n=40)" in strong
    assert "chitchat — win rate 20% (n=30)" in weak


def test_prompt_concept_winrates(tmp_path):
    plens = tmp_path / "plens"
    plens.mkdir()
    ids = ["b0", "b1", "b2", "b3", "b4", "b5"]
    # 2 prompt concepts; argmax picks concept 0 for b0-b2, concept 1 for b3-b5
    z = np.zeros((6, 2), dtype=np.float32)
    z[:3, 0] = 5.0
    z[3:, 1] = 5.0
    np.save(plens / Z_PROMPT, z)
    pd.DataFrame({"battle_id": ids}).to_parquet(plens / BATTLES)

    # model won b0,b1 (concept0 -> 2/3) and b3 only (concept1 -> 1/3); b2 lost
    win = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    out = prompt_concept_winrates(plens, ids, win, min_battles=3)
    out = out.set_index("prompt_concept")
    np.testing.assert_allclose(out.loc[0, "win_rate"], 2 / 3)
    np.testing.assert_allclose(out.loc[1, "win_rate"], 1 / 3)
    assert out.loc[0, "n"] == 3 and out.loc[1, "n"] == 3


def test_prompt_concept_winrates_min_battles_and_names(tmp_path):
    plens = tmp_path / "plens"
    plens.mkdir()
    ids = ["b0", "b1", "b2"]
    z = np.zeros((3, 2), dtype=np.float32)
    z[:2, 0] = 5.0   # concept 0: 2 battles
    z[2, 1] = 5.0    # concept 1: 1 battle (below min_battles)
    np.save(plens / Z_PROMPT, z)
    pd.DataFrame({"battle_id": ids}).to_parquet(plens / BATTLES)

    win = np.array([1.0, 0.0, 1.0])
    names = {0: "coding", 1: "math"}
    out = prompt_concept_winrates(plens, ids, win, prompt_names=names, min_battles=2)
    assert list(out["prompt_concept"]) == ["coding"]   # concept 1 filtered, id->name mapped
    np.testing.assert_allclose(out.iloc[0]["win_rate"], 0.5)


# --- CLI handler wiring (embedder/projector monkeypatched; no GPU) ---
import json

from prefscope import __main__ as cli


def test_cmd_report_writes_markdown_and_features_csv(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text(json.dumps({"input_rep": "difference"}))
    ann = {"per_sample": [
        {"instruction_id": "0", "model_a": "M", "model_b": "Y", "instruction": "p0",
         "completion_a": "a0", "completion_b": "b0", "judge_pref": 1.0},
        {"instruction_id": "1", "model_a": "Y", "model_b": "M", "instruction": "p1",
         "completion_a": "a1", "completion_b": "b1", "judge_pref": 0.0},
    ]}
    apath = tmp_path / "ann.json"
    apath.write_text(json.dumps(ann))

    def fake_run_diagnose(battles, model, embedder, projector, **kw):
        df = pd.DataFrame({"feature_id": [0, 1], "concept": ["chatty", "refuses"],
                           "fire_rate": [0.8, 0.2], "net_direction": [0.3, -0.4],
                           "helps_win": [0.1, 0.2]})
        return df, {"model": model, "n_battles": 2, "win_rate": 1.0, "n_features": 2}

    monkeypatch.setattr("prefscope.pipeline.diagnose.run_diagnose", fake_run_diagnose)
    monkeypatch.setattr(cli, "Embedder", lambda *a, **k: object())
    monkeypatch.setattr("prefscope.encode.sae.SAEProjector", lambda *a, **k: object())

    out_md = tmp_path / "report.md"
    rc = cli.main(["report", "--lens-dir", str(tmp_path), "--annotations", str(apath),
                   "--model", "M", "--out", str(out_md), "--device", "cpu"])
    assert rc == 0
    assert out_md.exists()
    assert "# M — concept report card" in out_md.read_text()
    assert (tmp_path / "report_features.csv").exists()
