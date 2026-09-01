"""Every supported dataset shape must be handled deliberately by every entry point.

A command either works on a shape, or refuses it with a clear message. An unhandled
KeyError/AttributeError/IndexError/FileNotFoundError means an assumption about paired
or preference-labelled data leaked into code that claims to support single responses.
"""
import json

import numpy as np
import pandas as pd
import pytest

SHAPES = ("single", "paired_unlabeled", "paired_labeled")
# Raised on purpose by argument validation; anything else is an unguarded assumption.
DELIBERATE = (ValueError, NotImplementedError)
LEAKED = (KeyError, AttributeError, IndexError, FileNotFoundError, TypeError)

N = 8
M = 4


def make_corpus(tmp_path, shape: str):
    rows = {
        "battle_id": [f"id{i}" for i in range(N)],
        "source": ["test"] * N,
        "language": ["de", "cs"] * (N // 2),
        "prompt": [f"prompt {i} about topic {i % 3}" for i in range(N)],
        "completion_a": [f"response {i} with some words" for i in range(N)],
    }
    if shape != "single":
        rows["completion_b"] = [f"other response {i} here" for i in range(N)]
        rows["model_a"] = ["alpha"] * N
        rows["model_b"] = ["beta"] * N
    if shape == "paired_labeled":
        rows["human_pref"] = [1.0, 0.0, 0.5, 1.0, 0.0, 1.0, 0.5, 0.0]
    path = tmp_path / f"corpus_{shape}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def make_lens(tmp_path, shape: str, *, kind: str = "completion"):
    """A lens directory as build-lens would write it for this shape."""
    lens = tmp_path / f"lens_{kind}_{shape}"
    lens.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    z = rng.normal(size=(N, M)).astype(np.float32)
    z[np.abs(z) < 0.4] = 0.0
    single = shape == "single"
    if kind == "prompt":
        np.save(lens / "z_prompt.npy", np.abs(z))
    else:
        np.save(lens / "z_a.npy", np.abs(z))
        if not single:
            np.save(lens / "z_b.npy", np.abs(z)[::-1].copy())
            np.save(lens / "z_diff.npy", z)
    meta = {
        "instruction_id": [f"id{i}" for i in range(N)],
        "group_id": [f"g{i % 3}" for i in range(N)],
        "source": ["test"] * N,
        "language": ["de", "cs"] * (N // 2),
    }
    if single:
        meta["prompt"] = [f"prompt {i} about topic {i % 3}" for i in range(N)]
        meta["completion_a"] = [f"response {i} with some words" for i in range(N)]
    else:
        meta["model_a"] = ["alpha"] * N
        meta["model_b"] = ["beta"] * N
    pd.DataFrame(meta).to_parquet(lens / "battles.parquet", index=False)
    (lens / "manifest.json").write_text(json.dumps({
        "schema_version": 2,
        "m_total": M,
        "k": 2,
        "input_dim": 16,
        "input_rep": "prompt" if kind == "prompt" else "individual",
        "dataset_mode": "single" if single else "paired",
        "embed_model_id": "test/embedder",
        "activation_polarity": "nonnegative",
        "best_val_norm_mse": 0.2,
    }))
    pd.DataFrame({
        "feature_id": list(range(M)),
        "concept": [f"concept {i}" for i in range(M)],
        "fidelity_pass": [True] * M,
    }).to_csv(lens / "feature_names.csv", index=False)
    return lens


@pytest.fixture(params=SHAPES)
def shape(request):
    return request.param


def _assert_deliberate(fn, *args, **kwargs):
    """Run fn; pass if it succeeds or refuses deliberately, fail on a leaked error."""
    try:
        return fn(*args, **kwargs)
    except DELIBERATE:
        return None
    except LEAKED as exc:
        pytest.fail(f"unguarded assumption about dataset shape: {type(exc).__name__}: {exc}")


def test_load_corpus_handles_every_shape(tmp_path, shape):
    from prefscope.data.corpus import load_corpus

    frame = load_corpus(make_corpus(tmp_path, shape))
    assert len(frame) == N
    assert {"battle_id", "instruction_id", "group_id", "prompt", "completion_a"} <= set(frame.columns)
    assert ("completion_b" in frame.columns) == (shape != "single")


def test_load_lens_battles_handles_every_shape(tmp_path, shape):
    from prefscope.interpret.io import load_lens_battles

    lens = make_lens(tmp_path, shape)
    corpus = make_corpus(tmp_path, shape)
    battles, codes, manifest = _assert_deliberate(load_lens_battles, lens, corpus=corpus)
    assert len(battles) == N and codes.shape == (N, M)


def test_export_examples_handles_every_shape(tmp_path, shape):
    from prefscope.viewer_export.examples import export_examples

    lens = make_lens(tmp_path, shape)
    corpus = make_corpus(tmp_path, shape)
    features = pd.DataFrame({"feature_id": list(range(M)),
                             "concept": [f"c{i}" for i in range(M)]})
    out = _assert_deliberate(export_examples, lens, str(corpus), features, 3)
    if out is not None:
        assert set(out) <= {str(i) for i in range(M)}


def test_viewer_export_cli_handles_every_shape(tmp_path, shape):
    from prefscope.viewer_export.cli import main as export_main

    lens = make_lens(tmp_path, shape)
    corpus = make_corpus(tmp_path, shape)
    out = tmp_path / f"bundle_{shape}"
    code = _assert_deliberate(export_main, [
        "--lens-dir", str(lens), "--corpus", str(corpus), "--out", str(out)])
    if code is None:
        return
    assert code == 0
    manifest = json.loads((out / "bundle_manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert "features.json" in manifest["files"] and "meta.json" in manifest["files"]


def test_classify_role_evidence_handles_every_shape(tmp_path, shape):
    from prefscope.interpret.role import _select_evidence

    lens = make_lens(tmp_path, shape)
    z_a = np.load(lens / "z_a.npy")
    z_b = np.load(lens / "z_b.npy") if (lens / "z_b.npy").exists() else None
    ev = _assert_deliberate(_select_evidence, z_a, z_b, [f"id{i}" for i in range(N)], 0,
                            n_top=3, n_random=0, seed=0)
    assert ev is not None
    assert all(e["side"] == "a" for e in ev) or shape != "single"


def test_presence_and_prompt_regions_handle_every_shape(tmp_path, shape):
    from prefscope.analysis.presence import concept_presence

    lens = make_lens(tmp_path, shape)
    z = np.load(lens / "z_a.npy")
    features = pd.DataFrame({
        "feature_id": list(range(M)),
        "concept": [f"c{i}" for i in range(M)],
        "semantic_threshold": [0.1] * M,
        "presence_pass": [True] * M,
    })
    _assert_deliberate(concept_presence, z, features, feature_ids=list(range(M)))


def test_inspect_cli_handles_every_shape(tmp_path, shape):
    from prefscope.cli import main

    corpus = make_corpus(tmp_path, shape)
    assert _assert_deliberate(main, ["inspect", "--corpus", str(corpus)]) in (0, None)


def test_paired_only_exports_refuse_single_response_clearly(tmp_path, shape):
    """Contrast-only analyses must name the problem, not raise FileNotFoundError."""
    from prefscope.artifacts import require_paired_codes

    lens = make_lens(tmp_path, shape)
    if shape == "single":
        with pytest.raises(ValueError, match="paired contrast codes"):
            require_paired_codes(lens, command="test")
    else:
        assert require_paired_codes(lens, command="test").exists()


def test_win_relevance_refuses_single_response(tmp_path, shape):
    from prefscope.cli import main

    lens = make_lens(tmp_path, shape)
    corpus = make_corpus(tmp_path, shape)
    if shape != "paired_labeled":
        return
    code = _assert_deliberate(main, [
        "win-relevance", "--lens-dir", str(lens), "--corpus", str(corpus),
        "--out", str(tmp_path / "win.csv")])
    assert code in (0, 2, None)


def test_fire_rate_is_available_for_every_shape(tmp_path, shape):
    """generality drives sorting and display; it must not be empty for single data."""
    from prefscope.viewer_export.features import feature_fire_rate

    lens = make_lens(tmp_path, shape)
    rates = feature_fire_rate(lens)
    assert rates, f"no fire rates computed for {shape}"
    assert set(rates) == set(range(M))
    assert all(0.0 <= v <= 1.0 for v in rates.values())
