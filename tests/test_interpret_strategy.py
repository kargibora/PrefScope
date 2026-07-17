"""Phase 1: interpreter strategies are real, registry-resolved components."""
import numpy as np
import pandas as pd
import pytest

from prefscope.core import registry
from prefscope.interpret.strategy import (
    IndividualNameStrategy, IndividualVerifyStrategy, LensCodes, NameStrategy,
    PairwiseNameStrategy, PromptVerifyStrategy, SingleTextNameStrategy, VerifyCodes,
    resolve_name_mode, resolve_verify_mode,
)


class FakeClient:
    def raw(self, messages, **kw):
        return '- "uses code blocks"'


def _battles(n=60):
    return pd.DataFrame({
        "instruction_id": [str(i) for i in range(n)],
        "prompt": [f"p{i}" for i in range(n)],
        "completion_a": [f"a{i}" for i in range(n)],
        "completion_b": [f"b{i}" for i in range(n)],
    })


def _z(n=60, m=3, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, m)).astype(np.float32)
    z[np.abs(z) < 0.3] = 0.0
    return z


def test_registered_and_resolvable():
    # importing the module registered the built-ins
    assert {"pairwise", "individual"} <= set(registry.available("interpreter"))
    assert isinstance(registry.make("interpreter", "individual"), IndividualNameStrategy)


def test_strategy_rejects_invalid_research_controls_up_front():
    with pytest.raises(ValueError, match="n_candidates"):
        IndividualNameStrategy(n_candidates=0)
    with pytest.raises(ValueError, match="sampling"):
        IndividualVerifyStrategy(sampling="random-ish")


def test_make_unknown_raises_valueerror_listing_available():
    with pytest.raises(ValueError, match="individual"):   # message lists available
        registry.make("interpreter", "nope")


def test_resolve_name_mode_uses_input_rep_not_file_probing():
    assert resolve_name_mode("auto", "individual") == "individual"
    assert resolve_name_mode("auto", "difference") == "pairwise"
    assert resolve_name_mode("pairwise", "individual") == "pairwise"   # explicit wins


def test_resolve_name_mode_prompt_lens_wins():
    assert resolve_name_mode("auto", "prompt", "prompt") == "single-text"
    assert resolve_name_mode("pairwise", "individual", "prompt") == "single-text"


def test_single_text_strategy_registered_and_guards_prompt_codes():
    assert isinstance(registry.make("interpreter", "single-text"), SingleTextNameStrategy)
    codes = LensCodes("d", "prompt", _battles(), None, None, None,
                      lens_kind="prompt", z_prompt=None, prompts=None)
    with pytest.raises(ValueError, match="prompt lens"):
        SingleTextNameStrategy().name(codes, FakeClient())


def test_individual_strategy_requires_z_a():
    codes = LensCodes("d", "individual", _battles(), _z(), z_a=None, z_b=None)
    with pytest.raises(ValueError, match="needs z_a"):
        IndividualNameStrategy().name(codes, FakeClient())


def test_end_to_end_dispatch_pairwise():
    # the path that matters: registry.make -> strategy.name -> the real namer
    codes = LensCodes("d", "difference", _battles(), _z(), z_a=None, z_b=None)
    strat = registry.make("interpreter", "pairwise", n_active=5, n_zero=5, verify_frac=0.2, seed=0)
    assert isinstance(strat, NameStrategy) and isinstance(strat, PairwiseNameStrategy)
    df = strat.name(codes, FakeClient())
    assert list(df["feature_id"]) == [0, 1, 2]
    assert (df["concept"] == "uses code blocks").all()


def test_end_to_end_dispatch_individual():
    codes = LensCodes("d", "individual", _battles(), _z(), z_a=_z(seed=1), z_b=_z(seed=2))
    df = registry.make("interpreter", "individual", n_active=5, n_zero=5).name(codes, FakeClient())
    assert "feature_id" in df.columns and "concept" in df.columns


def test_end_to_end_dispatch_individual_single_response():
    battles = _battles().drop(columns="completion_b")
    codes = LensCodes("d", "individual", battles, None, z_a=_z(seed=1), z_b=None)
    df = registry.make("interpreter", "individual", n_active=5, n_zero=5).name(
        codes, FakeClient())
    assert list(df["feature_id"]) == [0, 1, 2]


# --- verifier strategies ---------------------------------------------------------------

def _vcodes(**kw):
    base = dict(lens_kind="completion", input_rep="difference", battles=_battles(),
                instruction_ids=[str(i) for i in range(60)], z_diff=_z())
    base.update(kw)
    return VerifyCodes(**base)


def test_verifier_bucket_registered():
    assert {"pairwise", "individual", "prompt"} <= set(registry.available("verifier"))
    assert isinstance(registry.make("verifier", "individual"), IndividualVerifyStrategy)


def test_resolve_verify_mode():
    assert resolve_verify_mode("auto", "individual", "completion") == "individual"
    assert resolve_verify_mode("auto", "difference", "completion") == "pairwise"
    assert resolve_verify_mode("auto", "difference", "prompt") == "prompt"   # lens-kind wins
    assert resolve_verify_mode("pairwise", "individual", "completion") == "pairwise"


def test_individual_verify_requires_z_a():
    with pytest.raises(ValueError, match="needs z_a"):
        IndividualVerifyStrategy().verify(_vcodes(z_a=None, z_b=None), pd.DataFrame(), object())


def test_prompt_verify_requires_prompt_codes():
    with pytest.raises(ValueError, match="prompt"):
        PromptVerifyStrategy().verify(_vcodes(z_prompt=None, prompts=None), pd.DataFrame(), object())


def test_pairwise_verify_delegates_with_opts(monkeypatch):
    seen = {}

    def spy(battles, z_diff, names, client, **kw):
        seen.update(kw)
        seen["n_rows"] = len(battles)
        return pd.DataFrame({"fidelity_pass": [True]})

    monkeypatch.setattr("prefscope.interpret.verify.verify_features", spy)
    out = registry.make("verifier", "pairwise", n_per_bucket=7, seed=3).verify(
        _vcodes(), pd.DataFrame({"feature_id": [0]}), object())
    assert out["fidelity_pass"].iloc[0]
    assert seen["n_per_bucket"] == 7 and seen["seed"] == 3   # constructor opts forwarded


def test_individual_verify_forwards_negatives_and_embeddings(monkeypatch):
    """#6: the individual verifier must FORWARD the configured `negatives` strategy — it
    used to drop it, so the completion lens silently verified against random controls even
    under --negatives similar. It passes the SAE codes as the similarity space."""
    seen = {}

    def spy(texts, z, names, client, **kw):
        seen.update(kw)
        return pd.DataFrame({"fidelity_pass": [True]})

    monkeypatch.setattr("prefscope.interpret.verify.verify_single_text_features", spy)
    codes = _vcodes(z_a=_z(seed=1), z_b=_z(seed=2))
    # use "close" — the actual value the CLI/pipeline passes (the blocking bug was that
    # "close" wasn't a registered sampler, so this path crashed).
    registry.make("verifier", "individual", negatives="close").verify(
        codes, pd.DataFrame({"feature_id": [0]}), object())
    assert seen["negatives"] == "close"
    assert seen["embeddings"] is not None          # code-space similarity space forwarded


def test_individual_verify_random_negatives_passes_no_embeddings(monkeypatch):
    """With random negatives (the default) no similarity space is needed → embeddings None."""
    seen = {}

    def spy(texts, z, names, client, **kw):
        seen.update(kw)
        return pd.DataFrame({"fidelity_pass": [True]})

    monkeypatch.setattr("prefscope.interpret.verify.verify_single_text_features", spy)
    codes = _vcodes(z_a=_z(seed=1), z_b=_z(seed=2))
    registry.make("verifier", "individual").verify(  # default negatives="random"
        codes, pd.DataFrame({"feature_id": [0]}), object())
    assert seen["negatives"] == "random" and seen["embeddings"] is None


def test_individual_verify_single_response_alignment(monkeypatch):
    seen = {}

    def spy(texts, z, names, client, **kw):
        seen["texts"] = texts
        seen["shape"] = z.shape
        seen.update(kw)
        return pd.DataFrame({"fidelity_pass": [True]})

    monkeypatch.setattr("prefscope.interpret.verify.verify_single_text_features", spy)
    battles = _battles().drop(columns="completion_b")
    codes = _vcodes(battles=battles, z_a=_z(seed=1), z_b=None)
    IndividualVerifyStrategy().verify(codes, pd.DataFrame({"feature_id": [0]}), object())
    assert len(seen["texts"]) == 60 and seen["shape"] == (60, 3)
    assert len(seen["instruction_ids"]) == 60


def test_import_adapters_registers_interpreters_in_fresh_interpreter():
    # the documented entry point ("import prefscope.adapters") must populate the bucket,
    # not rely on the CLI happening to import strategy.py (the import-timing trap).
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "-c",
         "import prefscope.adapters; from prefscope.core import registry; "
         "assert {'pairwise','individual'} <= set(registry.available('interpreter'))"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
