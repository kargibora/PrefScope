"""Opt-in integration checks for real model and optional viewer dependencies."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def test_tiny_real_model_lens_roundtrip(tmp_path):
    """Download a tiny transformer, train a real lens, reload it, and encode text."""
    if os.environ.get("PREFSCOPE_RUN_MODEL_SMOKE") != "1":
        pytest.skip("set PREFSCOPE_RUN_MODEL_SMOKE=1 to allow model downloads")
    torch = pytest.importorskip("torch")
    from prefscope.api.config import SAEConfig, TrainConfig
    from prefscope.api.loaded_lens import Lens

    torch.set_num_threads(1)
    model_id = os.environ.get(
        "PREFSCOPE_SMOKE_MODEL_ID", "hf-internal-testing/tiny-random-bert"
    )
    prompts = [
        "Explain gravity.",
        "Write a greeting.",
        "What is two plus two?",
        "Name a primary color.",
    ] * 4
    completions = [
        "Gravity attracts masses.",
        "Hello and welcome!",
        "The answer is four.",
        "Red is a primary color.",
        "Objects fall toward Earth.",
        "Good morning!",
        "Two plus two equals 4.",
        "Blue is one example.",
        "Mass curves spacetime.",
        "Hi there.",
        "It is four.",
        "Yellow is also primary.",
        "Gravity is an attraction.",
        "Welcome, friend.",
        "4.",
        "Red.",
    ]
    data = pd.DataFrame(
        {
            "instruction_id": [f"smoke-{index}" for index in range(len(prompts))],
            "prompt": prompts,
            "completion_a": completions,
        }
    )
    config = TrainConfig(
        sae=SAEConfig(m=4, k=1, input_rep="individual", sae_type="simple-topk"),
        embed_model_id=model_id,
        val_frac=0.25,
        device="cpu",
        train_kwargs={
            "n_epochs": 1,
            "min_epochs": 1,
            "patience": 1,
            "batch": 4,
        },
    )

    lens = Lens.train(data, config=config, out=tmp_path / "tiny-lens")
    codes = lens.encode(["Explain gravity."], ["Masses attract each other."])

    assert codes.shape == (1, 4)
    assert np.isfinite(codes).all()
    assert np.count_nonzero(codes) == 1
    assert (tmp_path / "tiny-lens" / "manifest.json").is_file()


def test_viewer_extra_dependency_surface(tmp_path):
    """Check that the viewer extra provides its declared imports and test runner."""
    pytest.importorskip("plotly")
    umap = pytest.importorskip("umap")
    pytest.importorskip("streamlit")
    from plotly import graph_objects as go
    from streamlit.testing.v1 import AppTest

    app = tmp_path / "viewer_smoke.py"
    app.write_text(
        "import streamlit as st\n"
        "st.set_page_config(page_title='PrefScope smoke')\n"
        "st.title('PrefScope viewer smoke')\n",
        encoding="utf-8",
    )
    rendered = AppTest.from_file(str(app), default_timeout=15).run()

    assert not rendered.exception
    assert any("PrefScope viewer smoke" in str(title.value) for title in rendered.title)
    assert callable(umap.UMAP)
    assert "data" in go.Figure().to_plotly_json()
