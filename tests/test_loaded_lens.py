import json as _json

import numpy as np
import pandas as pd
import pytest
import torch

from prefscope.api.loaded_lens import Lens
from prefscope.core.types import PairItem


def _names():
    return pd.DataFrame({"feature_id": [0, 1, 2], "concept": ["a", "b", "c"],
                         "fidelity_pass": [True, False, True]})


def _write_synthetic_lens(tmp_path, m=3, d=4):
    """A minimal sae_model.pt matching SAEProjector's expected state_dict keys."""
    sd = {
        "encoder.weight": torch.zeros(m, d),
        "input_bias": torch.zeros(d),
        "neuron_bias": torch.zeros(m),
        "threshold": torch.tensor(0.1),
        "decoder.weight": torch.zeros(d, m),
    }
    torch.save({"state_dict": sd, "config": {"m_total": m, "k": 2}},
               tmp_path / "sae_model.pt")
    pd.DataFrame({"feature_id": list(range(m)), "concept": ["x"] * m,
                  "fidelity_pass": [True] * m}).to_csv(
        tmp_path / "feature_names.csv", index=False)
    (tmp_path / "manifest.json").write_text(_json.dumps(
        {"input_rep": "difference", "embed_model_id": "Qwen/Qwen3-Embedding-0.6B"}))


def test_from_dir_loads_projector_names_manifest(tmp_path):
    _write_synthetic_lens(tmp_path)
    lens = Lens.from_dir(tmp_path, device="cpu")
    assert lens.projector.m_total == 3 and lens.projector.input_dim == 4
    assert lens.input_rep == "difference"
    assert lens.names is not None
    assert lens.embedder is not None        # constructed lazily; no model download


def test_from_dir_merges_bundled_and_external_annotations(tmp_path):
    _write_synthetic_lens(tmp_path)
    pd.DataFrame({
        "feature_id": [0, 1, 2],
        "concept": ["x", "x", "x"],
        "fidelity_pass": [True, False, True],
    }).to_csv(tmp_path / "feature_fidelity.csv", index=False)
    external = tmp_path / "interpret"
    external.mkdir()
    pd.DataFrame({
        "feature_id": [0, 1, 2],
        "semantic_threshold": [0.2, 0.3, 0.4],
        "presence_pass": [True, True, False],
    }).to_csv(external / "feature_calibration.csv", index=False)

    lens = Lens.from_dir(tmp_path, annotations=external)

    assert {"concept", "fidelity_pass", "semantic_threshold", "presence_pass"} <= \
        set(lens.feature_table.columns)
    assert lens.feature_table.set_index("feature_id").loc[1, "semantic_threshold"] == 0.3


def test_inference_only_bundle_round_trips_through_regular_loader(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_synthetic_lens(source)
    loaded = Lens.from_dir(source)
    release = tmp_path / "release"

    loaded.save(release, inference_only=True)
    restored = Lens.from_dir(release)

    assert restored.projector.m_total == loaded.projector.m_total
    assert restored.input_rep == "difference"
    assert list(restored.concept_names) == list(loaded.concept_names)


@pytest.mark.parametrize("replacement", ["weights", "whitener"])
def test_loaded_native_identity_stays_bound_after_backing_replacement(tmp_path, replacement):
    from prefscope import PrecomputedRepresentationSource, RepresentationBatch
    from prefscope.sae.whiten import Whitener

    source = tmp_path / "source"
    source.mkdir()
    _write_synthetic_lens(source)
    checkpoint = torch.load(source / "sae_model.pt", weights_only=True)
    checkpoint["state_dict"]["encoder.weight"].fill_(1.0)
    torch.save(checkpoint, source / "sae_model.pt")
    lens = Lens.from_dir(source)
    items = [PairItem(id="row", x="prompt", y_a="A", y_b="B")]
    representations = PrecomputedRepresentationSource(
        RepresentationBatch(
            row_ids=("row",),
            arrays={"response_a": np.ones((1, 4)), "response_b": np.zeros((1, 4))},
            provenance={
                "representation_family": "text_embedding",
                "embed_model_id": "Qwen/Qwen3-Embedding-0.6B",
            },
        )
    )
    lens.representation_source = representations
    # Do not read feature_space_identity before replacing the backing files: load
    # must bind it eagerly, not when the first extraction happens.
    if replacement == "weights":
        checkpoint["state_dict"]["encoder.weight"].fill_(2.0)
        torch.save(checkpoint, source / "sae_model.pt")
    else:
        Whitener("standardize", np.zeros(4), std=np.full(4, 2.0)).save(source)

    restored = Lens.from_dir(source)
    restored.representation_source = representations
    original_features = lens.featurize(items)
    replacement_features = restored.featurize(items)
    original_matrix = original_features.matrix("z_diff")
    replacement_matrix = replacement_features.matrix("z_diff")

    assert not np.array_equal(original_matrix.values, replacement_matrix.values)
    assert lens.feature_space_id != restored.feature_space_id
    assert lens.backend.feature_space_identity == lens.feature_space_identity
    assert original_matrix.provenance["lens"]["feature_space_id"] == lens.feature_space_id
    lens.feature_catalog.validate_for(original_matrix, require_exact=True)
    with pytest.raises(ValueError, match="different feature spaces"):
        restored.feature_catalog.validate_for(original_matrix, require_exact=True)
    with pytest.raises(ValueError, match="changed since loading"):
        lens.save(tmp_path / "invalid-publication")
    assert not (tmp_path / "invalid-publication").exists()
