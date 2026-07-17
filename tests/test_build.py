import numpy as np
import pandas as pd

from prefscope.encode.build import build_codes


class FakeEmbedder:
    def encode(self, prompts, completions):
        # map each completion's first char to a 3-dim vector
        return np.array([[float(ord(c[0])), 0.0, 0.0] for c in completions],
                        dtype=np.float32)


class FakeProjector:
    m_total = 3
    input_dim = 3

    def project(self, x):
        return np.asarray(x, dtype=np.float32) * 2.0

    def residual_norm(self, x):
        return np.zeros(len(x), dtype=np.float32)


def test_build_codes_shapes_and_diff():
    oriented = pd.DataFrame({
        "prompt": ["p1", "p2"],
        "self_completion": ["a...", "b..."],   # ord 'a'=97, 'b'=98
        "other_completion": ["c...", "d..."],  # ord 'c'=99, 'd'=100
    })
    codes = build_codes(oriented, FakeEmbedder(), FakeProjector())
    assert codes["z_self"].shape == (2, 3)
    assert codes["z_other"].shape == (2, 3)
    # z_self row0 = [97*2,0,0]; z_other row0 = [99*2,0,0]; diff = [-4,0,0]
    np.testing.assert_allclose(codes["z_diff"][0], [97 * 2 - 99 * 2, 0, 0])
    np.testing.assert_allclose(codes["z_self"][1], [98 * 2, 0, 0])
    assert codes["residual_self"].shape == (2,)
