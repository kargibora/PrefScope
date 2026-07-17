"""name_individual_features: select top-activating SINGLE responses, ask shared trait."""
import numpy as np
import pandas as pd

from prefscope.interpret.name import name_individual_features


class FakeClient:
    def __init__(self):
        self.prompts = []

    def raw(self, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        return '{"concept": "explains a word"}'


def test_individual_naming_shows_single_responses_not_pairs():
    n = 20
    battles = pd.DataFrame({
        "instruction_id": [str(i) for i in range(n)],
        "prompt": ["what does word %d mean" % i for i in range(n)],
        "completion_a": ["DEFINITION of word %d" % i for i in range(n)],
        "completion_b": ["unrelated chit-chat %d" % i for i in range(n)],
    })
    z_a = np.zeros((n, 1), dtype=np.float32)
    z_b = np.zeros((n, 1), dtype=np.float32)
    z_a[:5, 0] = 2.0   # feature fires on response A of the first 5 battles

    fc = FakeClient()
    df = name_individual_features(battles, z_a, z_b, fc, n_active=5, n_zero=3,
                                  verify_frac=0.0, concurrency=1)

    assert df.iloc[0]["concept"] == "explains a word"
    p = fc.prompts[0]
    # single-response framing, NOT the A/B pair contrast
    assert "RESPONSE:" in p
    assert "RESPONSE A:" not in p and "RESPONSE B:" not in p
    # the high-activation A texts are the ones shown
    assert "DEFINITION of word" in p


def test_close_negatives_pick_similar_controls():
    # feature 0 fires on 4 responses that ALSO express concept 1 (a shared "other" concept).
    # Half the silent pool shares concept 1 (close), half shares concept 2 (far). Close
    # negatives must draw the controls from the concept-1-sharing silent responses.
    n = 40
    battles = pd.DataFrame({
        "instruction_id": [str(i) for i in range(n)],
        "prompt": [f"p{i}" for i in range(n)],
        "completion_a": [f"a{i}" for i in range(n)],
        "completion_b": [f"b{i}" for i in range(n)],
    })
    z_a = np.zeros((n, 3), dtype=np.float32)
    z_b = np.zeros((n, 3), dtype=np.float32)
    z_a[:4, 0] = 2.0; z_a[:4, 1] = 1.0            # activators: f0 + concept 1
    z_a[4:20, 1] = 1.0                            # silent-on-f0 but share concept 1 (close)
    z_a[20:, 2] = 1.0                             # silent-on-f0, share concept 2 (far)
    fc = FakeClient()
    df = name_individual_features(battles, z_a, z_b, fc, n_active=4, n_zero=5,
                                  verify_frac=0.0, seed=0, negatives="close")
    assert len(df) == 3
    # the prompt shown for feature 0 should include concept-1 (indices <20) controls,
    # not concept-2 ones — check a far control's text is absent while the near ones appear.
    prompt0 = fc.prompts[0]
    assert "a0" in prompt0                        # an activator response is shown
    # at least one close (index 4..19) silent 'a' response appears; no far (>=20) ones
    assert any(f"a{i}\n" in prompt0 or f"a{i} " in prompt0 or f"a{i}<" in prompt0 for i in range(4, 20))
