from prefscope.api.config import SAEConfig, TrainConfig


def test_sae_config_defaults():
    c = SAEConfig()
    assert c.m == 128
    assert c.k == 16
    assert c.input_rep == "individual"
    assert c.matryoshka_prefix == (8,)


def test_train_config_defaults_and_nesting():
    c = TrainConfig()
    assert isinstance(c.sae, SAEConfig)
    assert c.sae.m == 128
    assert c.embed_model_id is None
    assert c.val_frac == 0.1
    assert c.device == "cpu"
    assert c.max_train_rows is None
    assert c.train_kwargs == {}


def test_train_config_independent_sae_instances():
    a = TrainConfig()
    b = TrainConfig()
    assert a.sae is not b.sae
    a.sae.m = 64
    assert b.sae.m == 128


def test_train_config_custom_sae():
    c = TrainConfig(sae=SAEConfig(m=64, k=8, input_rep="difference"),
                    embed_model_id="some/model", train_kwargs={"epochs": 3})
    assert c.sae.m == 64 and c.sae.k == 8 and c.sae.input_rep == "difference"
    assert c.embed_model_id == "some/model"
    assert c.train_kwargs == {"epochs": 3}
