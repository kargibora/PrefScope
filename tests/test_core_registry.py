import pytest
from prefscope.core import registry


def test_register_and_get():
    @registry.register("widget", "alpha")
    class Alpha:
        pass

    assert registry.get("widget", "alpha") is Alpha
    assert "alpha" in registry.available("widget")


def test_get_unknown_name_raises_with_options():
    @registry.register("gadget", "known")
    class Known:
        pass

    with pytest.raises(KeyError) as exc:
        registry.get("gadget", "missing")
    assert "missing" in str(exc.value) and "known" in str(exc.value)


def test_duplicate_registration_raises():
    @registry.register("thing", "dup")
    class First:
        pass

    with pytest.raises(ValueError):
        @registry.register("thing", "dup")
        class Second:
            pass
