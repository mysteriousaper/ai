"""Casual ways around the §10 seal on Adapter.suggest must not produce a suggestion."""

from typing import Any

import pytest
from test_adapter_safety import PROFILE, SpyAdapter, make_state

from engine.adapters import base
from engine.adapters.base import Adapter, SafetyResponse


def _bypass(self: Any, state: Any, profile: Any) -> Any:
    return self.adapt(state, profile)


def _assert_safe(adapter: Any) -> None:
    out = adapter.suggest(make_state(True), PROFILE)
    assert isinstance(out, SafetyResponse)
    assert adapter.calls == 0


def test_type_setattr_on_subclass_does_not_win() -> None:
    class Sub(SpyAdapter):
        pass

    type.__setattr__(Sub, "suggest", _bypass)
    with pytest.raises(TypeError, match="suggest"):
        Sub()  # instantiation re-checks the class
    a = object.__new__(Sub)
    a.calls = 0
    _assert_safe(a)  # even without the instantiation check, lookup stays sealed
    assert Sub.suggest is base._sealed_suggest


def test_type_setattr_after_instantiation_does_not_win() -> None:
    class Sub(SpyAdapter):
        pass

    a = Sub()
    type.__setattr__(Sub, "suggest", _bypass)
    _assert_safe(a)
    out = type(a).suggest(a, make_state(True), PROFILE)
    assert isinstance(out, SafetyResponse)


def test_type_setattr_on_adapter_base_does_not_win() -> None:
    original = Adapter.__dict__["suggest"]
    type.__setattr__(Adapter, "suggest", _bypass)
    try:
        _assert_safe(_fresh_spy())
    finally:
        type.__setattr__(Adapter, "suggest", original)


def _fresh_spy() -> Any:
    a = object.__new__(SpyAdapter)
    a.calls = 0
    return a


def test_class_decorator_patch_is_caught() -> None:
    def deco(cls: Any) -> Any:
        type.__setattr__(cls, "suggest", _bypass)
        return cls

    @deco
    class Sub(SpyAdapter):
        pass

    with pytest.raises(TypeError, match="suggest"):
        Sub()


def test_type_setattr_getattribute_before_instantiation_is_caught() -> None:
    class Sub(SpyAdapter):
        pass

    type.__setattr__(Sub, "__getattribute__", object.__getattribute__)
    with pytest.raises(TypeError, match="__getattribute__"):
        Sub()


def test_derived_metaclass_cannot_skip_the_class_check() -> None:
    with pytest.raises(TypeError, match="metaclass"):

        class Meta(type(Adapter)):  # type: ignore[misc]
            def __new__(mcls, name: str, bases: Any, ns: Any, **kw: Any) -> Any:
                return type.__new__(mcls, name, bases, ns)


def test_derived_metaclass_cannot_reopen_setattr() -> None:
    with pytest.raises(TypeError, match="metaclass"):

        class Meta(type(Adapter)):  # type: ignore[misc]
            def __setattr__(cls, attr: str, value: Any) -> None:
                type.__setattr__(cls, attr, value)


def test_harmless_derived_metaclass_still_works() -> None:
    class Meta(type(Adapter)):  # type: ignore[misc]
        def describe(cls) -> str:
            return f"adapter {cls.__name__}"

    class Sub(SpyAdapter, metaclass=Meta):
        pass

    _assert_safe(Sub())


def test_sealed_descriptor_cannot_be_retargeted() -> None:
    desc = Adapter.__dict__["suggest"]
    with pytest.raises(AttributeError):
        desc._func = _bypass
    _assert_safe(SpyAdapter())
