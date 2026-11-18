"""Domain adapter interface (brief §4.9) with the core crisis-safety rule (brief §10).

Adapters turn an :class:`~engine.schemas.EmotionalStateObject` into a domain suggestion
(travel, tutoring, ...). Callers always go through :meth:`Adapter.suggest`, never ``adapt``.

**Hard rule (§10):** if ``state.safety_flag`` is set, ``suggest`` returns a
:class:`SafetyResponse` (human help resources, no "solution") and does **not** call ``adapt``.
This lives in the core, not in the adapter, so it is sealed against subclasses:

* defining ``suggest`` (directly or via a mixin earlier in the MRO) raises ``TypeError`` at
  class-creation time, as does overriding ``__getattribute__``;
* ``suggest`` is a data descriptor, so it cannot be shadowed by an instance attribute, and
  assigning or deleting it on the class or an instance raises.

Python cannot stop deliberate tampering with the core itself (e.g. editing this module); the
seal is against adapters accidentally or casually bypassing the rule.
"""

import inspect
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from types import MethodType
from typing import Any, ClassVar, Literal

from pydantic import Field

from engine.schemas import EmotionalStateObject, FeatureName, Probability, StrictModel

DEFAULT_SAFETY_MESSAGE = (
    "It sounds like things might be really hard right now. You don't have to deal with this "
    "alone: please reach out to someone you trust or a local helpline. If you are in danger, "
    "call your local emergency number now. This system is not a clinician and cannot help in "
    "an emergency."
)


class UserProfile(StrictModel):
    """Domain-neutral facts about the user that an adapter may use."""

    user_id: str | None = None
    region: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")  # ISO 3166-1 alpha-2
    preferences: dict[FeatureName, str | float | bool] = Field(default_factory=dict)


class SuggestionItem(StrictModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    score: Probability
    rationale: str = ""


class DomainSuggestion(StrictModel):
    kind: Literal["suggestion"] = "suggestion"
    adapter: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    items: list[SuggestionItem]


class HelpResource(StrictModel):
    """A human help service. Numbers must come from a curated, dated table (R3 §5)."""

    name: str = Field(min_length=1)
    contact: str = Field(min_length=1)
    region: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    url: str | None = None


class SafetyResponse(StrictModel):
    """What a product must show instead of a suggestion when ``safety_flag`` is set."""

    kind: Literal["safety"] = "safety"
    show_human_help: Literal[True] = True
    message: str = DEFAULT_SAFETY_MESSAGE
    resources: list[HelpResource] = Field(default_factory=list)
    reason: str | None = None


def build_safety_response(state: EmotionalStateObject, profile: UserProfile) -> SafetyResponse:
    """Core-owned safety response. Adapters cannot change it.

    ``resources`` is empty until the curated helpline table exists (R3 §5: numbers must be
    verified by the owner, and Find a Helpline's terms forbid scraping). ``profile.region`` is
    the hook for picking local resources.
    """
    return SafetyResponse(reason=state.safety_reason)


class _SealedSuggest:
    """Data descriptor for the sealed ``suggest``: instance attributes cannot shadow it.

    It holds no state, so there is nothing to retarget; it always binds the module-level
    :func:`_sealed_suggest`.
    """

    __slots__ = ()

    def __get__(self, obj: object, owner: type | None = None) -> Any:
        if obj is None:
            return _sealed_suggest
        return MethodType(_sealed_suggest, obj)

    def __set__(self, obj: object, value: object) -> None:
        raise AttributeError("Adapter.suggest is sealed (brief §10 safety rule)")

    def __delete__(self, obj: object) -> None:
        raise AttributeError("Adapter.suggest is sealed (brief §10 safety rule)")

    def __setattr__(self, attr: str, value: object) -> None:
        raise AttributeError("Adapter.suggest is sealed (brief §10 safety rule)")


# Set once, right after Adapter is defined; kept outside the metaclass so that reading them does
# not trigger the descriptor protocol.
_SEALED: _SealedSuggest | None = None
_SEALED_GETATTRIBUTE: Callable[..., Any] | None = None

# Metaclass hooks that, if overridden by a derived metaclass, could skip or undo the checks.
_PROTECTED_META_HOOKS = (
    "__new__",
    "__init__",
    "__call__",
    "__setattr__",
    "__delattr__",
    "__getattribute__",
)


def _check_sealed(cls: type) -> None:
    """Raise ``TypeError`` unless ``cls`` still resolves ``suggest`` and ``__getattribute__`` to
    the core versions (via ``__mro__``, so mixins and ``type.__setattr__`` patches count)."""
    if inspect.getattr_static(cls, "suggest", None) is not _SEALED:
        raise TypeError(
            f"{cls.__name__} must not override suggest(): the safety rule is part of the core"
        )
    if inspect.getattr_static(cls, "__getattribute__") is not _SEALED_GETATTRIBUTE:
        raise TypeError(f"{cls.__name__} must not override __getattribute__ (would hide suggest)")


class _AdapterMeta(ABCMeta):
    def __init_subclass__(mcls, **kwargs: Any) -> None:
        # A derived metaclass could skip __new__'s check or reopen __setattr__; refuse it.
        super().__init_subclass__(**kwargs)
        overridden = [h for h in _PROTECTED_META_HOOKS if h in mcls.__dict__]
        if overridden:
            raise TypeError(
                f"metaclass {mcls.__name__} must not override {', '.join(overridden)}: "
                "they enforce the brief §10 safety rule"
            )

    def __new__(
        mcls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs: Any
    ) -> "_AdapterMeta":
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        if _SEALED is not None:  # i.e. cls is a subclass of Adapter
            _check_sealed(cls)
        return cls

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        # Re-check on every instantiation: catches class decorators and type.__setattr__ patches
        # made after class creation.
        _check_sealed(cls)
        return super().__call__(*args, **kwargs)

    def __getattribute__(cls, attr: str) -> Any:
        if attr == "suggest":  # Sub.suggest(obj, ...) stays sealed even if the dict was patched
            return _sealed_suggest
        return super().__getattribute__(attr)

    def __setattr__(cls, attr: str, value: Any) -> None:
        if attr in ("suggest", "__getattribute__"):
            raise TypeError("Adapter.suggest is sealed (brief §10 safety rule)")
        super().__setattr__(attr, value)

    def __delattr__(cls, attr: str) -> None:
        if attr in ("suggest", "__getattribute__"):
            raise TypeError("Adapter.suggest is sealed (brief §10 safety rule)")
        super().__delattr__(attr)


def _sealed_suggest(
    self: "Adapter", state: EmotionalStateObject, profile: UserProfile
) -> DomainSuggestion | SafetyResponse:
    """Return ``self.adapt(state, profile)``, or the core :class:`SafetyResponse` without calling
    ``adapt`` when ``state.safety_flag`` is set (brief §10)."""
    if not isinstance(state, EmotionalStateObject):
        raise TypeError("state must be an EmotionalStateObject")
    if state.safety_flag is not False:  # anything but an explicit False is treated as set
        return build_safety_response(state, profile)
    result = self.adapt(state, profile)
    if not isinstance(result, DomainSuggestion):
        raise TypeError(f"{type(self).__name__}.adapt must return a DomainSuggestion")
    return result


class Adapter(metaclass=_AdapterMeta):
    """Base class for domain adapters. Implement :meth:`adapt`; call :meth:`suggest`."""

    name: ClassVar[str]
    version: ClassVar[str]

    @abstractmethod
    def adapt(self, state: EmotionalStateObject, profile: UserProfile) -> DomainSuggestion:
        """Map a (non-flagged) state to a domain suggestion. Never called when flagged."""

    suggest = _SealedSuggest()

    def __getattribute__(self, attr: str) -> Any:
        # Resolve ``suggest`` to the core version even if a class ``__dict__`` was patched with
        # ``type.__setattr__`` after creation (that bypasses the metaclass ``__setattr__``).
        if attr == "suggest":
            return MethodType(_sealed_suggest, self)
        return object.__getattribute__(self, attr)


_SEALED = Adapter.__dict__["suggest"]
_SEALED_GETATTRIBUTE = Adapter.__dict__["__getattribute__"]
