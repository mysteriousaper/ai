"""The interface every signal engine (A-F, brief §4.4) implements."""

from abc import ABC, abstractmethod

from engine.schemas import EngineInput, EngineOutput


class SignalEngine(ABC):
    """One independent signal source. Fusion only ever sees :class:`EngineOutput`.

    Implementations must be side-effect free with respect to user data: ``analyze`` must not log,
    store or send ``EngineInput.text`` or keystroke data anywhere (brief §1.8).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier, e.g. ``"lexicon"``; used as ``EngineOutput.engine``."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Engine version; used as ``EngineOutput.engine_version``."""

    @property
    @abstractmethod
    def supported_languages(self) -> frozenset[str]:
        """Primary language subtags this engine can read, e.g. ``{"en"}``."""

    @abstractmethod
    def analyze(self, engine_input: EngineInput) -> EngineOutput:
        """Read one message and return this engine's estimate."""

    def supports(self, language: str | None) -> bool:
        """True if ``language`` (a BCP-47-like tag such as ``"en-GB"``) is supported.

        Matching is on the primary subtag. ``None`` (language unknown) is never supported, so
        callers must route unknown-language input explicitly (brief §9).
        """
        if not language:
            return False
        return language.split("-", 1)[0].lower() in self.supported_languages
