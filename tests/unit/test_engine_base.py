"""The SignalEngine interface (brief §4.4)."""

from datetime import UTC, datetime

import pytest

from engine.engines.base import SignalEngine
from engine.schemas import EngineInput, EngineOutput, MessageContext


class EchoEngine(SignalEngine):
    """Test double: reports coverage 1 and a flat neutral score."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def version(self) -> str:
        return "0.0.1"

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset({"en", "ru"})

    def analyze(self, engine_input: EngineInput) -> EngineOutput:
        return EngineOutput(
            engine=self.name,
            engine_version=self.version,
            emotions={"neutral": 1.0},
            features={"chars": float(len(engine_input.text))},
            confidence=0.1,
            coverage=1.0,
        )


def _input(language: str | None = "en") -> EngineInput:
    ctx = MessageContext(
        session_id="s", message_index=0, timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        language=language,
    )
    return EngineInput(text="hello", context=ctx)


def test_cannot_instantiate_abstract_engine() -> None:
    with pytest.raises(TypeError):
        SignalEngine()  # type: ignore[abstract]


def test_incomplete_engine_is_abstract() -> None:
    class Half(SignalEngine):
        @property
        def name(self) -> str:
            return "half"

    with pytest.raises(TypeError):
        Half()  # type: ignore[abstract]


def test_double_implements_interface() -> None:
    eng = EchoEngine()
    out = eng.analyze(_input())
    assert isinstance(out, EngineOutput)
    assert out.engine == "echo" and out.features["chars"] == 5.0


@pytest.mark.parametrize(
    ("lang", "ok"), [("en", True), ("en-GB", True), ("ru", True), ("hy", False), (None, False)]
)
def test_supports_uses_primary_language_subtag(lang: str | None, ok: bool) -> None:
    assert EchoEngine().supports(lang) is ok
