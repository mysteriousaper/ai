import json

import pytest

from engine.engines import transformer as T
from engine.mappings.labels import CANONICAL
from engine.schemas import EngineInput, MessageContext

pytestmark = pytest.mark.skipif(not T.models_available(), reason="transformer model files missing")


@pytest.fixture(scope="module")
def eng():
    return T.TransformerEngine()


def test_28_labels_and_probs_in_range(eng):
    o = eng(EngineInput(text="I am so happy to see you!"))
    assert set(o.emotions) == set(CANONICAL) and len(o.emotions) == 28
    assert all(0.0 <= p <= 1.0 for p in o.emotions.values())
    assert 0.0 <= o.confidence <= 1.0 and o.coverage == 1.0
    assert o.emotions["joy"] > o.emotions["sadness"]
    assert len(o.features["top5"]) == 5 and o.features["backend"] in ("onnx", "torch")


def test_label_map_by_name():
    import json as _j

    cfg = _j.loads((T.TORCH_DIR / "config.json").read_text()) if (T.TORCH_DIR / "config.json").exists() else None
    if cfg is None:
        pytest.skip("torch config missing")
    assert [cfg["id2label"][str(i)] for i in range(28)] == list(CANONICAL)


def test_thresholds_file_respected(eng, tmp_path, monkeypatch):
    f = tmp_path / "thr.json"
    f.write_text(json.dumps(dict.fromkeys(CANONICAL, 0.0)))
    monkeypatch.setattr(T, "THRESHOLDS_PATH", f)
    e = T.TransformerEngine(eng.backend_name)
    e._backend = eng.backend
    assert len(e(EngineInput(text="hello there")).features["predicted_labels"]) == 28
    f.write_text(json.dumps(dict.fromkeys(CANONICAL, 1.1)))
    e = T.TransformerEngine(eng.backend_name)
    e._backend = eng.backend
    assert e(EngineInput(text="hello there")).features["predicted_labels"] == []


def test_default_threshold_is_half(eng, monkeypatch, tmp_path):
    monkeypatch.setattr(T, "THRESHOLDS_PATH", tmp_path / "missing.json")
    e = T.TransformerEngine(eng.backend_name)
    assert (e.thresholds == 0.5).all()


def test_non_english_skipped(eng):
    o = eng(EngineInput(text="Ich bin so glücklich", context=MessageContext(language="de")))
    assert o.features["skipped"].startswith("unsupported_language") and o.coverage == 0.0 and not o.emotions


def test_empty_text_skipped(eng):
    o = eng(EngineInput(text="   "))
    assert o.features["skipped"] == "empty_text" and o.confidence == 0.0


def test_long_text_coverage_and_no_crash(eng):
    o = eng(EngineInput(text="I am sad. " * 400))
    assert 0 < o.coverage < 1.0


@pytest.mark.skipif(not T.RIDGE_PATH.exists(), reason="VAD ridge not fitted")
def test_vad_present(eng):
    o = eng(EngineInput(text="This is wonderful news"))
    assert o.vad is not None and all(s > 0 for s in o.vad.std)
