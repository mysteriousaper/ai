import pytest

from engine.engines import semantic_cluster as sc
from engine.engines.semantic_cluster import SemanticClusterEngine, artifacts_available
from engine.mappings.labels import CANONICAL
from engine.schemas import EngineInput

needs_map = pytest.mark.skipif(not artifacts_available(), reason="cluster map not built")


def test_missing_artifacts_gives_empty(tmp_path):
    out = SemanticClusterEngine(map_dir=tmp_path)(EngineInput(text="I am happy"))
    assert out.features["skipped"] == "cluster_map_missing"
    assert out.confidence == 0.0 and out.coverage == 0.0 and out.emotions == {}


def test_metadata():
    assert SemanticClusterEngine.name == "semantic_cluster"
    assert SemanticClusterEngine.version == "0.1.0"
    assert SemanticClusterEngine.languages == frozenset({"en"})
    assert sc.TEMPERATURE == 0.05


@needs_map
def test_output_valid_and_deterministic():
    eng = SemanticClusterEngine()
    a = eng(EngineInput(text="I am so grateful, thank you!"))
    b = eng(EngineInput(text="I am so grateful, thank you!"))
    assert a == b
    assert set(a.emotions) == set(CANONICAL)
    assert all(0.0 <= p <= 1.0 for p in a.emotions.values())
    assert sum(a.emotions.values()) <= 28
    assert a.vad is None and a.coverage == 1.0 and 0.0 <= a.confidence <= 1.0
    assert len(a.features["top_leaves"]) == 3


@needs_map
def test_gratitude_leaf():
    out = SemanticClusterEngine()(EngineInput(text="I am so grateful, thank you!"))
    assert "gratitude" in out.features["top_leaves"][0]["dominant_labels"]


@needs_map
@pytest.mark.xfail(
    reason="known limitation: MiniLM places intensifier-style phrases near admiration leaves; fear is ~1% of GoEmotions",
    strict=False,
)
def test_fear_in_top3():
    out = SemanticClusterEngine()(EngineInput(text="this is terrifying"))
    top3 = sorted(out.emotions, key=out.emotions.get, reverse=True)[:3]
    assert {"fear", "nervousness"} & set(top3)
