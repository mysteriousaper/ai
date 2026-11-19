import json
from collections import Counter
from pathlib import Path

from engine.engines.lexicon import LexiconEngine
from engine.mappings.labels import CANONICAL
from engine.schemas import DesiredGoal, TemporalFocus

PATH = Path("eval/challenge_set.jsonl")
ITEMS = [json.loads(line) for line in PATH.read_text().splitlines() if line.strip()]


def test_size_ids_and_categories():
    assert len(ITEMS) >= 40
    assert len({i["id"] for i in ITEMS}) == len(ITEMS)
    counts = Counter(i["category"] for i in ITEMS)
    assert all(n >= 3 for n in counts.values()), counts


def test_labels_are_valid():
    for i in ITEMS:
        e = i["expected"]
        assert e["valence_sign"] in (-1, 0, 1)
        assert 1 <= len(e["top_emotions"]) <= 3 and set(e["top_emotions"]) <= set(CANONICAL), i["id"]
        assert e["experiencer_of_main_emotion"] in {"self", "other", "unknown"}
        TemporalFocus(e["temporal_focus"])
        assert all(DesiredGoal(g) for g in e["desired_goals"])
        assert isinstance(e["sarcasm"], bool)


def test_run_challenge_smoke():
    from eval.challenge import run_challenge

    res = run_challenge(LexiconEngine(), ITEMS[:8])
    assert res["ALL"]["n"] == 8
