import math

import pandas as pd
import pytest

from engine.mappings import labels as L

RAW = "data/raw"


def test_canonical_is_goemotions_28():
    assert len(L.CANONICAL) == 28
    assert len(set(L.CANONICAL)) == 28
    assert L.CANONICAL[-1] == "neutral"


@pytest.mark.parametrize("table", [L.TO_EKMAN, L.TO_PLUTCHIK], ids=["ekman", "plutchik"])
def test_every_emotion_mapped_and_rows_sum_to_one(table):
    non_neutral = set(L.CANONICAL) - {L.NEUTRAL}
    assert set(table) == non_neutral
    for label, row in table.items():
        assert math.isclose(sum(row.values()), 1.0), label
        assert all(w > 0 for w in row.values()), label


def test_targets_are_valid_and_fully_covered():
    assert {c for row in L.TO_EKMAN.values() for c in row} == set(L.EKMAN)
    assert {c for row in L.TO_PLUTCHIK.values() for c in row} == set(L.PLUTCHIK)


def test_ekman_matches_goemotions_reference_groups():
    # spot checks against GoEmotions' published ekman_mapping.json
    assert L.TO_EKMAN["grief"] == {"sadness": 1.0}
    assert L.TO_EKMAN["disapproval"] == {"anger": 1.0}
    assert L.TO_EKMAN["curiosity"] == {"surprise": 1.0}
    assert L.TO_EKMAN["admiration"] == {"joy": 1.0}


def test_projection_examples():
    probs = {"remorse": 0.8, "joy": 0.1}
    plut = L.to_plutchik(probs)
    assert plut["sadness"] == pytest.approx(0.4)
    assert plut["disgust"] == pytest.approx(0.4)
    assert plut["joy"] == pytest.approx(0.1)
    assert L.to_ekman(probs)["sadness"] == pytest.approx(0.8)


def test_neutral_projects_to_nothing():
    assert all(v == 0 for v in L.to_ekman({"neutral": 0.9}).values())


def test_other_dataset_maps_point_to_canonical():
    for table in (L.DAILYDIALOG_TO_CANONICAL, L.SEMEVAL_EC_TO_CANONICAL, L.EMOSURV_TO_CANONICAL):
        assert set(table.values()) <= set(L.CANONICAL)


def test_ed_maps_cover_all_32_situations_in_the_data():
    path = f"{RAW}/empathetic_dialogues/empatheticdialogues/train.csv"
    try:
        ed = pd.read_csv(path, usecols=["context"], on_bad_lines="skip")
    except FileNotFoundError:
        pytest.skip("EmpatheticDialogues not fetched")
    labels = set(ed.context.unique())
    assert len(labels) == 32
    assert labels == set(L.ED_TO_EVENT)
    assert set(L.ED_TO_AGENT) <= labels


def test_goemotions_label_order_matches_dataset():
    try:
        raw = pd.read_parquet(f"{RAW}/go_emotions/raw.parquet").head(1)
    except FileNotFoundError:
        pytest.skip("GoEmotions not fetched")
    assert [c for c in raw.columns if c in L.CANONICAL] == list(L.CANONICAL)
