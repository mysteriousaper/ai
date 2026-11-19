import pytest

from engine.preprocessing.clauses import analyze_clauses, sarcasm_cues, temporal_focus


def test_other_then_self():
    c = analyze_clauses("My brother is nervous; I am excited.")
    assert [x.experiencer for x in c] == ["other", "self"]
    assert c[0].who == "My brother" and c[0].tense == "present"


def test_past_then_present_with_contrast():
    c = analyze_clauses("I was happy last week, but now I'm anxious.")
    assert [x.tense for x in c] == ["past", "present"]
    assert [x.contrast_after for x in c] == [False, True]
    assert temporal_focus(c) == "mixed"


def test_current_and_desire_clause():
    c = analyze_clauses("I feel anxious but want to feel adventurous.")
    assert len(c) == 2
    assert not c[0].is_desire and c[0].tense == "present" and c[0].experiencer == "self"
    assert c[1].is_desire and c[1].experiencer == "self" and c[1].contrast_after


def test_used_to_is_past_and_would_rather_is_desire():
    c = analyze_clauses("I used to love crowds; now I'd rather be alone.")
    assert len(c) == 2
    assert c[0].tense == "past" and not c[0].is_desire
    assert c[1].is_desire and c[1].tense == "present"


def test_negation_is_per_clause():
    c = analyze_clauses("I am not upset, just tired.")
    assert [x.negated for x in c] == [True, False]
    assert c[1].experiencer == "self"


@pytest.mark.parametrize("text,tense", [
    ("I will be fine.", "future"), ("I'm going to miss her.", "future"), ("I gonna cry.", "future"),
    ("I cried a lot.", "past"),
])
def test_tense(text, tense):
    assert analyze_clauses(text)[0].tense == tense


def test_non_english_and_empty():
    assert analyze_clauses("") == []
    assert analyze_clauses("Ես շատ ուրախ եմ այսօր") == []


def test_non_person_subject_is_not_an_experiencer():
    assert analyze_clauses("The flight was delayed.")[0].experiencer == "unknown"
    assert analyze_clauses("She is furious.")[0].experiencer == "other"


def test_sarcasm_cues():
    assert sarcasm_cues("Fantastic, another delayed flight.") >= 0.5
    assert sarcasm_cues("Oh great, it rained again.") >= 0.5
    assert sarcasm_cues("Yeah right.") >= 0.5
    assert sarcasm_cues("Had a fantastic day at the beach") < 0.5
    assert sarcasm_cues("I love Mondays") < 0.5
    assert 0.0 <= sarcasm_cues("Perfect, just what I needed, my laptop is broken again!!!") <= 1.0
