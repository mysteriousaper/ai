"""Rule-based clause analysis (English only): who feels it, when, negated?, desire?, contrast?

Built on the shared spaCy pipeline (`pipeline._nlp`). Pure heuristics, no training. Used from
Session 4 on to attribute emotions to an experiencer, weight them by time, and separate the
current state from the desired state (schema v0.2).

`sarcasm_cues` is only a *flag*: a cue score in [0, 1] that should lower confidence downstream.
It never flips a label, because lexical cues alone cannot tell sarcasm from a genuine complaint.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

from engine.preprocessing.pipeline import _nlp, detect_language

SELF_PRONOUNS = {"i", "me", "we", "us", "myself", "ourselves"}
OTHER_PRONOUNS = {"he", "she", "they", "him", "her", "them", "you", "who"}
NON_PERSON_PRONOUNS = {"it", "this", "that", "there", "everything", "something", "nothing", "everyone"}
PERSON_NOUNS = {
    "brother", "sister", "mom", "mum", "mother", "dad", "father", "wife", "husband", "friend", "boss",
    "colleague", "coworker", "son", "daughter", "kid", "child", "parent", "partner", "girlfriend",
    "boyfriend", "cousin", "uncle", "aunt", "grandma", "grandmother", "grandpa", "grandfather", "teacher",
    "neighbor", "neighbour", "manager", "family", "team", "people", "everyone", "man", "woman", "girl", "boy",
}
NEGATORS = {"not", "n't", "never", "no", "nobody", "nothing", "neither", "nor", "cannot", "hardly"}
CONTRAST_WORDS = {"but", "however", "though", "although", "yet"}
HARD_BREAKS = {".", "!", "?", ";", ":", "…"}
DESIRE_VERBS = {"want", "wish", "hope", "need", "prefer", "crave", "long"}
WOULD = {"would", "'d", "d"}
PAST_TIME = re.compile(r"\b(yesterday|last (week|month|year|night)|ago|back then|used to|earlier|before)\b", re.IGNORECASE)
FUTURE_TIME = re.compile(r"\b(tomorrow|next (week|month|year)|soon|tonight|later)\b", re.IGNORECASE)
PRESENT_TIME = re.compile(r"\b(now|today|currently|right now|these days)\b", re.IGNORECASE)


@dataclass
class Clause:
    text: str
    experiencer: str = "unknown"  # self | other | unknown
    who: str | None = None  # subject phrase when explicit, e.g. "my brother"
    tense: str = "unknown"  # past | present | future | unknown
    negated: bool = False
    is_desire: bool = False
    contrast_after: bool = False


def _split(doc) -> list[tuple[int, int, bool]]:
    """Token spans (start, end, contrast_after) of candidate clauses."""
    spans: list[tuple[int, int, bool]] = []
    start, contrast, n = 0, False, len(doc)
    for i, t in enumerate(doc):
        low = t.lower_
        brk = False
        if t.text in HARD_BREAKS or t.text == ",":
            brk = True
        elif low in CONTRAST_WORDS and i == start and t.dep_ in {"cc", "advmod", "mark"}:
            start, contrast = i + 1, True
            continue
        elif low in CONTRAST_WORDS and i > start and t.dep_ in {"cc", "advmod", "mark"}:
            spans.append((start, i, contrast))
            start, contrast = i + 1, True
            continue
        elif (low in {"and", "so", "because"} and t.dep_ in {"cc", "mark"} and i > start
              and any(x.dep_ in {"nsubj", "nsubjpass"} for x in doc[i + 1 : i + 4])):
            spans.append((start, i, contrast))
            start, contrast = i + 1, False
            continue
        if brk:
            spans.append((start, i, contrast))
            start = i + 1
            contrast = contrast and t.text == ","  # a hard stop ends the contrast
            if t.text in HARD_BREAKS:
                contrast = False
    if start < n:
        spans.append((start, n, contrast))
    spans = [s for s in spans if any(not x.is_punct and not x.is_space for x in doc[s[0] : s[1]])]
    return _merge_fragments(doc, spans)


def _has_predicate(doc, s: int, e: int) -> bool:
    return any(t.pos_ in {"VERB", "AUX", "ADJ"} for t in doc[s:e])


def _merge_fragments(doc, spans: list[tuple[int, int, bool]]) -> list[tuple[int, int, bool]]:
    """Fragments with no verb/adjective ("Yesterday,", "Honestly,") join their neighbour."""
    out: list[tuple[int, int, bool]] = []
    carry: tuple[int, bool] | None = None
    for s, e, c in spans:
        if carry is not None:
            s, c = carry[0], c or carry[1]
            carry = None
        if not _has_predicate(doc, s, e) and (s, e, c) is not spans[-1]:
            carry = (s, c)
            continue
        out.append((s, e, c))
    if carry is not None and out:
        s, e, c = out[-1]
        out[-1] = (s, doc[-1].i + 1, c)
    elif carry is not None:
        out.append((carry[0], len(doc), carry[1]))
    return out


def _subject(doc, s: int, e: int):
    for t in doc[s:e]:
        if t.dep_ in {"nsubj", "nsubjpass"}:
            return t
    return None


def _tense(doc, s: int, e: int, text: str) -> str:
    toks = list(doc[s:e])
    low = [t.lower_ for t in toks]
    if re.search(r"\bgon+a\b", text.lower()):
        return "future"
    for i, t in enumerate(toks):
        if t.lower_ == "used" and i + 1 < len(toks) and toks[i + 1].lower_ == "to":
            return "past"
        if t.lower_ in {"will", "'ll", "shall", "gonna"} or (t.lower_ == "wo" and "n't" in low):
            return "future"
        if t.lemma_ == "go" and t.tag_ == "VBG" and i + 2 < len(toks) and toks[i + 1].lower_ == "to" \
                and toks[i + 2].pos_ in {"VERB", "AUX"}:
            return "future"
    for t in toks:
        tense = t.morph.get("Tense")
        if tense and t.pos_ in {"VERB", "AUX"}:
            return "past" if tense[0] == "Past" else "present"
    if PAST_TIME.search(text):
        return "past"
    if FUTURE_TIME.search(text):
        return "future"
    if any(t.pos_ in {"VERB", "AUX"} for t in toks) or PRESENT_TIME.search(text):
        return "present"
    return "unknown"


def _is_desire(toks) -> bool:
    lemmas = [t.lemma_.lower() for t in toks]
    low = [t.lower_ for t in toks]
    if any(t.lemma_.lower() in DESIRE_VERBS and t.pos_ == "VERB" for t in toks):
        return True
    has_would = any(w in WOULD for w in low)
    if has_would and any(w in {"like", "love", "prefer", "rather"} for w in low + lemmas):
        return True
    if "rather" in low and has_would:
        return True
    return any(t.lemma_ == "look" and i + 1 < len(toks) and toks[i + 1].lower_ == "for" for i, t in enumerate(toks))


def analyze_clauses(text: str) -> list[Clause]:
    """Split a message into clauses with experiencer, tense, negation, desire and contrast flags."""
    if not text or not text.strip() or detect_language(text) != "en":
        return []
    doc = _nlp()(text.strip())
    first_person = any(t.lower_ in SELF_PRONOUNS or t.lower_ in {"my", "mine", "i'm", "i've"} for t in doc)
    clauses: list[Clause] = []
    for s, e, contrast in _split(doc):
        toks = list(doc[s:e])
        ctext = doc[s:e].text.strip()
        c = Clause(text=ctext, contrast_after=contrast)
        subj = _subject(doc, s, e)
        if subj is None:
            c.experiencer = "self" if first_person else "unknown"
        else:
            sub_toks = [x for x in subj.subtree if s <= x.i < e]
            phrase = doc[sub_toks[0].i : sub_toks[-1].i + 1].text
            low = subj.lower_
            if low in SELF_PRONOUNS:
                c.experiencer = "self"
            elif low in NON_PERSON_PRONOUNS:
                c.experiencer = "unknown"
            elif low in OTHER_PRONOUNS or subj.pos_ == "PROPN" or subj.lemma_.lower() in PERSON_NOUNS or subj.pos_ in {"NOUN"} and any(x.lower_ in {"my", "our"} for x in sub_toks) \
                    and subj.lemma_.lower() in PERSON_NOUNS:
                c.experiencer, c.who = "other", phrase
            else:
                c.experiencer = "unknown"  # non-sentient subject ("the flight"): an event, not an experiencer
        if c.experiencer == "other" and c.who is None:
            c.who = phrase if subj is not None else None
        c.tense = _tense(doc, s, e, ctext)
        c.negated = any(t.dep_ == "neg" or t.lower_ in NEGATORS for t in toks)
        c.is_desire = _is_desire(toks)
        clauses.append(c)
    # verbless fragments ("just tired") inherit the tense of the clause before them
    for prev, cur in itertools.pairwise(clauses):
        if cur.tense == "unknown":
            cur.tense = prev.tense
    return clauses


def temporal_focus(clauses: list[Clause]) -> str:
    """Overall time focus of the current-state clauses (desire clauses are about the wish, not the time)."""
    tenses = {c.tense for c in clauses if c.tense != "unknown" and not c.is_desire}
    if not tenses:
        tenses = {c.tense for c in clauses if c.tense != "unknown"}
    if not tenses:
        return "unknown"
    return tenses.pop() if len(tenses) == 1 else "mixed"


# ------------------------------------------------------------------ sarcasm cues

_POS = re.compile(
    r"\b(fantastic|great|wonderful|lovely|love|loved|perfect|awesome|brilliant|amazing|fabulous|terrific|"
    r"super|thrilled|just what i needed|exactly what i needed|my favou?rite)\b", re.IGNORECASE)
_NEG_STRONG = re.compile(
    r"\b(delay(ed|s)?|broken|broke|stuck|cancel(l)?ed|late|ruined|crashed|flat|lost|jammed|failed|"
    r"traffic|rained|spilled|overdue)\b", re.IGNORECASE)
_NEG_WEAK = re.compile(r"\b(again|another|monday|mondays|yet another|still)\b", re.IGNORECASE)
_PATTERNS = re.compile(r"(\byeah,? right\b|\boh,? (great|wonderful|fantastic|perfect|joy|lovely)\b|\bsure,? because\b|/s\s*$)", re.IGNORECASE)


def sarcasm_cues(text: str) -> float:
    """Heuristic sarcasm-cue score in [0, 1]. A flag that lowers confidence; never a classifier."""
    if not text:
        return 0.0
    score = 0.0
    if _PATTERNS.search(text):
        score = 0.8
    if _POS.search(text):
        strong = len({m.group(0).lower() for m in _NEG_STRONG.finditer(text)})
        weak = len({m.group(0).lower() for m in _NEG_WEAK.finditer(text)})
        if strong or weak:
            score = max(score, min(1.0, 0.2 + 0.5 * min(strong, 2) + 0.2 * min(weak, 2)))
    return round(score, 2)
