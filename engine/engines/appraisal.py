"""Engine E: appraisal ("what happened" + needs), fully local.

Event type = 0.5 * zero-shot NLI (multilingual MiniLM) + 0.5 * fallback LogisticRegression on
MiniLM sentence embeddings trained on EmpatheticDialogues (English only). The Claude API path is an
optional stub, off by default (env WIYMOOD_USE_CLAUDE) and never called by default.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from engine import safety
from engine.engines.base import Engine
from engine.mappings.labels import ED_TO_AGENT, ED_TO_EVENT, Agent, EventType, NeedType
from engine.preprocessing import clauses as clause_mod
from engine.schemas import (
    Attribution,
    DesiredGoal,
    DesiredState,
    EngineInput,
    EngineOutput,
    NeedScore,
    Situation,
    TemporalFocus,
)

ROOT = Path(__file__).resolve().parents[2]
NLI_DIR = ROOT / "data/models/MoritzLaurer__multilingual-MiniLMv2-L6-mnli-xnli"
EMB_DIR = ROOT / "data/models/sentence-transformers__all-MiniLM-L6-v2"
FALLBACK = ROOT / "data/processed/appraisal/fallback.joblib"

EVENT_TEMPLATE = "This text is about {}."
NEED_TEMPLATE = "The writer needs {}."
AGENT_TEMPLATE = "What happened is caused by {}."
GOAL_TEMPLATE = "The writer wants {}."

EVENT_DESC: dict[EventType, str] = {
    EventType.LOSS_OR_FAILURE: "a loss or a failure",
    EventType.OWN_MISTAKE_OR_WRONGDOING: "the writer's own mistake or wrongdoing",
    EventType.CONFLICT_OR_MISTREATMENT: "a conflict or being mistreated by someone",
    EventType.UNFAIRNESS: "something unfair",
    EventType.THREAT_OR_DANGER: "a threat or danger",
    EventType.UNCERTAINTY_ABOUT_FUTURE: "worry about an uncertain future",
    EventType.ACHIEVEMENT_OR_SUCCESS: "an achievement or a success",
    EventType.POSITIVE_ANTICIPATION: "looking forward to something good",
    EventType.PLEASANT_EVENT: "a pleasant event",
    EventType.SOCIAL_CONNECTION: "closeness, kindness or gratitude between people",
    EventType.REJECTION_OR_LONELINESS: "rejection or loneliness",
    EventType.REMINISCENCE: "remembering the past",
    EventType.SURPRISE_EVENT: "a surprising event",
    EventType.ROUTINE_OR_BOREDOM: "routine or boredom",
    EventType.HEALTH_OR_BODY: "health or the body",
    EventType.OTHER: "something else",
}
NEED_DESC: dict[NeedType, str] = {
    NeedType.EMOTIONAL_SUPPORT: "emotional support and comfort",
    NeedType.PRACTICAL_SOLUTION: "a practical solution or advice",
    NeedType.INFORMATION: "information or an answer",
    NeedType.CHANGE_OF_SCENE: "a change of scene or a distraction",
    NeedType.REST: "rest or sleep",
    NeedType.CELEBRATION: "to celebrate and share good news",
    NeedType.CONNECTION: "company and connection with other people",
}
AGENT_DESC: dict[Agent, str] = {
    Agent.SELF: "the writer themselves",
    Agent.OTHER: "another person",
    Agent.CIRCUMSTANCE: "circumstances, luck or nobody in particular",
}
GOAL_DESC: dict[DesiredGoal, str] = {
    DesiredGoal.CALM: "to feel calm", DesiredGoal.REST: "to rest", DesiredGoal.RELIEF: "relief",
    DesiredGoal.JOY: "to feel joy", DesiredGoal.EXCITEMENT: "to feel excited",
    DesiredGoal.NOVELTY: "something new", DesiredGoal.CONNECTION: "to be with other people",
    DesiredGoal.SOLITUDE: "to be alone", DesiredGoal.CONTROL: "to be in control",
    DesiredGoal.SAFETY: "to be safe", DesiredGoal.UNDERSTANDING: "to be understood",
    DesiredGoal.CHANGE: "change",
}
# Keyword shortcut for desired goals (checked before NLI; documented heuristic).
GOAL_KEYWORDS: dict[DesiredGoal, tuple[str, ...]] = {
    DesiredGoal.CALM: ("calm", "peace", "relax", "settle down", "chill"),
    DesiredGoal.REST: ("rest", "sleep", "nap", "break"),
    DesiredGoal.RELIEF: ("relief", "relieved"),
    DesiredGoal.JOY: ("happy", "happier", "joy", "cheer"),
    DesiredGoal.EXCITEMENT: ("excited", "excitement", "thrill"),
    DesiredGoal.NOVELTY: ("something new", "new experience", "adventure"),
    DesiredGoal.CONNECTION: ("company", "friends", "be with", "talk to someone", "connect"),
    DesiredGoal.SOLITUDE: ("alone", "be left alone", "solitude", "quiet time"),
    DesiredGoal.CONTROL: ("in control", "control"),
    DesiredGoal.SAFETY: ("safe", "safety"),
    DesiredGoal.UNDERSTANDING: ("understood", "understand me", "listened to", "heard"),
    DesiredGoal.CHANGE: ("change", "different", "start over", "fresh start"),
}
POSITIVE = {EventType.ACHIEVEMENT_OR_SUCCESS, EventType.POSITIVE_ANTICIPATION, EventType.PLEASANT_EVENT,
            EventType.SOCIAL_CONNECTION}
NEGATIVE = {EventType.LOSS_OR_FAILURE, EventType.OWN_MISTAKE_OR_WRONGDOING, EventType.CONFLICT_OR_MISTREATMENT,
            EventType.UNFAIRNESS, EventType.THREAT_OR_DANGER, EventType.UNCERTAINTY_ABOUT_FUTURE,
            EventType.REJECTION_OR_LONELINESS, EventType.HEALTH_OR_BODY}
# Controllability heuristic (documented, not learned): who caused it decides how much the writer can steer it.
CONTROL_BY_AGENT = {Agent.SELF: 0.3, Agent.CIRCUMSTANCE: -0.3, Agent.OTHER: -0.1, Agent.UNKNOWN: 0.0}
# Surface emotion words -> canonical label, for attributions only.
EMOTION_WORDS = {
    "anxious": "nervousness", "nervous": "nervousness", "worried": "nervousness", "sad": "sadness",
    "angry": "anger", "furious": "anger", "annoyed": "annoyance", "afraid": "fear", "scared": "fear",
    "happy": "joy", "glad": "joy", "excited": "excitement", "proud": "pride", "ashamed": "embarrassment",
    "embarrassed": "embarrassment", "guilty": "remorse", "lonely": "sadness", "disappointed": "disappointment",
    "grateful": "gratitude", "jealous": "annoyance", "relieved": "relief", "calm": "relief", "upset": "sadness",
    "depressed": "sadness", "miserable": "sadness", "thrilled": "excitement", "confused": "confusion",
    "curious": "curiosity", "disgusted": "disgust", "grief": "grief", "overjoyed": "joy",
}
_SENT_SPLIT = re.compile(r"(?<=[.!?։])\s+")


def models_available() -> bool:
    return NLI_DIR.exists()


def _claude_appraisal(text: str, **kwargs: Any) -> dict[str, Any]:
    """Optional LLM appraisal (Claude API). Stub: off by default, never called unless WIYMOOD_USE_CLAUDE=1."""
    raise NotImplementedError("Claude appraisal path is a stub (zero-money rule); use the local path.")


def _use_claude() -> bool:
    return os.environ.get("WIYMOOD_USE_CLAUDE", "0").lower() in {"1", "true", "yes"}


def _entropy_certainty(p: np.ndarray) -> float:
    p = p / max(p.sum(), 1e-12)
    h = -float(np.sum(p * np.log(p + 1e-12)))
    return float(np.clip(1.0 - h / math.log(len(p)), 0.0, 1.0))


class AppraisalEngine(Engine):
    name = "appraisal"
    version = "0.1.0"
    requires = frozenset({"text"})
    languages = None

    def __init__(self) -> None:
        self._zs: Any = None
        self._fb: Any = None
        self._emb: Any = None
        self._fb_tried = False

    # ---- lazy loading
    def _pipe(self) -> Any:
        if self._zs is None:
            from transformers import pipeline

            self._zs = pipeline("zero-shot-classification", model=str(NLI_DIR), device=-1)
        return self._zs

    def _fallback(self) -> tuple[Any, Any] | None:
        if not self._fb_tried:
            self._fb_tried = True
            if FALLBACK.exists() and EMB_DIR.exists():
                import joblib
                from sentence_transformers import SentenceTransformer

                self._fb = joblib.load(FALLBACK)
                self._emb = SentenceTransformer(str(EMB_DIR), device="cpu")
        return (self._fb, self._emb) if self._fb is not None else None

    # ---- zero-shot helpers
    def _zero_shot(self, text: str, desc: dict[Any, str], template: str, multi: bool) -> dict[Any, float]:
        inv = {v: k for k, v in desc.items()}
        res = self._pipe()(text, candidate_labels=list(desc.values()), hypothesis_template=template,
                           multi_label=multi)
        return {inv[lab]: float(s) for lab, s in zip(res["labels"], res["scores"], strict=True)}

    def _fallback_probs(self, text: str) -> tuple[dict[EventType, float], dict[str, float]] | None:
        fb = self._fallback()
        if fb is None:
            return None
        clf, emb = fb
        x = emb.encode([text], normalize_embeddings=True)
        probs = clf.predict_proba(x)[0]
        events = {EventType(c): float(p) for c, p in zip(clf.classes_, probs, strict=True)}
        return events, {}

    def _evidence(self, text: str, event: EventType) -> list[str]:
        sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
        if len(sents) <= 1:
            sents = [c.text for c in clause_mod.analyze_clauses(text)] or sents
        if len(sents) <= 1:
            return [text.strip()[:200]]
        pipe = self._pipe()
        res = pipe(sents, candidate_labels=[EVENT_DESC[event]], hypothesis_template=EVENT_TEMPLATE, multi_label=True)
        if isinstance(res, dict):
            res = [res]
        best = max(range(len(sents)), key=lambda i: res[i]["scores"][0])
        return [sents[best][:200]]

    def _desired(self, clauses: list[clause_mod.Clause]) -> DesiredState | None:
        want = [c for c in clauses if c.is_desire and not c.negated]
        if not want:
            return None
        goals: list[DesiredGoal] = []
        for c in want:
            low = c.text.lower()
            hit = [g for g, kws in GOAL_KEYWORDS.items() if any(re.search(rf"\b{re.escape(k)}", low) for k in kws)]
            if not hit:
                zs = self._zero_shot(c.text, GOAL_DESC, GOAL_TEMPLATE, multi=False)
                top, p = max(zs.items(), key=lambda kv: kv[1])
                hit = [top] if p >= 0.5 else []
            goals += [g for g in hit if g not in goals]
        return DesiredState(goals=goals, evidence_spans=[c.text for c in want]) if goals else None

    @staticmethod
    def _attributions(clauses: list[clause_mod.Clause]) -> list[Attribution]:
        out: list[Attribution] = []
        for c in clauses:
            if c.is_desire or c.negated:
                continue
            for w in re.findall(r"[a-z]+", c.text.lower()):
                if w in EMOTION_WORDS:
                    try:
                        t = TemporalFocus(c.tense)
                    except ValueError:
                        t = TemporalFocus.UNKNOWN
                    out.append(Attribution(experiencer=c.experiencer, who=c.who, label=EMOTION_WORDS[w],  # type: ignore[arg-type]
                                           p=0.6, time=t, span=c.text))
                    break
        return out

    def analyze(self, inp: EngineInput) -> EngineOutput:
        if not models_available():
            return EngineOutput.empty(self.name, self.version, "nli_model_missing")
        text = inp.text.strip()
        if _use_claude():
            _claude_appraisal(text)  # raises NotImplementedError by design
        try:
            return self._analyze(text, inp)
        except Exception as exc:  # noqa: BLE001  never raise on odd text
            return EngineOutput.empty(self.name, self.version, f"error:{type(exc).__name__}")

    def _analyze(self, text: str, inp: EngineInput) -> EngineOutput:
        zs_ev = self._zero_shot(text, EVENT_DESC, EVENT_TEMPLATE, multi=False)
        probs = {e: zs_ev[e] for e in EventType}
        fb_used = False
        fb_top: tuple[str, float] | None = None
        english = clause_mod.detect_language(text) == "en"
        if english:
            fb = self._fallback_probs(text)
            if fb is not None:
                fb_used = True
                fbp = fb[0]
                probs = {e: 0.5 * zs_ev[e] + 0.5 * fbp.get(e, 0.0) for e in EventType}
                fb_top = (max(fbp, key=fbp.get).value, max(fbp.values()))  # type: ignore[arg-type]
        total = sum(probs.values()) or 1.0
        probs = {e: v / total for e, v in probs.items()}
        ranking = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        event, p_event = ranking[0]
        arr = np.array([probs[e] for e in EventType])
        certainty = _entropy_certainty(arr)
        goal_cong = float(np.clip(sum(probs[e] for e in POSITIVE) - sum(probs[e] for e in NEGATIVE), -1.0, 1.0))

        ag = self._zero_shot(text, AGENT_DESC, AGENT_TEMPLATE, multi=False)
        agent = max(ag, key=ag.get)  # type: ignore[arg-type]
        if fb_top is not None and fb_top[1] > 0.5:
            # weak ED prior: only used when the zero-shot agent is not decisive
            prior = next((a for lab, a in ED_TO_AGENT.items() if ED_TO_EVENT[lab].value == fb_top[0]), None)
            if prior is not None and ag[agent] < 0.6:
                agent = prior
        needs_zs = self._zero_shot(text, NEED_DESC, NEED_TEMPLATE, multi=True)
        needs = [NeedScore(type=n, p=float(np.clip(needs_zs[n], 0, 1))) for n in NeedType]

        clauses = clause_mod.analyze_clauses(text)
        desired = self._desired(clauses)
        situation = Situation(
            event_type=event, event_type_p=float(p_event), agent=agent, goal_congruence=goal_cong,
            certainty=certainty, controllability=CONTROL_BY_AGENT[agent], evidence_spans=self._evidence(text, event),
        )
        sarcasm = clause_mod.sarcasm_cues(text)
        return EngineOutput(
            engine=self.name, version=self.version, features={
                "event_ranking": [(e.value, round(p, 4)) for e, p in ranking],
                "agent_probs": {a.value: round(p, 4) for a, p in ag.items()},
                "fallback_used": fb_used, "sarcasm_cues": sarcasm,
            },
            confidence=float(np.clip(p_event * certainty, 0, 1)), coverage=1.0, situation=situation, needs=needs,
            safety=safety.screen(text), attributions=self._attributions(clauses), desired_state=desired,
            sarcasm_suspected=sarcasm >= 0.5,
        )
