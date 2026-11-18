"""Crisis-safety rule. Lives in the CORE, not in adapters (brief §10).

Phase 2 ships a conservative phrase baseline: it prefers false positives over misses
(e.g. "I would never kill myself" still flags CONCERN-or-higher on purpose). Phase 3 adds the
LLM appraisal's own safety judgement, and the stricter of the two always wins (`merge`).
This is a screening heuristic, not a clinical assessment.
"""

from __future__ import annotations

import re

from engine.schemas import SafetyFlag, SafetyLevel

_URGENT = [
    r"\bkill(ing)?\s+my\s*self\b", r"\bend(ing)?\s+(it\s+all|my\s+life)\b", r"\bsuicid(e|al)\b",
    r"\bwant(ed)?\s+to\s+die\b", r"\bwish\s+i\s+(was|were)\s+dead\b", r"\bbetter\s+off\s+dead\b",
    r"\bno\s+reason\s+to\s+live\b", r"\b(hurt|harm|cut)(ting)?\s+my\s*self\b", r"\bself[\s-]?harm\b",
    r"\bdon'?t\s+want\s+to\s+(live|be\s+alive|exist)\b", r"\boverdose\b",
    r"\b(he|she|they|someone)\s+(is\s+going\s+to|will|wants\s+to)\s+(hurt|kill)\s+me\b",
]
_CONCERN = [
    r"\bhopeless\b", r"\bcan'?t\s+(go\s+on|take\s+(it|this)\s+any\s*more)\b", r"\bworthless\b",
    r"\bno\s+way\s+out\b", r"\bgive\s+up\s+on\s+everything\b", r"\bnobody\s+would\s+(care|notice)\b",
    r"\bi'?m\s+a\s+burden\b", r"\bdisappear\s+forever\b",
]
_URGENT_RE = [re.compile(p, re.IGNORECASE) for p in _URGENT]
_CONCERN_RE = [re.compile(p, re.IGNORECASE) for p in _CONCERN]

_ORDER = {SafetyLevel.NONE: 0, SafetyLevel.CONCERN: 1, SafetyLevel.URGENT: 2}


def screen(text: str) -> SafetyFlag:
    urgent = [p.pattern for p in _URGENT_RE if p.search(text)]
    if urgent:
        return SafetyFlag(level=SafetyLevel.URGENT, reasons=[f"phrase:{r}" for r in urgent])
    concern = [p.pattern for p in _CONCERN_RE if p.search(text)]
    if concern:
        return SafetyFlag(level=SafetyLevel.CONCERN, reasons=[f"phrase:{r}" for r in concern])
    return SafetyFlag()


def merge(*flags: SafetyFlag | None) -> SafetyFlag:
    """Combine flags from several sources: the most severe level wins, reasons are pooled."""
    present = [f for f in flags if f is not None]
    if not present:
        return SafetyFlag()
    level = max((f.level for f in present), key=_ORDER.__getitem__)
    reasons = [r for f in present for r in f.reasons]
    return SafetyFlag(level=level, reasons=reasons)
