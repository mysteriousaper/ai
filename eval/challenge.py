"""Hand-labelled English challenge set (negation, contrast, desire, sarcasm...) scored per category.

`run_challenge(engine)` scores any Engine on current-speaker valence sign and top-emotion hit@3,
skipping fields the engine does not output. CLI: `uv run python eval/challenge.py lexicon [stylometric|transformer|all]`.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.schemas import EngineInput

PATH = Path(__file__).with_name("challenge_set.jsonl")
NEUTRAL_BAND = 0.1  # |valence| below this counts as sign 0


def load_challenge(path: Path = PATH) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sign(v: float) -> int:
    return 0 if abs(v) < NEUTRAL_BAND else (1 if v > 0 else -1)


def run_challenge(engine, items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Return {category: {n, valence_acc, valence_n, hit3, hit3_n}} plus an 'ALL' row."""
    items = items if items is not None else load_challenge()
    acc: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(("n", "v_ok", "v_n", "h_ok", "h_n"), 0))
    for it in items:
        out = engine(EngineInput(text=it["text"]))
        exp = it["expected"]
        for key in (it["category"], "ALL"):
            a = acc[key]
            a["n"] += 1
            if out.vad is not None:
                a["v_n"] += 1
                a["v_ok"] += int(_sign(out.vad.valence) == exp["valence_sign"])
            if out.emotions and exp.get("top_emotions"):
                top3 = [k for k, _ in sorted(out.emotions.items(), key=lambda kv: kv[1], reverse=True)[:3]]
                a["h_n"] += 1
                a["h_ok"] += int(any(e in top3 for e in exp["top_emotions"]))
    return {
        k: {"n": a["n"], "valence_acc": a["v_ok"] / a["v_n"] if a["v_n"] else None, "valence_n": a["v_n"],
            "hit3": a["h_ok"] / a["h_n"] if a["h_n"] else None, "hit3_n": a["h_n"]}
        for k, a in acc.items()
    }


def format_table(name: str, res: dict[str, Any]) -> str:
    f = lambda x: "  -  " if x is None else f"{x:.2f}"
    rows = [f"{name}: category | n | valence_acc | hit@3", "-" * 50]
    for cat in sorted(k for k in res if k != "ALL") + ["ALL"]:
        r = res[cat]
        rows.append(f"{cat:<13} | {r['n']:>2} | {f(r['valence_acc'])} | {f(r['hit3'])}")
    return "\n".join(rows)


def _engines(which: str) -> dict[str, Any]:
    from engine.engines.lexicon import LexiconEngine
    from engine.engines.stylometric import StylometricEngine

    avail: dict[str, Any] = {"lexicon": LexiconEngine, "stylometric": StylometricEngine}
    try:
        from engine.engines.transformer import TransformerEngine

        avail["transformer"] = TransformerEngine
    except Exception as e:  # noqa: BLE001 — optional engine, may be missing or mid-development
        print(f"(transformer unavailable: {e})")
    names = list(avail) if which == "all" else [which]
    out: dict[str, Any] = {}
    for n in names:
        try:
            out[n] = avail[n]()
        except Exception as e:  # noqa: BLE001 — e.g. model files not downloaded
            print(f"({n} could not be built: {e})")
    return out


def main(argv: list[str]) -> None:
    which = argv[0] if argv else "lexicon"
    for name, eng in _engines(which).items():
        try:
            print(format_table(name, run_challenge(eng)), end="\n\n")
        except Exception as e:  # noqa: BLE001
            print(f"({name} failed: {e})")


if __name__ == "__main__":
    main(sys.argv[1:])
