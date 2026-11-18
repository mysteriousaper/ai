"""Write JSON Schemas for the public models to docs/schema/."""

from __future__ import annotations

import json
from pathlib import Path

from engine.schemas import SCHEMA_VERSION, EmotionalStateObject, EngineInput, EngineOutput

OUT = Path(__file__).resolve().parents[1] / "docs" / "schema"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, model in {"emotional_state_object": EmotionalStateObject, "engine_output": EngineOutput,
                        "engine_input": EngineInput}.items():
        path = OUT / f"{name}.v{SCHEMA_VERSION}.json"
        path.write_text(json.dumps(model.model_json_schema(), indent=2) + "\n")
        print("wrote", path.relative_to(OUT.parents[1]))


if __name__ == "__main__":
    main()
