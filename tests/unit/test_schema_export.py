"""The committed JSON Schemas in docs/schemas/ must match the Pydantic models (drift test)."""

import json
import subprocess
import sys
from pathlib import Path

from engine.schemas import json_schemas

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "docs" / "schemas"
SCRIPT = REPO_ROOT / "scripts" / "export_schemas.py"


def test_committed_schemas_match_models() -> None:
    expected = json_schemas()
    committed = {p.name.removesuffix(".schema.json") for p in SCHEMA_DIR.glob("*.schema.json")}
    assert committed == set(expected), "run: uv run python scripts/export_schemas.py"
    for stem, schema in expected.items():
        on_disk = json.loads((SCHEMA_DIR / f"{stem}.schema.json").read_text(encoding="utf-8"))
        assert on_disk == schema, f"{stem} drifted; run: uv run python scripts/export_schemas.py"


def test_export_script_writes_files(tmp_path: Path) -> None:
    subprocess.run([sys.executable, str(SCRIPT), "--out", str(tmp_path)], check=True)
    for stem, schema in json_schemas().items():
        text = (tmp_path / f"{stem}.schema.json").read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert json.loads(text) == schema


def test_export_script_check_mode(tmp_path: Path) -> None:
    stale = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(tmp_path), "--check"], check=False
    )
    assert stale.returncode == 1
    subprocess.run([sys.executable, str(SCRIPT), "--out", str(tmp_path)], check=True)
    fresh = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(tmp_path), "--check"], check=False
    )
    assert fresh.returncode == 0


def test_schema_rejects_unknown_emotion_labels_and_extra_fields() -> None:
    eo = json_schemas()["engine_output"]
    assert eo["additionalProperties"] is False
    names = eo["properties"]["emotions"]["propertyNames"]
    assert names == {"$ref": "#/$defs/EmotionLabel"}
    assert len(eo["$defs"]["EmotionLabel"]["enum"]) == 28
    eso = json_schemas()["emotional_state_object"]
    assert eso["properties"]["schema_version"]["const"] == "0.1"
