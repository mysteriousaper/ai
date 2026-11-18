"""Write the JSON Schemas of the Engine's public models to docs/schemas/.

Usage:
    uv run python scripts/export_schemas.py            # (re)write docs/schemas/*.schema.json
    uv run python scripts/export_schemas.py --check    # exit 1 if the files are out of date
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow running without installing the package
    sys.path.insert(0, str(REPO_ROOT))

from engine.schemas import json_schemas  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "docs" / "schemas"


def render(schema: dict[str, object]) -> str:
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="verify only, write nothing")
    args = parser.parse_args(argv)

    schemas = {f"{stem}.schema.json": render(s) for stem, s in json_schemas().items()}
    if args.check:
        on_disk = {p.name for p in args.out.glob("*.schema.json")} if args.out.is_dir() else set()
        stale = sorted(
            name
            for name, text in schemas.items()
            if not (args.out / name).is_file()
            or (args.out / name).read_text(encoding="utf-8") != text
        )
        extra = sorted(on_disk - set(schemas))
        for name in stale:
            print(f"out of date: {name}")
        for name in extra:
            print(f"not produced by any model: {name}")
        return 1 if stale or extra else 0

    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in schemas.items():
        (args.out / name).write_text(text, encoding="utf-8")
        print(f"wrote {args.out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
