"""Reproducible, licence-aware dataset fetcher (brief §3).

Reads ``config/data_sources.toml``, downloads every ``auto`` source into ``data/raw/<id>/``,
snapshots ``hf_dataset`` sources from the Hugging Face Hub at a pinned revision, and reports
``manual`` sources (never downloaded) as present or missing. Every file is sha256-hashed and
checked against ``config/data_sources.lock.json``; only ``--record-checksums`` writes that file.

Usage::

    uv run python scripts/fetch_data.py --list
    uv run python scripts/fetch_data.py --dry-run
    uv run python scripts/fetch_data.py --only emobank,goemotions
    uv run python scripts/fetch_data.py --commercial-only --record-checksums
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "config" / "data_sources.toml"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw"

LICENCE_CLASSES = frozenset(
    {"commercial-OK", "research-only", "needs-licence", "unclear", "unverified"}
)
ACCESS_KINDS = frozenset({"auto", "manual", "hf_dataset"})
ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz")
RESERVED_NAMES = frozenset({"SOURCE.json"})

_ID_RE = re.compile(r"^[a-z0-9_]+$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

_COMMON_KEYS = {
    "id": str,
    "name": str,
    "licence": str,
    "licence_class": str,
    "access": str,
    "homepage": str,
    "verified": bool,
    "instructions": str,
}
_OPTIONAL_KEYS = {"version": str, "licence_url": str, "notes": str}
_ACCESS_KEYS: dict[str, dict[str, type]] = {
    "auto": {"files": list},
    "manual": {"expected": list},
    "hf_dataset": {"repo_id": str, "revision": str},
}
_FILE_KEYS = {"url": str, "dest": str, "extract": bool}


class ManifestError(ValueError):
    """The manifest is malformed. The message names the source and the offending key."""


@dataclass(frozen=True)
class FileSpec:
    url: str
    dest: str
    extract: bool


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    licence: str
    licence_class: str
    access: str
    homepage: str
    verified: bool
    instructions: str
    version: str = ""
    licence_url: str = ""
    notes: str = ""
    files: tuple[FileSpec, ...] = ()
    expected: tuple[str, ...] = ()
    repo_id: str = ""
    revision: str = ""

    @property
    def revision_pinned(self) -> bool:
        return bool(_SHA_RE.match(self.revision))


# --------------------------------------------------------------------------- manifest


def safe_relative_path(value: str) -> PurePosixPath:
    """Return ``value`` as a relative POSIX path that cannot leave its base directory.

    Raises ``ValueError`` for empty, absolute, backslash, drive-letter or ``..`` paths.
    """
    if not value or value.strip() != value:
        raise ValueError(f"empty or padded path {value!r}")
    if "\\" in value or ":" in value or "\x00" in value:
        raise ValueError(f"path {value!r} contains a backslash, colon or NUL")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError(f"path {value!r} is absolute")
    if any(part == ".." for part in path.parts):
        raise ValueError(f"path {value!r} contains '..'")
    if not path.parts or path.parts == (".",):
        raise ValueError(f"path {value!r} is empty")
    return path


def _check_keys(where: str, table: dict[str, Any], schema: dict[str, type]) -> None:
    unknown = sorted(set(table) - set(schema))
    if unknown:
        raise ManifestError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(schema)}")
    for key, value in table.items():
        expected = schema[key]
        # bool is a subclass of int; neither field type here is int, so isinstance suffices.
        if not isinstance(value, expected):
            raise ManifestError(
                f"{where}: key {key!r} must be {expected.__name__}, got {type(value).__name__}"
            )


def _parse_file(where: str, raw: Any) -> FileSpec:
    if not isinstance(raw, dict):
        raise ManifestError(f"{where}: each [[source.files]] entry must be a table")
    _check_keys(where, raw, _FILE_KEYS)
    missing = sorted(set(_FILE_KEYS) - set(raw))
    if missing:
        raise ManifestError(f"{where}: missing key(s) {missing}")
    url, dest, extract = raw["url"], raw["dest"], raw["extract"]
    if not url.startswith(("https://", "http://")):
        raise ManifestError(f"{where}: url {url!r} must be http(s)")
    try:
        path = safe_relative_path(dest)
    except ValueError as exc:
        raise ManifestError(f"{where}: dest escapes the source directory: {exc}") from None
    if path.name in RESERVED_NAMES or path.name.endswith((".part", ".bad", ".extracted")):
        raise ManifestError(f"{where}: dest {dest!r} uses a reserved file name")
    if extract and not dest.lower().endswith(ARCHIVE_SUFFIXES):
        raise ManifestError(
            f"{where}: extract = true needs a dest ending in {', '.join(ARCHIVE_SUFFIXES)}"
        )
    return FileSpec(url=url, dest=str(path), extract=extract)


def _parse_source(index: int, raw: Any) -> Source:
    if not isinstance(raw, dict):
        raise ManifestError(f"source #{index}: must be a table")
    sid = raw.get("id")
    where = f"source {sid!r}" if isinstance(sid, str) else f"source #{index}"
    access = raw.get("access")
    if access not in ACCESS_KINDS:
        raise ManifestError(f"{where}: access must be one of {sorted(ACCESS_KINDS)}, got {access!r}")
    schema = {**_COMMON_KEYS, **_OPTIONAL_KEYS, **_ACCESS_KEYS[access]}
    try:
        _check_keys(where, raw, schema)
    except ManifestError:
        others = {k for kind, keys in _ACCESS_KEYS.items() if kind != access for k in keys}
        bad = sorted(set(raw) & others)
        if bad:
            raise ManifestError(
                f"{where}: unknown key(s) {bad} for access = {access!r} (they belong to another "
                f"access kind)"
            ) from None
        raise
    missing = sorted((set(_COMMON_KEYS) | set(_ACCESS_KEYS[access])) - set(raw))
    if missing:
        raise ManifestError(f"{where}: missing key(s) {missing}")
    if not _ID_RE.match(raw["id"]):
        raise ManifestError(f"{where}: id must match [a-z0-9_]+")
    if raw["licence_class"] not in LICENCE_CLASSES:
        raise ManifestError(
            f"{where}: licence_class must be one of {sorted(LICENCE_CLASSES)}, "
            f"got {raw['licence_class']!r}"
        )

    files: tuple[FileSpec, ...] = ()
    expected: tuple[str, ...] = ()
    if access == "auto":
        if not raw["files"]:
            raise ManifestError(f"{where}: an auto source needs at least one [[source.files]]")
        files = tuple(
            _parse_file(f"{where} file #{i}", f) for i, f in enumerate(raw["files"], start=1)
        )
        dests = [f.dest for f in files]
        if len(set(dests)) != len(dests):
            raise ManifestError(f"{where}: duplicate dest in files")
    elif access == "manual":
        if not raw["expected"]:
            raise ManifestError(f"{where}: expected must list at least one path")
        checked = []
        for item in raw["expected"]:
            if not isinstance(item, str):
                raise ManifestError(f"{where}: expected entries must be strings")
            try:
                checked.append(str(safe_relative_path(item)))
            except ValueError as exc:
                raise ManifestError(f"{where}: bad expected path: {exc}") from None
        expected = tuple(checked)
        if not raw["instructions"].strip():
            raise ManifestError(f"{where}: a manual source needs instructions")
    else:
        if not re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", raw["repo_id"]):
            raise ManifestError(f"{where}: repo_id must look like 'owner/name'")
        if not raw["revision"].strip():
            raise ManifestError(f"{where}: revision must not be empty (prefer a commit sha)")

    return Source(
        id=raw["id"],
        name=raw["name"],
        licence=raw["licence"],
        licence_class=raw["licence_class"],
        access=access,
        homepage=raw["homepage"],
        verified=raw["verified"],
        instructions=raw["instructions"],
        version=raw.get("version", ""),
        licence_url=raw.get("licence_url", ""),
        notes=raw.get("notes", ""),
        files=files,
        expected=expected,
        repo_id=raw.get("repo_id", ""),
        revision=raw.get("revision", ""),
    )


def parse_manifest(data: dict[str, Any]) -> list[Source]:
    """Validate a decoded manifest and return its sources in file order."""
    unknown = sorted(set(data) - {"source"})
    if unknown:
        raise ManifestError(f"manifest: unknown key(s) {unknown} at top level; only [[source]]")
    raw_sources = data.get("source")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ManifestError("manifest: needs at least one [[source]] table")
    sources = [_parse_source(i, raw) for i, raw in enumerate(raw_sources, start=1)]
    seen: set[str] = set()
    for source in sources:
        if source.id in seen:
            raise ManifestError(f"manifest: duplicate source id {source.id!r}")
        seen.add(source.id)
    return sources


def load_manifest(path: Path) -> list[Source]:
    """Read and validate the TOML manifest at ``path``."""
    try:
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ManifestError(f"manifest not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError(f"manifest {path} is not valid TOML: {exc}") from None
    return parse_manifest(data)
