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

import argparse
import hashlib
import http.client
import json
import os
import re
import stat
import sys
import tarfile
import tomllib
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "config" / "data_sources.toml"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw"

LICENCE_CLASSES = frozenset(
    {"commercial-OK", "research-only", "needs-licence", "unclear", "unverified"}
)
ACCESS_KINDS = frozenset({"auto", "manual", "hf_dataset"})
ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz")
RESERVED_NAMES = frozenset({"SOURCE.json"})
SOURCE_META = "SOURCE.json"
LOCK_VERSION = 1
CHUNK_SIZE = 1 << 16
DEFAULT_TIMEOUT = 60.0
USER_AGENT = "emotion-engine-fetch-data/0.1 (+python-urllib)"

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
    """The manifest (or lock file) is malformed. The message names the source and the key."""


class FetchError(RuntimeError):
    """A download, checksum or extraction step failed. The message says what and where."""


class ChecksumMismatchError(FetchError):
    """A file's sha256 differs from the lock file. The file is kept as ``<dest>.bad``."""


class UnsafeArchiveError(FetchError):
    """An archive member would land outside the target directory, or is a link/device."""


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


def lock_path_for(manifest: Path) -> Path:
    """``config/data_sources.toml`` -> ``config/data_sources.lock.json``."""
    return manifest.with_name(f"{manifest.stem}.lock.json")


# --------------------------------------------------------------------------- lock file

Lock = dict[str, dict[str, str]]
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def load_lock(path: Path) -> Lock:
    """Read ``{source_id: {dest: sha256}}`` from the lock file; a missing file is empty."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise ManifestError(f"lock file {path} is not valid JSON: {exc}") from None
    if not isinstance(raw, dict) or set(raw) != {"version", "files"}:
        raise ManifestError(f"lock file {path}: expected keys 'version' and 'files'")
    if raw["version"] != LOCK_VERSION:
        raise ManifestError(f"lock file {path}: unsupported version {raw['version']!r}")
    files = raw["files"]
    if not isinstance(files, dict):
        raise ManifestError(f"lock file {path}: 'files' must be an object")
    lock: Lock = {}
    for sid, entries in files.items():
        if not isinstance(entries, dict):
            raise ManifestError(f"lock file {path}: entry {sid!r} must be an object")
        for dest, digest in entries.items():
            if not isinstance(digest, str) or not _HEX64_RE.match(digest):
                raise ManifestError(f"lock file {path}: {sid}/{dest} is not a lowercase sha256")
        lock[sid] = dict(entries)
    return lock


def save_lock(path: Path, lock: Lock) -> None:
    """Write the lock file atomically with sorted keys (stable diffs)."""
    payload = {
        "version": LOCK_VERSION,
        "files": {sid: dict(sorted(lock[sid].items())) for sid in sorted(lock) if lock[sid]},
    }
    _write_json_atomic(Path(path), payload)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------- hashing / paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_inside(base: Path, relative: str) -> Path:
    """Join ``relative`` onto ``base`` and refuse anything that resolves outside ``base``.

    Resolving follows symlinks, so a symlinked sub-directory pointing elsewhere is caught too.
    """
    rel = safe_relative_path(relative)
    base_resolved = base.resolve()
    target = (base_resolved / rel).resolve()
    if target != base_resolved and not target.is_relative_to(base_resolved):
        raise UnsafeArchiveError(f"path {relative!r} escapes {base}")
    return target


# --------------------------------------------------------------------------- download


def download(
    url: str,
    dest: Path,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    expected_sha256: str | None = None,
) -> str:
    """Stream ``url`` to ``dest`` via ``dest.part`` and return the sha256 of the bytes.

    The ``.part`` file is removed on any failure (including Ctrl-C). If ``expected_sha256``
    is given and differs, the bytes are kept as ``dest.bad`` and ``ChecksumMismatchError``
    is raised; ``dest`` itself is never replaced by bad data.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    received = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, open(part, "wb") as out:
            length_header = response.headers.get("Content-Length")
            while chunk := response.read(CHUNK_SIZE):
                digest.update(chunk)
                out.write(chunk)
                received += len(chunk)
            if length_header is not None and received != int(length_header):
                raise FetchError(
                    f"incomplete download of {url}: got {received} of {length_header} bytes"
                )
    except urllib.error.HTTPError as exc:
        part.unlink(missing_ok=True)
        raise FetchError(f"HTTP {exc.code} {exc.reason} for {url}") from None
    except urllib.error.URLError as exc:
        part.unlink(missing_ok=True)
        raise FetchError(f"cannot reach {url}: {exc.reason}") from None
    except TimeoutError:
        part.unlink(missing_ok=True)
        raise FetchError(f"timed out after {timeout:g}s while fetching {url}") from None
    except http.client.IncompleteRead as exc:
        part.unlink(missing_ok=True)
        raise FetchError(
            f"incomplete download of {url}: connection closed after {received + len(exc.partial)}"
            " bytes"
        ) from None
    except BaseException:
        part.unlink(missing_ok=True)
        raise

    hexdigest = digest.hexdigest()
    if expected_sha256 is not None and hexdigest != expected_sha256:
        bad = dest.with_name(dest.name + ".bad")
        os.replace(part, bad)
        raise ChecksumMismatchError(
            f"sha256 mismatch for {dest.name}: lock says {expected_sha256}, got {hexdigest}; "
            f"kept as {bad}"
        )
    os.replace(part, dest)
    return hexdigest


# --------------------------------------------------------------------------- extraction


def _check_member(name: str, archive: Path) -> None:
    try:
        safe_relative_path(name.rstrip("/") or name)
    except ValueError as exc:
        raise UnsafeArchiveError(f"{archive.name}: unsafe member {name!r}: {exc}") from None


def safe_extract(archive: Path, target: Path) -> list[str]:
    """Extract a .zip / .tar.gz / .tgz into ``target`` after validating every member.

    Rejects absolute paths, ``..``, links that escape (or any zip symlink) and device files
    *before* writing anything; tar extraction additionally runs with ``filter="data"``.
    Returns the member names extracted.
    """
    target.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith(".zip"):
        return _extract_zip(archive, target)
    if name.endswith((".tar.gz", ".tgz")):
        return _extract_tar(archive, target)
    raise FetchError(f"don't know how to extract {archive.name}")


def _extract_zip(archive: Path, target: Path) -> list[str]:
    try:
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
            for info in infos:
                _check_member(info.filename, archive)
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise UnsafeArchiveError(f"{archive.name}: symlink member {info.filename!r}")
                resolve_inside(target, info.filename.rstrip("/"))
            zf.extractall(target)
            return [info.filename for info in infos]
    except zipfile.BadZipFile as exc:
        raise FetchError(f"{archive.name} is not a valid zip: {exc}") from None


def _extract_tar(archive: Path, target: Path) -> list[str]:
    try:
        with tarfile.open(archive, "r:gz") as tf:
            members = tf.getmembers()
            for member in members:
                _check_member(member.name, archive)
                resolve_inside(target, member.name.rstrip("/"))
                if member.isdev():
                    raise UnsafeArchiveError(f"{archive.name}: device member {member.name!r}")
                if member.issym() or member.islnk():
                    link = member.linkname
                    if link.startswith("/"):
                        raise UnsafeArchiveError(
                            f"{archive.name}: absolute link {member.name!r} -> {link!r}"
                        )
                    base = PurePosixPath(member.name).parent if member.issym() else PurePosixPath()
                    joined = os.path.normpath(str(base / link))
                    if joined == ".." or joined.startswith("../"):
                        raise UnsafeArchiveError(
                            f"{archive.name}: link {member.name!r} -> {link!r} escapes"
                        )
            tf.extractall(target, filter="data")
            return [m.name for m in members]
    except tarfile.FilterError as exc:
        raise UnsafeArchiveError(f"{archive.name}: {exc}") from None
    except tarfile.TarError as exc:
        raise FetchError(f"{archive.name} is not a valid tar.gz: {exc}") from None


# --------------------------------------------------------------------------- fetching


@dataclass
class SourceResult:
    """Outcome for one source; ``status`` is one of
    ok | skipped | failed | present | missing | planned."""

    id: str
    access: str
    licence_class: str
    status: str
    detail: str = ""
    unpinned: list[str] = field(default_factory=list)
    sha256s: dict[str, str] = field(default_factory=dict)
    instructions: str = ""

    @property
    def failed(self) -> bool:
        return self.status == "failed"


SnapshotFn = Callable[..., str]


def _now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_previous_shas(src_dir: Path) -> dict[str, str]:
    try:
        meta = json.loads((src_dir / SOURCE_META).read_text(encoding="utf-8"))
        shas = meta.get("sha256s", {})
        return {k: v for k, v in shas.items() if isinstance(k, str) and isinstance(v, str)}
    except (OSError, ValueError, AttributeError):
        return {}


def write_source_meta(
    source: Source, src_dir: Path, urls: Sequence[str], sha256s: dict[str, str], **extra: Any
) -> Path:
    """Write ``SOURCE.json`` (provenance + licence) next to the fetched files."""
    meta: dict[str, Any] = {
        "id": source.id,
        "name": source.name,
        "access": source.access,
        "urls": list(urls),
        "sha256s": dict(sorted(sha256s.items())),
        "licence": source.licence,
        "licence_class": source.licence_class,
        "licence_url": source.licence_url,
        "verified": source.verified,
        "fetched_at": _now_utc(),
        **extra,
    }
    path = src_dir / SOURCE_META
    _write_json_atomic(path, meta)
    return path


def fetch_source(
    source: Source,
    data_dir: Path,
    lock: Lock,
    *,
    record: bool = False,
    dry_run: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    snapshot_download: SnapshotFn | None = None,
) -> SourceResult:
    """Fetch (or check) one source into ``data_dir/<id>/``. Never raises for fetch errors.

    ``lock`` is updated in place only when ``record`` is true and a file has no entry yet;
    an existing entry is never overwritten.
    """
    src_dir = Path(data_dir) / source.id
    try:
        if source.access == "manual":
            return _check_manual(source, src_dir, dry_run=dry_run)
        if source.access == "hf_dataset":
            return _fetch_hf(
                source, src_dir, lock, record=record, dry_run=dry_run, timeout=timeout,
                snapshot_download=snapshot_download,
            )
        return _fetch_auto(source, src_dir, lock, record=record, dry_run=dry_run, timeout=timeout)
    except (FetchError, OSError) as exc:
        return SourceResult(
            source.id, source.access, source.licence_class, "failed", detail=str(exc),
            instructions=source.instructions,
        )


def _fetch_auto(
    source: Source, src_dir: Path, lock: Lock, *, record: bool, dry_run: bool, timeout: float
) -> SourceResult:
    pinned = lock.get(source.id, {})
    previous = _read_previous_shas(src_dir)
    shas: dict[str, str] = {}
    unpinned: list[str] = []
    notes: list[str] = []
    changed = False
    for spec in source.files:
        target = resolve_inside(src_dir, spec.dest)
        expected = pinned.get(spec.dest)
        known = expected or previous.get(spec.dest)
        up_to_date = target.is_file() and known is not None and sha256_file(target) == known
        if dry_run:
            notes.append(f"{spec.dest}: {'up to date' if up_to_date else 'would download'}")
            if expected is None:
                unpinned.append(spec.dest)
            continue
        fetched = not up_to_date
        if up_to_date:
            assert known is not None
            digest = known
        else:
            digest = download(spec.url, target, timeout=timeout, expected_sha256=expected)
            changed = True
        if spec.extract:
            marker = target.with_name(target.name + ".extracted")
            if fetched or not marker.is_file() or marker.read_text().strip() != digest:
                safe_extract(target, target.parent)
                marker.write_text(digest + "\n", encoding="utf-8")
                changed = True
        shas[spec.dest] = digest
        if expected is None:
            if record:
                lock.setdefault(source.id, {})[spec.dest] = digest
            else:
                unpinned.append(spec.dest)

    base = SourceResult(
        source.id, source.access, source.licence_class, "planned", unpinned=unpinned,
        instructions=source.instructions,
    )
    if dry_run:
        base.detail = "; ".join(notes)
        return base
    if changed or not (src_dir / SOURCE_META).is_file():
        write_source_meta(source, src_dir, [f.url for f in source.files], shas)
    base.status = "ok" if changed else "skipped"
    base.sha256s = shas
    base.detail = f"{len(shas)} file(s)" + ("" if changed else ", hashes match")
    return base


def _iter_data_files(src_dir: Path) -> Iterable[tuple[str, Path]]:
    """Yield (relative posix path, path) for real data files, skipping our own bookkeeping."""
    if not src_dir.is_dir():
        return
    for path in sorted(src_dir.rglob("*")):
        rel = path.relative_to(src_dir)
        if not path.is_file() or rel.parts[0] == ".cache":
            continue
        if rel.as_posix() == SOURCE_META or path.name.endswith((".part", ".bad")):
            continue
        yield rel.as_posix(), path


def _fetch_hf(
    source: Source,
    src_dir: Path,
    lock: Lock,
    *,
    record: bool,
    dry_run: bool,
    timeout: float,
    snapshot_download: SnapshotFn | None,
) -> SourceResult:
    url = f"https://huggingface.co/datasets/{source.repo_id}"
    pin_note = "" if source.revision_pinned else f"; revision {source.revision!r} is not a sha"
    result = SourceResult(
        source.id, source.access, source.licence_class, "planned",
        instructions=source.instructions,
    )
    if dry_run:
        result.detail = f"would snapshot {source.repo_id}@{source.revision}{pin_note}"
        return result
    if snapshot_download is None:
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")  # ADR-0007
        from huggingface_hub import snapshot_download as hf_snapshot

        snapshot_download = hf_snapshot
    src_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=source.repo_id,
            repo_type="dataset",
            revision=source.revision,
            local_dir=str(src_dir),
            etag_timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - huggingface_hub raises many types; report all
        raise FetchError(
            f"snapshot of {source.repo_id}@{source.revision} failed: {type(exc).__name__}: {exc}"
        ) from None

    pinned = lock.get(source.id, {})
    shas = {rel: sha256_file(path) for rel, path in _iter_data_files(src_dir)}
    if not shas:
        raise FetchError(f"snapshot of {source.repo_id}@{source.revision} produced no files")
    mismatched = sorted(rel for rel, digest in shas.items() if rel in pinned and pinned[rel] != digest)
    vanished = sorted(set(pinned) - set(shas))
    if mismatched or vanished:
        raise ChecksumMismatchError(
            f"sha256 mismatch for {source.repo_id}: changed {mismatched or '-'}, "
            f"missing {vanished or '-'}"
        )
    for rel, digest in shas.items():
        if rel not in pinned:
            if record:
                lock.setdefault(source.id, {})[rel] = digest
            else:
                result.unpinned.append(rel)
    write_source_meta(
        source, src_dir, [url], shas, repo_id=source.repo_id, revision=source.revision
    )
    result.status = "ok"
    result.sha256s = shas
    result.detail = f"{source.repo_id}@{source.revision[:12]}, {len(shas)} file(s){pin_note}"
    return result


def _check_manual(source: Source, src_dir: Path, *, dry_run: bool) -> SourceResult:
    files = dict(_iter_data_files(src_dir))
    matched: dict[str, Path] = {}
    missing: list[str] = []
    for pattern in source.expected:
        hits = {
            path.relative_to(src_dir).as_posix(): path
            for path in (src_dir.glob(pattern) if src_dir.is_dir() else [])
            if path.relative_to(src_dir).as_posix() in files
        }
        if hits:
            matched.update(hits)
        else:
            missing.append(pattern)
    result = SourceResult(
        source.id, source.access, source.licence_class, "missing" if missing else "present",
        instructions=source.instructions,
    )
    if missing:
        result.detail = f"missing {missing} in {src_dir}"
        return result
    result.sha256s = {rel: sha256_file(path) for rel, path in sorted(matched.items())}
    result.unpinned = sorted(result.sha256s)
    result.detail = f"{len(matched)} file(s) present"
    if not dry_run:
        write_source_meta(source, src_dir, [], result.sha256s, homepage=source.homepage)
    return result


# --------------------------------------------------------------------------- CLI


def select_sources(
    sources: Sequence[Source], only: Sequence[str] | None, commercial_only: bool
) -> tuple[list[Source], list[Source]]:
    """Return (selected, excluded-by-licence). Unknown ``only`` ids raise ``ManifestError``."""
    chosen = list(sources)
    if only:
        known = {s.id for s in sources}
        unknown = sorted(set(only) - known)
        if unknown:
            raise ManifestError(f"--only: unknown source id(s) {unknown}; known: {sorted(known)}")
        chosen = [s for s in chosen if s.id in set(only)]
    if not commercial_only:
        return chosen, []
    return (
        [s for s in chosen if s.licence_class == "commercial-OK"],
        [s for s in chosen if s.licence_class != "commercial-OK"],
    )


def _table(rows: Sequence[Sequence[str]], out: TextIO) -> None:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for n, row in enumerate(rows):
        cells = [cell.ljust(w) for cell, w in zip(row, widths, strict=True)]
        print("  ".join(cells).rstrip(), file=out)
        if n == 0:
            print("  ".join("-" * w for w in widths), file=out)


def print_listing(sources: Sequence[Source], out: TextIO) -> None:
    rows = [["id", "access", "licence_class", "verified", "files", "name"]]
    for s in sources:
        count = str(len(s.files)) if s.access == "auto" else {
            "manual": f"{len(s.expected)} expected", "hf_dataset": f"{s.repo_id}@{s.revision[:8]}"
        }[s.access]
        rows.append([s.id, s.access, s.licence_class, str(s.verified).lower(), count, s.name])
    _table(rows, out)


def _status_label(result: SourceResult) -> str:
    label = result.status.upper() if result.status in {"failed", "missing"} else result.status
    if result.unpinned and result.status in {"ok", "skipped", "planned"}:
        label += " (unpinned)"
    return label


def print_summary(results: Sequence[SourceResult], out: TextIO) -> None:
    rows = [["id", "access", "licence_class", "status", "detail"]]
    for r in results:
        rows.append([r.id, r.access, r.licence_class, _status_label(r), r.detail])
    print(file=out)
    _table(rows, out)

    unpinned = [r for r in results if r.unpinned and r.access != "manual" and not r.failed]
    if unpinned:
        print("\nUnpinned (no lock entry; run with --record-checksums to pin):", file=out)
        for r in unpinned:
            shown = ", ".join(r.unpinned[:5]) + (" ..." if len(r.unpinned) > 5 else "")
            print(f"  {r.id}: {shown}", file=out)

    missing = [r for r in results if r.status == "missing"]
    if missing:
        bar = "!" * 78
        print(f"\n{bar}\nMISSING MANUAL SOURCES ({len(missing)}): these need you.", file=out)
        for r in missing:
            print(f"\n  [{r.id}] {r.detail}\n  -> {r.instructions}", file=out)
        print(bar, file=out)

    failed = [r for r in results if r.failed]
    if failed:
        print(f"\nFAILED ({len(failed)}):", file=out)
        for r in failed:
            print(f"  {r.id}: {r.detail}", file=out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch_data.py",
        description="Fetch datasets and lexicons listed in config/data_sources.toml.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--list", action="store_true", help="list sources and exit")
    parser.add_argument(
        "--only", action="append", metavar="ID[,ID...]", help="fetch only these source ids"
    )
    parser.add_argument("--dry-run", action="store_true", help="show the plan; no network/writes")
    parser.add_argument(
        "--commercial-only", action="store_true", help="only licence_class = commercial-OK"
    )
    parser.add_argument(
        "--record-checksums",
        action="store_true",
        help="add sha256s of files without a lock entry to the lock file",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="seconds per read")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    out: TextIO | None = None,
    err: TextIO | None = None,
    snapshot_download: SnapshotFn | None = None,
) -> int:
    """CLI entry point. Exit codes: 0 ok, 1 an auto/hf source failed, 2 usage/manifest error."""
    out = out or sys.stdout
    err = err or sys.stderr
    args = build_parser().parse_args(argv)
    only = [part.strip() for item in args.only or [] for part in item.split(",") if part.strip()]
    lock_path = lock_path_for(args.manifest)
    try:
        sources = load_manifest(args.manifest)
        selected, excluded = select_sources(sources, only, args.commercial_only)
        lock = load_lock(lock_path)
    except ManifestError as exc:
        print(f"error: {exc}", file=err)
        return 2

    if args.list:
        print_listing(selected, out)
        if excluded:
            print(f"\nexcluded by --commercial-only: {', '.join(s.id for s in excluded)}", file=out)
        return 0

    if excluded:
        print(
            f"--commercial-only: excluded {len(excluded)} source(s) that are not commercial-OK: "
            f"{', '.join(s.id for s in excluded)}",
            file=out,
        )
    if any(not s.verified for s in selected):
        print("note: licence classes are unverified (verified = false); check before use.", file=out)

    before = json.dumps(lock, sort_keys=True)
    results = []
    for source in selected:
        print(f"[{source.id}] {source.access} ...", file=out, flush=True)
        results.append(
            fetch_source(
                source,
                args.data_dir,
                lock,
                record=args.record_checksums and not args.dry_run,
                dry_run=args.dry_run,
                timeout=args.timeout,
                snapshot_download=snapshot_download,
            )
        )
    if args.record_checksums and not args.dry_run and json.dumps(lock, sort_keys=True) != before:
        save_lock(lock_path, lock)
        print(f"recorded checksums in {lock_path}", file=out)

    print_summary(results, out)
    failed = any(r.failed and r.access in {"auto", "hf_dataset"} for r in results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
