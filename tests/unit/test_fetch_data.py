"""Tests for scripts/fetch_data.py: manifest validation, downloads, locking, safe extraction.

No real network: downloads go to a local http.server on 127.0.0.1.
"""

from __future__ import annotations

import hashlib
import http.server
import io
import json
import tarfile
import threading
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import pytest

from scripts import fetch_data as fd

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_MANIFEST = REPO_ROOT / "config" / "data_sources.toml"


# --------------------------------------------------------------------------- helpers


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_manifest(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "sources.toml"
    path.write_text(body, encoding="utf-8")
    return path


def auto_source(
    sid: str = "demo",
    url: str = "http://127.0.0.1:1/a.txt",
    dest: str = "a.txt",
    licence_class: str = "commercial-OK",
    extract: bool = False,
    extra: str = "",
) -> str:
    return f"""
[[source]]
id = "{sid}"
name = "Demo {sid}"
licence = "MIT"
licence_class = "{licence_class}"
access = "auto"
homepage = "https://example.org/{sid}"
verified = false
instructions = ""
{extra}
[[source.files]]
url = "{url}"
dest = "{dest}"
extract = {str(extract).lower()}
"""


def manual_source(sid: str = "hand", expected: str = '["data.csv"]') -> str:
    return f"""
[[source]]
id = "{sid}"
name = "Manual {sid}"
licence = "unknown"
licence_class = "unclear"
access = "manual"
homepage = "https://example.org/{sid}"
verified = false
instructions = "Log in at example.org and save data.csv into data/raw/{sid}/."
expected = {expected}
"""


def hf_source(sid: str = "hfds", revision: str = "a" * 40) -> str:
    return f"""
[[source]]
id = "{sid}"
name = "HF {sid}"
licence = "Apache-2.0"
licence_class = "commercial-OK"
access = "hf_dataset"
homepage = "https://huggingface.co/datasets/org/{sid}"
verified = false
instructions = ""
repo_id = "org/{sid}"
revision = "{revision}"
"""


class _Handler(http.server.BaseHTTPRequestHandler):
    routes: ClassVar[dict[str, bytes]] = {}
    truncated: ClassVar[dict[str, bytes]] = {}
    hits: ClassVar[dict[str, int]] = {}

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self.hits[self.path] = self.hits.get(self.path, 0) + 1
        if self.path in self.routes:
            body = self.routes[self.path]
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path in self.truncated:
            body = self.truncated[self.path]
            self.send_response(200)
            self.send_header("Content-Length", str(len(body) * 10))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            self.close_connection = True
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


@pytest.fixture
def server() -> Iterator[tuple[str, type[_Handler]]]:
    handler = type("H", (_Handler,), {"routes": {}, "truncated": {}, "hits": {}})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", handler
    finally:
        httpd.shutdown()
        httpd.server_close()


# --------------------------------------------------------------------------- manifest


def test_load_minimal_manifest(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, auto_source() + manual_source() + hf_source())
    sources = fd.load_manifest(path)
    assert [s.id for s in sources] == ["demo", "hand", "hfds"]
    assert sources[0].files[0].dest == "a.txt"
    assert sources[0].verified is False
    assert sources[1].expected == ("data.csv",)
    assert sources[2].repo_id == "org/hfds"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (auto_source(extra='colour = "red"'), "unknown key"),
        (auto_source(licence_class="free"), "licence_class"),
        (auto_source(sid="Bad-Id"), "id"),
        (auto_source().replace('access = "auto"', 'access = "ftp"'), "access"),
        (auto_source().replace("verified = false", 'verified = "no"'), "verified"),
        (auto_source().replace('name = "Demo demo"\n', ""), "missing"),
        (auto_source() + auto_source(), "duplicate"),
        (auto_source(dest="../evil.txt"), "dest"),
        (auto_source(dest="/etc/passwd"), "dest"),
        (auto_source(dest="sub/../../x"), "dest"),
        (auto_source(dest="SOURCE.json"), "dest"),
        (auto_source(url="file:///etc/passwd"), "url"),
        (auto_source(extract=True, dest="a.rar"), "extract"),
        (manual_source(expected='["../x.csv"]'), "expected"),
        (manual_source(expected="[]"), "expected"),
        (hf_source(revision=""), "revision"),
        (hf_source().replace('repo_id = "org/hfds"\n', ""), "repo_id"),
        (
            auto_source().replace("[[source.files]]", 'repo_id = "x/y"\n[[source.files]]'),
            "repo_id",
        ),
        ('schema = 1\n' + auto_source(), "unknown key"),
        (auto_source().replace('extract = false', 'extract = false\nmd5 = "x"'), "unknown key"),
    ],
)
def test_manifest_rejects_bad_input(tmp_path: Path, body: str, message: str) -> None:
    path = write_manifest(tmp_path, body)
    with pytest.raises(fd.ManifestError, match=message):
        fd.load_manifest(path)


def test_auto_source_needs_files(tmp_path: Path) -> None:
    body = auto_source().split("[[source.files]]")[0]
    with pytest.raises(fd.ManifestError, match="files"):
        fd.load_manifest(write_manifest(tmp_path, body))


def test_manifest_error_names_the_source(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, auto_source(sid="named", licence_class="nope"))
    with pytest.raises(fd.ManifestError, match="named"):
        fd.load_manifest(path)


def test_real_manifest_is_valid_and_unverified() -> None:
    sources = fd.load_manifest(REAL_MANIFEST)
    ids = {s.id for s in sources}
    assert {"goemotions", "nrc_vad_v2", "emobank", "isear", "emosurv"} <= ids
    assert all(s.verified is False for s in sources)
    for s in sources:
        if s.access == "manual":
            assert s.instructions.strip(), s.id


# --------------------------------------------------------------------------- auto downloads


def load(tmp_path: Path, body: str) -> list[fd.Source]:
    return fd.load_manifest(write_manifest(tmp_path, body))


def test_happy_path_downloads_and_writes_source_json(tmp_path: Path, server) -> None:
    base, handler = server
    handler.routes["/a.txt"] = b"hello"
    handler.routes["/b.txt"] = b"world"
    body = auto_source(url=f"{base}/a.txt") + f"""
[[source.files]]
url = "{base}/b.txt"
dest = "nested/b.txt"
extract = false
"""
    (source,) = load(tmp_path, body)
    data_dir = tmp_path / "raw"
    lock = {"demo": {"a.txt": sha(b"hello")}}

    result = fd.fetch_source(source, data_dir, lock)

    assert result.status == "ok"
    assert not result.failed
    assert (data_dir / "demo" / "a.txt").read_bytes() == b"hello"
    assert (data_dir / "demo" / "nested" / "b.txt").read_bytes() == b"world"
    assert result.unpinned == ["nested/b.txt"]
    assert not list(data_dir.rglob("*.part"))

    meta = json.loads((data_dir / "demo" / "SOURCE.json").read_text())
    assert meta["id"] == "demo"
    assert meta["name"] == "Demo demo"
    assert meta["urls"] == [f"{base}/a.txt", f"{base}/b.txt"]
    assert meta["sha256s"] == {"a.txt": sha(b"hello"), "nested/b.txt": sha(b"world")}
    assert meta["licence"] == "MIT"
    assert meta["licence_class"] == "commercial-OK"
    assert meta["verified"] is False
    assert meta["fetched_at"].endswith("Z")


def test_checksum_mismatch_is_an_error_and_keeps_bad_file(tmp_path: Path, server) -> None:
    base, handler = server
    handler.routes["/a.txt"] = b"tampered"
    (source,) = load(tmp_path, auto_source(url=f"{base}/a.txt"))
    data_dir = tmp_path / "raw"

    result = fd.fetch_source(source, data_dir, {"demo": {"a.txt": sha(b"original")}})

    assert result.failed
    assert "mismatch" in result.detail
    assert not (data_dir / "demo" / "a.txt").exists()
    assert (data_dir / "demo" / "a.txt.bad").read_bytes() == b"tampered"
    assert not list(data_dir.rglob("*.part"))
    assert not (data_dir / "demo" / "SOURCE.json").exists()


def test_interrupted_download_leaves_no_partial_file(tmp_path: Path, server) -> None:
    base, handler = server
    handler.truncated["/a.txt"] = b"only part of it"
    (source,) = load(tmp_path, auto_source(url=f"{base}/a.txt"))
    data_dir = tmp_path / "raw"

    result = fd.fetch_source(source, data_dir, {})

    assert result.failed
    assert not (data_dir / "demo" / "a.txt").exists()
    assert not list(data_dir.rglob("*.part"))


def test_http_error_is_reported_clearly(tmp_path: Path, server) -> None:
    base, _ = server
    (source,) = load(tmp_path, auto_source(url=f"{base}/missing.txt"))
    result = fd.fetch_source(source, tmp_path / "raw", {})
    assert result.failed
    assert "404" in result.detail


def test_download_removes_part_on_any_exception(tmp_path: Path, monkeypatch) -> None:
    class Boom(io.BytesIO):
        headers: ClassVar[dict[str, str]] = {}

        def read(self, n: int | None = -1) -> bytes:
            raise KeyboardInterrupt

    monkeypatch.setattr(fd.urllib.request, "urlopen", lambda *a, **k: Boom())
    dest = tmp_path / "x.bin"
    with pytest.raises(KeyboardInterrupt):
        fd.download("http://127.0.0.1/x", dest, timeout=1)
    assert not dest.exists()
    assert not list(tmp_path.iterdir())


def test_idempotent_rerun_skips_download(tmp_path: Path, server) -> None:
    base, handler = server
    handler.routes["/a.txt"] = b"hello"
    (source,) = load(tmp_path, auto_source(url=f"{base}/a.txt"))
    data_dir = tmp_path / "raw"
    lock = {"demo": {"a.txt": sha(b"hello")}}

    first = fd.fetch_source(source, data_dir, lock)
    second = fd.fetch_source(source, data_dir, lock)

    assert first.status == "ok"
    assert second.status == "skipped"
    assert handler.hits["/a.txt"] == 1


def test_unpinned_rerun_skips_using_source_json(tmp_path: Path, server) -> None:
    base, handler = server
    handler.routes["/a.txt"] = b"hello"
    (source,) = load(tmp_path, auto_source(url=f"{base}/a.txt"))
    data_dir = tmp_path / "raw"

    fd.fetch_source(source, data_dir, {})
    second = fd.fetch_source(source, data_dir, {})

    assert second.status == "skipped"
    assert second.unpinned == ["a.txt"]
    assert handler.hits["/a.txt"] == 1


def test_corrupted_local_file_is_refetched(tmp_path: Path, server) -> None:
    base, handler = server
    handler.routes["/a.txt"] = b"hello"
    (source,) = load(tmp_path, auto_source(url=f"{base}/a.txt"))
    data_dir = tmp_path / "raw"
    lock = {"demo": {"a.txt": sha(b"hello")}}
    fd.fetch_source(source, data_dir, lock)
    (data_dir / "demo" / "a.txt").write_bytes(b"corrupt")

    result = fd.fetch_source(source, data_dir, lock)

    assert result.status == "ok"
    assert (data_dir / "demo" / "a.txt").read_bytes() == b"hello"
    assert handler.hits["/a.txt"] == 2


def test_record_checksums_fills_lock(tmp_path: Path, server) -> None:
    base, handler = server
    handler.routes["/a.txt"] = b"hello"
    (source,) = load(tmp_path, auto_source(url=f"{base}/a.txt"))
    lock: dict[str, dict[str, str]] = {}

    result = fd.fetch_source(source, tmp_path / "raw", lock, record=True)

    assert result.unpinned == []
    assert lock == {"demo": {"a.txt": sha(b"hello")}}


def test_lock_roundtrip_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "lock.json"
    assert fd.load_lock(path) == {}
    fd.save_lock(path, {"b": {"x": "0" * 64}, "a": {"y": "1" * 64}})
    assert fd.load_lock(path) == {"a": {"y": "1" * 64}, "b": {"x": "0" * 64}}
    path.write_text('{"version": 1, "files": {"a": {"y": "nothex"}}}')
    with pytest.raises(fd.ManifestError, match="sha256"):
        fd.load_lock(path)
