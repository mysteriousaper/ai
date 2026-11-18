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
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
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
