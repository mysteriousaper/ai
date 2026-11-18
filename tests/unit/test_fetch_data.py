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
import time
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
    redirects: ClassVar[dict[str, str]] = {}
    trickle: ClassVar[dict[str, int]] = {}
    raw: ClassVar[dict[str, bytes]] = {}

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self.hits[self.path] = self.hits.get(self.path, 0) + 1
        if self.path in self.raw:
            self.wfile.write(self.raw[self.path])
            self.wfile.flush()
            self.close_connection = True
        elif self.path in self.redirects:
            self.send_response(302)
            self.send_header("Location", self.redirects[self.path])
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path in self.trickle:
            # one byte every 50 ms: every read is fast, the whole transfer is slow
            count = self.trickle[self.path]
            self.send_response(200)
            self.send_header("Content-Length", str(count))
            self.end_headers()
            try:
                for _ in range(count):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.05)
            except OSError:
                pass
            self.close_connection = True
        elif self.path in self.routes:
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
    handler = type(
        "H",
        (_Handler,),
        {"routes": {}, "truncated": {}, "hits": {}, "redirects": {}, "trickle": {}, "raw": {}},
    )
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


def test_user_agent_avoids_blocked_token() -> None:
    # saifmohammad.com (NRC lexicons, SemEval-2018) answers 406 to "urllib" user agents.
    assert "urllib" not in fd.USER_AGENT.lower()


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

        read1 = read

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


# --------------------------------------------------------------------------- extraction


def make_zip(entries: dict[str, bytes], symlink: tuple[str, str] | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
        if symlink is not None:
            info = zipfile.ZipInfo(symlink[0])
            info.external_attr = 0o120777 << 16
            zf.writestr(info, symlink[1])
    return buf.getvalue()


def make_tar(entries: dict[str, bytes], links: dict[str, tuple[str, bytes]] | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        for name, (target, kind) in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type = kind
            info.linkname = target
            tf.addfile(info)
    return buf.getvalue()


def test_zip_is_extracted_with_marker(tmp_path: Path, server) -> None:
    base, handler = server
    handler.routes["/d.zip"] = make_zip({"inner/x.txt": b"x", "y.txt": b"y"})
    (source,) = load(tmp_path, auto_source(url=f"{base}/d.zip", dest="d.zip", extract=True))
    data_dir = tmp_path / "raw"

    result = fd.fetch_source(source, data_dir, {})

    assert result.status == "ok", result.detail
    assert (data_dir / "demo" / "inner" / "x.txt").read_bytes() == b"x"
    assert (data_dir / "demo" / "d.zip.extracted").is_file()
    assert fd.fetch_source(source, data_dir, {}).status == "skipped"


def test_tar_gz_is_extracted(tmp_path: Path) -> None:
    archive = tmp_path / "a.tar.gz"
    archive.write_bytes(make_tar({"dir/x.csv": b"1,2"}))
    names = fd.safe_extract(archive, tmp_path / "out")
    assert names == ["dir/x.csv"]
    assert (tmp_path / "out" / "dir" / "x.csv").read_bytes() == b"1,2"


@pytest.mark.parametrize("evil", ["../evil.txt", "/abs/evil.txt", "a/../../evil.txt"])
def test_zip_path_traversal_is_rejected(tmp_path: Path, evil: str) -> None:
    archive = tmp_path / "evil.zip"
    archive.write_bytes(make_zip({"ok.txt": b"ok", evil: b"pwned"}))
    out = tmp_path / "deep" / "out"
    with pytest.raises(fd.UnsafeArchiveError):
        fd.safe_extract(archive, out)
    assert not (tmp_path / "deep" / "evil.txt").exists()
    assert not (out / "ok.txt").exists(), "nothing is written before validation passes"


def test_zip_symlink_member_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "link.zip"
    archive.write_bytes(make_zip({"ok.txt": b"ok"}, symlink=("link", "/etc/passwd")))
    with pytest.raises(fd.UnsafeArchiveError, match="symlink"):
        fd.safe_extract(archive, tmp_path / "out")


@pytest.mark.parametrize("evil", ["../evil.txt", "/abs/evil.txt", "a/../../evil.txt"])
def test_tar_path_traversal_is_rejected(tmp_path: Path, evil: str) -> None:
    archive = tmp_path / "evil.tar.gz"
    archive.write_bytes(make_tar({"ok.txt": b"ok", evil: b"pwned"}))
    out = tmp_path / "deep" / "out"
    with pytest.raises(fd.UnsafeArchiveError):
        fd.safe_extract(archive, out)
    assert not (tmp_path / "deep" / "evil.txt").exists()
    assert not (out / "ok.txt").exists()


@pytest.mark.parametrize(
    ("target", "kind"),
    [
        ("/etc/passwd", tarfile.SYMTYPE),
        ("../../outside", tarfile.SYMTYPE),
        ("../outside", tarfile.LNKTYPE),
    ],
)
def test_tar_escaping_link_is_rejected(tmp_path: Path, target: str, kind: bytes) -> None:
    archive = tmp_path / "links.tar.gz"
    archive.write_bytes(make_tar({"ok.txt": b"ok"}, links={"sub/link": (target, kind)}))
    with pytest.raises(fd.UnsafeArchiveError):
        fd.safe_extract(archive, tmp_path / "out")


def test_tar_internal_symlink_is_allowed(tmp_path: Path) -> None:
    archive = tmp_path / "ok.tar.gz"
    archive.write_bytes(
        make_tar({"a/x.txt": b"x"}, links={"a/y.txt": ("x.txt", tarfile.SYMTYPE)})
    )
    fd.safe_extract(archive, tmp_path / "out")
    assert (tmp_path / "out" / "a" / "y.txt").read_bytes() == b"x"


def test_unsafe_archive_fails_the_source(tmp_path: Path, server) -> None:
    base, handler = server
    handler.routes["/d.zip"] = make_zip({"../evil.txt": b"pwned"})
    (source,) = load(tmp_path, auto_source(url=f"{base}/d.zip", dest="d.zip", extract=True))
    result = fd.fetch_source(source, tmp_path / "raw", {})
    assert result.failed
    assert "unsafe" in result.detail
    assert not (tmp_path / "raw" / "evil.txt").exists()


def test_dest_through_symlinked_dir_is_rejected(tmp_path: Path, server) -> None:
    base, handler = server
    handler.routes["/a.txt"] = b"hello"
    (source,) = load(tmp_path, auto_source(url=f"{base}/a.txt", dest="sub/a.txt"))
    data_dir = tmp_path / "raw"
    (data_dir / "demo").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (data_dir / "demo" / "sub").symlink_to(outside)

    result = fd.fetch_source(source, data_dir, {})

    assert result.failed
    assert "escapes" in result.detail
    assert not (outside / "a.txt").exists()


# --------------------------------------------------------------------------- manual + hf


def test_manual_source_missing(tmp_path: Path) -> None:
    (source,) = load(tmp_path, manual_source())
    result = fd.fetch_source(source, tmp_path / "raw", {})
    assert result.status == "missing"
    assert not result.failed
    assert "data.csv" in result.detail
    assert "Log in" in result.instructions
    assert not (tmp_path / "raw" / "hand").exists(), "manual sources never create files"


def test_manual_source_present_writes_source_json(tmp_path: Path) -> None:
    (source,) = load(tmp_path, manual_source(expected='["data.csv", "**/*.txt"]'))
    src = tmp_path / "raw" / "hand"
    (src / "deep").mkdir(parents=True)
    (src / "data.csv").write_bytes(b"a,b")
    (src / "deep" / "n.txt").write_bytes(b"n")

    result = fd.fetch_source(source, tmp_path / "raw", {})

    assert result.status == "present"
    meta = json.loads((src / "SOURCE.json").read_text())
    assert meta["sha256s"] == {"data.csv": sha(b"a,b"), "deep/n.txt": sha(b"n")}
    assert meta["urls"] == []


def test_manual_source_never_downloads(tmp_path: Path, monkeypatch) -> None:
    def no_network(*a: object, **k: object) -> None:
        raise AssertionError("manual source tried the network")

    monkeypatch.setattr(fd.urllib.request, "urlopen", no_network)
    (source,) = load(tmp_path, manual_source())
    assert fd.fetch_source(source, tmp_path / "raw", {}).status == "missing"


def fake_snapshot(files: dict[str, bytes], calls: list[dict[str, object]]):
    def snapshot(**kwargs: object) -> str:
        calls.append(kwargs)
        local = Path(str(kwargs["local_dir"]))
        for rel, data in files.items():
            (local / rel).parent.mkdir(parents=True, exist_ok=True)
            (local / rel).write_bytes(data)
        (local / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        (local / ".cache" / "huggingface" / "meta").write_text("x")
        return str(local)

    return snapshot


def test_hf_dataset_snapshot(tmp_path: Path) -> None:
    (source,) = load(tmp_path, hf_source())
    calls: list[dict[str, object]] = []
    files = {"data/train.parquet": b"PAR1", "README.md": b"# card"}
    lock = {"hfds": {"README.md": sha(b"# card")}}

    result = fd.fetch_source(
        source, tmp_path / "raw", lock, snapshot_download=fake_snapshot(files, calls)
    )

    assert result.status == "ok", result.detail
    assert calls[0]["repo_id"] == "org/hfds"
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["revision"] == "a" * 40
    assert calls[0]["local_dir"] == str(tmp_path / "raw" / "hfds")
    assert result.unpinned == ["data/train.parquet"]
    meta = json.loads((tmp_path / "raw" / "hfds" / "SOURCE.json").read_text())
    assert meta["revision"] == "a" * 40
    assert set(meta["sha256s"]) == {"data/train.parquet", "README.md"}


def test_hf_dataset_lock_mismatch_fails(tmp_path: Path) -> None:
    (source,) = load(tmp_path, hf_source())
    lock = {"hfds": {"README.md": sha(b"other")}}
    result = fd.fetch_source(
        source, tmp_path / "raw", lock, snapshot_download=fake_snapshot({"README.md": b"x"}, [])
    )
    assert result.failed
    assert "mismatch" in result.detail


def test_hf_dataset_error_is_reported(tmp_path: Path) -> None:
    (source,) = load(tmp_path, hf_source())

    def broken(**kwargs: object) -> str:
        raise ConnectionError("no route to host")

    result = fd.fetch_source(source, tmp_path / "raw", {}, snapshot_download=broken)
    assert result.failed
    assert "no route to host" in result.detail


# --------------------------------------------------------------------------- CLI


def cli_setup(tmp_path: Path, base: str, handler: type[_Handler]) -> tuple[Path, Path]:
    handler.routes["/free.txt"] = b"free"
    handler.routes["/nc.txt"] = b"nc"
    body = (
        auto_source(sid="free", url=f"{base}/free.txt", dest="free.txt")
        + auto_source(sid="nc", url=f"{base}/nc.txt", dest="nc.txt", licence_class="research-only")
        + manual_source()
    )
    return write_manifest(tmp_path, body), tmp_path / "raw"


def run_cli(manifest: Path, data_dir: Path, *args: str) -> int:
    return fd.main(["--manifest", str(manifest), "--data-dir", str(data_dir), *args])


def test_cli_list(tmp_path: Path, server, capsys) -> None:
    manifest, data_dir = cli_setup(tmp_path, *server)
    assert run_cli(manifest, data_dir, "--list") == 0
    out = capsys.readouterr().out
    for sid in ("free", "nc", "hand"):
        assert sid in out
    assert "research-only" in out
    assert not data_dir.exists()


def test_cli_fetch_all_reports_missing_manual_but_exits_zero(tmp_path: Path, server, capsys) -> None:
    manifest, data_dir = cli_setup(tmp_path, *server)
    assert run_cli(manifest, data_dir) == 0
    out = capsys.readouterr().out
    assert (data_dir / "free" / "free.txt").is_file()
    assert (data_dir / "nc" / "nc.txt").is_file()
    assert "MISSING" in out
    assert "Log in at example.org" in out
    assert "unpinned" in out
    assert not fd.lock_path_for(manifest).exists(), "lock is only written by --record-checksums"


def test_cli_commercial_only(tmp_path: Path, server, capsys) -> None:
    manifest, data_dir = cli_setup(tmp_path, *server)
    assert run_cli(manifest, data_dir, "--commercial-only") == 0
    out = capsys.readouterr().out
    assert (data_dir / "free" / "free.txt").is_file()
    assert not (data_dir / "nc").exists()
    assert not (data_dir / "hand").exists()
    assert "nc" in out and "excluded" in out


def test_cli_only_and_unknown_id(tmp_path: Path, server, capsys) -> None:
    manifest, data_dir = cli_setup(tmp_path, *server)
    assert run_cli(manifest, data_dir, "--only", "nc") == 0
    assert (data_dir / "nc" / "nc.txt").is_file()
    assert not (data_dir / "free").exists()
    assert run_cli(manifest, data_dir, "--only", "nope") == 2
    assert "nope" in capsys.readouterr().err


def test_cli_dry_run_touches_nothing(tmp_path: Path, server, monkeypatch, capsys) -> None:
    manifest, data_dir = cli_setup(tmp_path, *server)

    def no_network(*a: object, **k: object) -> None:
        raise AssertionError("dry run used the network")

    monkeypatch.setattr(fd.urllib.request, "urlopen", no_network)
    assert run_cli(manifest, data_dir, "--dry-run", "--record-checksums") == 0
    assert not data_dir.exists()
    assert not fd.lock_path_for(manifest).exists()
    assert "would download" in capsys.readouterr().out


def test_cli_failure_exits_one(tmp_path: Path, server, capsys) -> None:
    manifest, data_dir = cli_setup(tmp_path, *server)
    del server[1].routes["/nc.txt"]
    assert run_cli(manifest, data_dir) == 1
    assert "FAILED" in capsys.readouterr().out
    assert (data_dir / "free" / "free.txt").is_file(), "one failure does not stop the others"


def test_cli_record_checksums_then_verify(tmp_path: Path, server, capsys) -> None:
    manifest, data_dir = cli_setup(tmp_path, *server)
    assert run_cli(manifest, data_dir, "--record-checksums") == 0
    lock = fd.load_lock(fd.lock_path_for(manifest))
    assert lock == {"free": {"free.txt": sha(b"free")}, "nc": {"nc.txt": sha(b"nc")}}

    server[1].routes["/nc.txt"] = b"changed upstream"
    (data_dir / "nc" / "nc.txt").unlink()
    assert run_cli(manifest, data_dir) == 1
    assert (data_dir / "nc" / "nc.txt.bad").read_bytes() == b"changed upstream"
    assert "mismatch" in capsys.readouterr().out


def test_cli_bad_manifest_exits_two(tmp_path: Path, capsys) -> None:
    manifest = write_manifest(tmp_path, auto_source(licence_class="whatever"))
    assert run_cli(manifest, tmp_path / "raw") == 2
    assert "licence_class" in capsys.readouterr().err


# --------------------------------------------------------------------------- review fixes


def test_slow_trickle_download_hits_overall_deadline(tmp_path: Path, server) -> None:
    base, handler = server
    handler.trickle["/slow.bin"] = 60  # ~3 s in total, but each read returns within 50 ms
    dest = tmp_path / "slow.bin"
    started = time.monotonic()
    with pytest.raises(fd.FetchError, match="deadline"):
        fd.download(f"{base}/slow.bin", dest, timeout=2, deadline=0.5)
    assert time.monotonic() - started < 2.0
    assert not dest.exists()
    assert not list(tmp_path.iterdir())


# --------------------------------------------------------------------------- network


@pytest.mark.network
@pytest.mark.parametrize(
    "url",
    [f.url for s in fd.load_manifest(REAL_MANIFEST) if s.access == "auto" for f in s.files],
)
def test_auto_urls_are_reachable(url: str) -> None:
    """HEAD-check every auto URL in the real manifest (no body is downloaded)."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": fd.USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            assert response.status < 400
    except urllib.error.HTTPError as exc:
        pytest.fail(f"{url}: HTTP {exc.code}")
