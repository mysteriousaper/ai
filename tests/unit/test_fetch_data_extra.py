"""Extra tester-written tests for scripts/fetch_data.py and config/data_sources.toml.

Covers manifest edge cases, CLI exit codes/output, Hypothesis path-safety properties,
lock-file corruption, ``.part`` leftovers from a crashed run, and invariants of the real
manifest. No real network: downloads go to a local http.server on 127.0.0.1.

Tests marked ``xfail(strict=True, reason="BUG: ...")`` document product bugs; they start
passing (and so fail strictly) once the bug is fixed.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import threading
import urllib.parse
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import ClassVar

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scripts import fetch_data as fd

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_MANIFEST = REPO_ROOT / "config" / "data_sources.toml"


# --------------------------------------------------------------------------- helpers


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def toml_str(value: str) -> str:
    """Encode ``value`` as a TOML basic string (JSON string escapes are valid TOML)."""
    return json.dumps(value, ensure_ascii=False)


def auto_src(
    sid: str = "demo",
    url: str = "https://example.org/a.txt",
    dest: str = "a.txt",
    name: str = "Demo",
    licence_class: str = "commercial-OK",
    extract: bool = False,
    extra: str = "",
) -> str:
    return f"""
[[source]]
id = {toml_str(sid)}
name = {toml_str(name)}
licence = "MIT"
licence_class = "{licence_class}"
access = "auto"
homepage = "https://example.org/"
verified = false
instructions = ""
{extra}
[[source.files]]
url = {toml_str(url)}
dest = {toml_str(dest)}
extract = {str(extract).lower()}
"""


def manual_src(
    sid: str = "hand", expected: str = '["data.csv"]', instructions: str = "Save data.csv here."
) -> str:
    return f"""
[[source]]
id = "{sid}"
name = "Manual {sid}"
licence = "unknown"
licence_class = "unclear"
access = "manual"
homepage = "https://example.org/{sid}"
verified = false
instructions = {toml_str(instructions)}
expected = {expected}
"""


def hf_src(sid: str = "hfds", revision: str | None = "a" * 40, repo_id: str = "org/ds") -> str:
    rev = "" if revision is None else f"revision = {toml_str(revision)}\n"
    return f"""
[[source]]
id = "{sid}"
name = "HF {sid}"
licence = "Apache-2.0"
licence_class = "commercial-OK"
access = "hf_dataset"
homepage = "https://huggingface.co/datasets/org/ds"
verified = false
instructions = ""
repo_id = {toml_str(repo_id)}
{rev}"""


def write(tmp_path: Path, body: str, name: str = "sources.toml") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def load(tmp_path: Path, body: str) -> list[fd.Source]:
    return fd.load_manifest(write(tmp_path, body))


def run_cli(manifest: Path, data_dir: Path, *args: str, **kw: object) -> int:
    return fd.main(["--manifest", str(manifest), "--data-dir", str(data_dir), *args], **kw)


class _Handler(http.server.BaseHTTPRequestHandler):
    routes: ClassVar[dict[str, bytes]] = {}
    hits: ClassVar[dict[str, int]] = {}

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        path = urllib.parse.unquote(self.path)
        self.hits[path] = self.hits.get(path, 0) + 1
        body = self.routes.get(path)
        self.send_response(200 if body is not None else 404)
        self.send_header("Content-Length", str(len(body or b"")))
        self.end_headers()
        self.wfile.write(body or b"")


@pytest.fixture
def server() -> Iterator[tuple[str, type[_Handler]]]:
    handler = type("H", (_Handler,), {"routes": {}, "hits": {}})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", handler
    finally:
        httpd.shutdown()
        httpd.server_close()


def no_network(*a: object, **k: object) -> None:
    raise AssertionError("used the network")


# --------------------------------------------------------------------------- manifest: ids


def test_duplicate_ids_non_adjacent_and_across_access_kinds(tmp_path: Path) -> None:
    body = auto_src(sid="x") + manual_src(sid="other") + hf_src(sid="x")
    with pytest.raises(fd.ManifestError, match=r"duplicate source id 'x'"):
        load(tmp_path, body)


def test_ids_differing_only_by_case_cannot_collide(tmp_path: Path) -> None:
    # Upper case is rejected outright, so case-insensitive filesystems cannot merge two ids.
    with pytest.raises(fd.ManifestError, match="id must match"):
        load(tmp_path, auto_src(sid="x") + auto_src(sid="X"))


@pytest.mark.parametrize("sid", ["данные", "տվյալ", "a-b", "a b", "", "a/b", "..", "ａ"])
def test_non_ascii_or_unsafe_ids_are_rejected(tmp_path: Path, sid: str) -> None:
    with pytest.raises(fd.ManifestError, match="id must match"):
        load(tmp_path, auto_src(sid=sid))


@pytest.mark.xfail(
    strict=True,
    reason="BUG: _ID_RE uses '$', which also matches before a trailing newline, so "
    "id = \"abc\\n\" is accepted and creates data/raw/'abc\\n'/ (use \\Z or fullmatch)",
)
def test_id_with_trailing_newline_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(fd.ManifestError, match="id"):
        load(tmp_path, auto_src(sid="abc\n"))


# --------------------------------------------------------------------------- manifest: lists


def test_auto_source_with_empty_files_array_is_rejected(tmp_path: Path) -> None:
    body = (
        auto_src()
        .split("[[source.files]]")[0]
        .replace('instructions = ""', 'instructions = ""\nfiles = []')
    )
    with pytest.raises(fd.ManifestError, match="at least one"):
        load(tmp_path, body)


def test_empty_manifest_and_empty_source_list_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(fd.ManifestError, match="at least one"):
        load(tmp_path, "")
    with pytest.raises(fd.ManifestError, match="at least one"):
        load(tmp_path, "source = []\n")


def test_manual_expected_must_hold_strings(tmp_path: Path) -> None:
    with pytest.raises(fd.ManifestError, match="strings"):
        load(tmp_path, manual_src(expected="[1]"))


@pytest.mark.parametrize("instructions", ["", "   ", "\n\t"])
def test_manual_source_needs_non_blank_instructions(tmp_path: Path, instructions: str) -> None:
    with pytest.raises(fd.ManifestError, match="instructions"):
        load(tmp_path, manual_src(instructions=instructions))


def test_duplicate_dest_after_normalisation_is_rejected(tmp_path: Path) -> None:
    body = (
        auto_src(dest="sub/a.txt") + '\n[[source.files]]\nurl = "https://e.org/b"\n'
        'dest = "sub/./a.txt"\nextract = false\n'
    )
    with pytest.raises(fd.ManifestError, match="duplicate dest"):
        load(tmp_path, body)


@pytest.mark.parametrize("dest", ["a.txt.part", "a.txt.bad", "a.zip.extracted", "sub/SOURCE.json"])
def test_bookkeeping_names_are_reserved_for_dest(tmp_path: Path, dest: str) -> None:
    with pytest.raises(fd.ManifestError, match="reserved"):
        load(tmp_path, auto_src(dest=dest))


def test_missing_optional_fields_default_to_empty(tmp_path: Path) -> None:
    (source,) = load(tmp_path, auto_src())
    assert (source.version, source.licence_url, source.notes) == ("", "", "")


# --------------------------------------------------------------------------- manifest: hf


def test_hf_source_without_revision_names_the_missing_key(tmp_path: Path) -> None:
    with pytest.raises(fd.ManifestError, match=r"missing key\(s\) \['revision'\]"):
        load(tmp_path, hf_src(revision=None))


@pytest.mark.parametrize("revision", ["", " ", "\t\n"])
def test_hf_blank_revision_is_rejected(tmp_path: Path, revision: str) -> None:
    with pytest.raises(fd.ManifestError, match="revision"):
        load(tmp_path, hf_src(revision=revision))


def test_hf_branch_revision_is_accepted_but_flagged_unpinned(tmp_path: Path) -> None:
    (source,) = load(tmp_path, hf_src(revision="main"))
    assert not source.revision_pinned
    result = fd.fetch_source(source, tmp_path / "raw", {}, dry_run=True)
    assert "not a sha" in result.detail
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize("rev", ["A" * 40, "a" * 39, "a" * 41, "g" * 40])
def test_revision_pinned_needs_exactly_40_lowercase_hex(tmp_path: Path, rev: str) -> None:
    (source,) = load(tmp_path, hf_src(revision=rev))
    assert not source.revision_pinned


@pytest.mark.xfail(
    strict=True,
    reason="BUG: _SHA_RE uses '$', so a 40-hex revision followed by a newline counts as "
    "pinned although it is not a valid commit sha",
)
def test_revision_with_trailing_newline_is_not_pinned(tmp_path: Path) -> None:
    (source,) = load(tmp_path, hf_src(revision="a" * 40 + "\n"))
    assert not source.revision_pinned


@pytest.mark.parametrize("repo_id", ["noslash", "a/b/c", "/b", "a/", "a b/c"])
def test_hf_bad_repo_id_is_rejected(tmp_path: Path, repo_id: str) -> None:
    with pytest.raises(fd.ManifestError, match="repo_id"):
        load(tmp_path, hf_src(repo_id=repo_id))


@pytest.mark.xfail(
    strict=True,
    reason="BUG: repo_id regex uses '$' (matches before a trailing newline) and allows "
    "'.'/'..' as owner or name, so 'org/name\\n', '../x' and 'org/..' pass validation",
)
@pytest.mark.parametrize("repo_id", ["org/name\n", "../x", "org/.."])
def test_hf_repo_id_suspicious_forms_are_rejected(tmp_path: Path, repo_id: str) -> None:
    with pytest.raises(fd.ManifestError, match="repo_id"):
        load(tmp_path, hf_src(repo_id=repo_id))


# --------------------------------------------------------------------------- manifest: urls


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.org/a.txt",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "//example.org/a.txt",
        "example.org/a.txt",
        " https://example.org/a.txt",
        "",
        "data:text/plain,hi",
    ],
)
def test_non_http_urls_are_rejected(tmp_path: Path, url: str) -> None:
    with pytest.raises(fd.ManifestError, match="url"):
        load(tmp_path, auto_src(url=url))


def test_plain_http_is_accepted_by_the_parser(tmp_path: Path) -> None:
    # Documented current behaviour: the parser accepts http:// (the unit tests rely on a
    # local http server). The https-only policy is enforced on the real manifest below.
    (source,) = load(tmp_path, auto_src(url="http://example.org/a.txt"))
    assert source.files[0].url.startswith("http://")


# --------------------------------------------------------------------------- manifest: unicode


UNICODE_NAME = "Эмоции Հույզեր 😀 naïve"


def test_unicode_name_round_trips_into_listing_and_source_json(
    tmp_path: Path, server, capsys
) -> None:
    base, handler = server
    handler.routes["/a.txt"] = b"hello"
    manifest = write(tmp_path, auto_src(url=f"{base}/a.txt", name=UNICODE_NAME))
    data_dir = tmp_path / "raw"
    assert run_cli(manifest, data_dir, "--list") == 0
    assert UNICODE_NAME in capsys.readouterr().out
    assert run_cli(manifest, data_dir) == 0
    meta = json.loads((data_dir / "demo" / "SOURCE.json").read_text(encoding="utf-8"))
    assert meta["name"] == UNICODE_NAME


def test_unicode_dest_is_downloaded_under_that_name(tmp_path: Path, server) -> None:
    base, handler = server
    handler.routes["/x.csv"] = b"1,2"
    (source,) = load(tmp_path, auto_src(url=f"{base}/x.csv", dest="данные/տվյալ.csv"))
    result = fd.fetch_source(source, tmp_path / "raw", {})
    assert result.status == "ok", result.detail
    assert (tmp_path / "raw" / "demo" / "данные" / "տվյալ.csv").read_bytes() == b"1,2"


@pytest.mark.xfail(
    strict=True,
    reason="BUG: load_manifest catches FileNotFoundError and TOMLDecodeError only; a "
    "non-UTF-8 manifest raises UnicodeDecodeError and the CLI crashes instead of exit 2",
)
def test_manifest_file_that_is_not_utf8_gives_a_manifest_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_bytes(auto_src(name="caf\xe9").encode("latin-1"))
    with pytest.raises(fd.ManifestError):
        fd.load_manifest(path)


# --------------------------------------------------------------------------- CLI


def cli_manifest(tmp_path: Path, base: str = "https://example.org") -> Path:
    body = (
        auto_src(sid="free", url=f"{base}/free.txt", dest="free.txt", name="Free one")
        + auto_src(sid="nc", url=f"{base}/nc.txt", dest="nc.txt", licence_class="research-only")
        + manual_src()
        + hf_src()
    )
    return write(tmp_path, body)


def test_cli_list_exit_zero_shows_all_columns_and_writes_nothing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(fd.urllib.request, "urlopen", no_network)
    manifest = cli_manifest(tmp_path)
    data_dir = tmp_path / "raw"
    assert run_cli(manifest, data_dir, "--list") == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[0].split() == ["id", "access", "licence_class", "verified", "files", "name"]
    assert set(lines[1]) <= {"-", " "}
    body = "\n".join(lines[2:])
    for sid in ("free", "nc", "hand", "hfds"):
        assert sid in body
    assert "1 expected" in body and "org/ds@aaaaaaaa" in body
    assert captured.err == ""
    assert not data_dir.exists()
    assert not fd.lock_path_for(manifest).exists()


def test_cli_list_with_commercial_only_names_the_excluded(tmp_path: Path, capsys) -> None:
    assert run_cli(cli_manifest(tmp_path), tmp_path / "raw", "--list", "--commercial-only") == 0
    out = capsys.readouterr().out
    assert "excluded by --commercial-only: nc, hand" in out


@pytest.mark.parametrize(
    "only", [["--only", "nope"], ["--only", "free,nope"], ["--only", "free", "--only", "nope"]]
)
def test_cli_only_unknown_id_exits_two_before_doing_anything(
    tmp_path: Path, monkeypatch, capsys, only: list[str]
) -> None:
    monkeypatch.setattr(fd.urllib.request, "urlopen", no_network)
    manifest = cli_manifest(tmp_path)
    assert run_cli(manifest, tmp_path / "raw", *only) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("error: --only: unknown source id(s) ['nope']")
    assert "known:" in captured.err and "free" in captured.err
    assert captured.out == ""
    assert not (tmp_path / "raw").exists()


def test_cli_list_with_unknown_only_also_exits_two(tmp_path: Path, capsys) -> None:
    assert run_cli(cli_manifest(tmp_path), tmp_path / "raw", "--list", "--only", "nope") == 2
    assert "nope" in capsys.readouterr().err


def test_cli_only_accepts_spaces_and_repeats(tmp_path: Path, capsys) -> None:
    assert (
        run_cli(
            cli_manifest(tmp_path),
            tmp_path / "raw",
            "--list",
            "--only",
            " free , nc ",
            "--only",
            "hand",
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "free" in out and "nc" in out and "hand" in out and "hfds" not in out


def test_cli_dry_run_with_record_checksums_leaves_existing_lock_byte_identical(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(fd.urllib.request, "urlopen", no_network)
    manifest = cli_manifest(tmp_path)
    lock_path = fd.lock_path_for(manifest)
    fd.save_lock(lock_path, {"free": {"free.txt": "0" * 64}})
    before = lock_path.read_bytes()

    def no_snapshot(**kw: object) -> str:
        raise AssertionError("dry run snapshotted")

    rc = run_cli(
        manifest, tmp_path / "raw", "--dry-run", "--record-checksums", snapshot_download=no_snapshot
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert lock_path.read_bytes() == before
    assert "recorded checksums" not in out
    assert "would download" in out
    assert "would snapshot org/ds@" in out
    assert "MISSING" in out  # the manual source is still reported
    assert not (tmp_path / "raw").exists()
    assert not lock_path.with_name(lock_path.name + ".part").exists()


def test_cli_dry_run_reports_up_to_date_files(tmp_path: Path, server, capsys) -> None:
    base, handler = server
    handler.routes["/free.txt"] = b"free"
    handler.routes["/nc.txt"] = b"nc"
    manifest = cli_manifest(tmp_path, base)
    data_dir = tmp_path / "raw"
    assert run_cli(manifest, data_dir, "--only", "free,nc", "--record-checksums") == 0
    hits = dict(handler.hits)
    capsys.readouterr()
    assert run_cli(manifest, data_dir, "--only", "free,nc", "--dry-run", "--record-checksums") == 0
    out = capsys.readouterr().out
    assert "free.txt: up to date" in out and "nc.txt: up to date" in out
    assert "(unpinned)" not in out
    assert handler.hits == hits


@pytest.mark.xfail(
    strict=True,
    reason="BUG: --timeout accepts negative values; socket.settimeout raises ValueError, "
    "which fetch_source does not catch, so the CLI crashes with a traceback instead of "
    "a usage error (exit 2)",
)
def test_cli_negative_timeout_is_a_usage_error(tmp_path: Path, server, capsys) -> None:
    base, handler = server
    handler.routes["/free.txt"] = b"free"
    manifest = cli_manifest(tmp_path, base)
    try:
        rc = run_cli(manifest, tmp_path / "raw", "--only", "free", "--timeout", "-1")
    except SystemExit as exc:  # argparse rejecting the value is the right outcome
        rc = exc.code
    assert rc == 2


# --------------------------------------------------------------------------- properties

PATH_ALPHABET = st.sampled_from(
    ["a", "b", ".", "..", "...", "/", "//", "\\", ":", "\x00", " ", "~", "link", "ё", "😀", "C:"]
)
path_like = st.lists(PATH_ALPHABET, max_size=12).map("".join)
any_path_text = st.one_of(path_like, st.text(max_size=40))


@pytest.fixture(scope="module")
def jail(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A base dir with a symlink 'link' that points outside it, plus a real sub-dir."""
    root = tmp_path_factory.mktemp("jail")
    base = root / "base"
    (base / "a").mkdir(parents=True)
    outside = root / "outside"
    outside.mkdir()
    os.symlink(outside, base / "link")
    return base


@settings(max_examples=400, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(value=any_path_text)
def test_property_safe_relative_path_never_escapes(value: str) -> None:
    try:
        path = fd.safe_relative_path(value)
    except ValueError:
        return
    assert isinstance(path, PurePosixPath)
    assert not path.is_absolute()
    assert path.parts and ".." not in path.parts
    assert not any(c in str(path) for c in "\\:\x00")


@pytest.mark.xfail(
    strict=True,
    reason="BUG (minor): safe_relative_path only rejects padding on the whole string, so "
    "'a /' is accepted and normalised to 'a ', which safe_relative_path itself rejects; "
    "components with leading/trailing spaces (e.g. dest 'a /x.csv') slip through",
)
@pytest.mark.parametrize("value", ["a /", "a /b", "a/b/ c"])
def test_safe_relative_path_is_idempotent_on_padded_components(value: str) -> None:
    path = fd.safe_relative_path(value)
    assert fd.safe_relative_path(str(path)) == path
    assert all(part == part.strip() for part in path.parts)


@settings(max_examples=400)
@given(value=any_path_text)
def test_property_safe_relative_path_accepts_only_unpadded_components(value: str) -> None:
    """Idempotence over inputs whose components are not space-padded (bug above aside)."""
    try:
        path = fd.safe_relative_path(value)
    except ValueError:
        return
    if all(part == part.strip() for part in path.parts):
        assert fd.safe_relative_path(str(path)) == path


@settings(max_examples=400, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(value=any_path_text)
def test_property_resolve_inside_never_returns_a_path_outside_base(jail: Path, value: str) -> None:
    base = jail.resolve()
    try:
        target = fd.resolve_inside(jail, value)
    except (ValueError, fd.UnsafeArchiveError):
        return
    assert target == base or target.is_relative_to(base), (value, target)
    first = fd.safe_relative_path(value).parts[0]
    assert first != "link", "a path through the escaping symlink got in"


@settings(max_examples=200)
@given(parts=st.lists(st.sampled_from(["a", "b", "ё", "x.csv"]), min_size=1, max_size=6))
def test_property_plain_relative_paths_are_accepted(parts: list[str]) -> None:
    value = "/".join(parts)
    assert str(fd.safe_relative_path(value)) == value


@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(dest=any_path_text)
def test_property_manifest_dest_is_rejected_or_inside(tmp_path_factory, dest: str) -> None:
    raw = {
        "source": [
            {
                "id": "s",
                "name": "n",
                "licence": "l",
                "licence_class": "unclear",
                "access": "auto",
                "homepage": "https://e.org",
                "verified": False,
                "instructions": "",
                "files": [{"url": "https://e.org/x", "dest": dest, "extract": False}],
            }
        ]
    }
    try:
        (source,) = fd.parse_manifest(raw)
    except fd.ManifestError:
        return
    base = Path("/srv/raw/s")
    joined = Path(os.path.normpath(base / source.files[0].dest))
    assert joined.is_relative_to(base) and joined != base


# --------------------------------------------------------------------------- lock corruption


def corrupt_cli(tmp_path: Path, payload: bytes, capsys) -> str:
    manifest = cli_manifest(tmp_path)
    fd.lock_path_for(manifest).write_bytes(payload)
    rc = run_cli(manifest, tmp_path / "raw", "--record-checksums")
    captured = capsys.readouterr()
    assert rc == 2, captured
    assert captured.err.startswith("error: lock file ")
    assert fd.lock_path_for(manifest).read_bytes() == payload, "a corrupt lock is never rewritten"
    assert not (tmp_path / "raw").exists()
    return captured.err


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"", "not valid JSON"),
        (b'{"version": 1, "files": {"a": {"x": "', "not valid JSON"),  # crash mid-write
        (b"\xef\xbb\xbf{}", "not valid JSON"),  # UTF-8 BOM
        (b"[]", "expected keys"),
        (b'{"version": 1}', "expected keys"),
        (b'{"version": 1, "files": {}, "extra": 0}', "expected keys"),
        (b'{"version": 2, "files": {}}', "unsupported version 2"),
        (b'{"version": "1", "files": {}}', "unsupported version '1'"),
        (b'{"version": 0, "files": {}}', "unsupported version 0"),
        (b'{"version": 1, "files": []}', "'files' must be an object"),
        (b'{"version": 1, "files": {"a": ["x"]}}', "entry 'a' must be an object"),
        (b'{"version": 1, "files": {"a": {"x": "ABC"}}}', "a/x is not a lowercase sha256"),
        (b'{"version": 1, "files": {"a": {"x": 5}}}', "a/x is not a lowercase sha256"),
    ],
)
def test_corrupt_lock_gives_clear_error_and_exit_two(
    tmp_path: Path, capsys, payload: bytes, message: str
) -> None:
    err = corrupt_cli(tmp_path, payload, capsys)
    assert message in err
    assert str(fd.lock_path_for(tmp_path / "sources.toml")) in err


@pytest.mark.xfail(
    strict=True,
    reason="BUG: load_lock reads with encoding='utf-8' but only catches JSONDecodeError; a "
    "non-UTF-8 (binary-corrupted) lock raises UnicodeDecodeError and the CLI crashes",
)
def test_non_utf8_lock_gives_clear_error(tmp_path: Path, capsys) -> None:
    corrupt_cli(tmp_path, b"\xff\xfe\x00garbage", capsys)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: a lock path that is a directory raises IsADirectoryError out of load_lock; "
    "the CLI crashes instead of reporting a lock-file error",
)
def test_lock_path_that_is_a_directory_gives_clear_error(tmp_path: Path, capsys) -> None:
    manifest = cli_manifest(tmp_path)
    fd.lock_path_for(manifest).mkdir()
    assert run_cli(manifest, tmp_path / "raw") == 2
    assert "lock file" in capsys.readouterr().err


@pytest.mark.xfail(
    strict=True,
    reason="BUG: version check is 'raw[\"version\"] != 1', so true and 1.0 are accepted as "
    "version 1",
)
@pytest.mark.parametrize("version", ["true", "1.0"])
def test_lock_version_must_be_the_integer_one(tmp_path: Path, version: str) -> None:
    path = tmp_path / "l.json"
    path.write_text(f'{{"version": {version}, "files": {{}}}}', encoding="utf-8")
    with pytest.raises(fd.ManifestError, match="version"):
        fd.load_lock(path)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: _HEX64_RE uses '$', so a digest with a trailing newline passes lock "
    "validation and later fails every download as a checksum mismatch",
)
def test_lock_digest_with_trailing_newline_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "l.json"
    path.write_text(json.dumps({"version": 1, "files": {"a": {"x": "a" * 64 + "\n"}}}))
    with pytest.raises(fd.ManifestError, match="sha256"):
        fd.load_lock(path)


# --------------------------------------------------------------------------- .part leftovers


def test_stale_part_from_crash_is_overwritten_and_removed(tmp_path: Path, server) -> None:
    base, handler = server
    handler.routes["/a.txt"] = b"good bytes"
    (source,) = load(tmp_path, auto_src(url=f"{base}/a.txt"))
    src_dir = tmp_path / "raw" / "demo"
    src_dir.mkdir(parents=True)
    part = src_dir / "a.txt.part"
    part.write_bytes(b"half-written garbage from a previous crash, longer than the body")
    result = fd.fetch_source(source, tmp_path / "raw", {"demo": {"a.txt": sha(b"good bytes")}})
    assert result.status == "ok", result.detail
    assert (src_dir / "a.txt").read_bytes() == b"good bytes"
    assert not part.exists()


def test_stale_part_never_counts_as_a_present_manual_file(tmp_path: Path) -> None:
    (source,) = load(tmp_path, manual_src(expected='["**/*.csv"]'))
    src_dir = tmp_path / "raw" / "hand"
    src_dir.mkdir(parents=True)
    (src_dir / "data.csv.part").write_bytes(b"partial")
    (src_dir / "old.csv.bad").write_bytes(b"bad")
    result = fd.fetch_source(source, tmp_path / "raw", {})
    assert result.status == "missing"


def test_stale_part_is_not_hashed_into_an_hf_snapshot(tmp_path: Path) -> None:
    (source,) = load(tmp_path, hf_src())
    src_dir = tmp_path / "raw" / "hfds"
    src_dir.mkdir(parents=True)
    (src_dir / "train.parquet.part").write_bytes(b"partial")

    def snap(**kw: object) -> str:
        (Path(str(kw["local_dir"])) / "train.parquet").write_bytes(b"data")
        return str(kw["local_dir"])

    lock: fd.Lock = {}
    result = fd.fetch_source(source, tmp_path / "raw", lock, record=True, snapshot_download=snap)
    assert result.status == "ok", result.detail
    assert set(result.sha256s) == {"train.parquet"}
    assert lock == {"hfds": {"train.parquet": sha(b"data")}}


def test_stale_part_does_not_break_an_up_to_date_skip(tmp_path: Path, server) -> None:
    base, handler = server
    handler.routes["/a.txt"] = b"v1"
    (source,) = load(tmp_path, auto_src(url=f"{base}/a.txt"))
    lock = {"demo": {"a.txt": sha(b"v1")}}
    assert fd.fetch_source(source, tmp_path / "raw", lock).status == "ok"
    (tmp_path / "raw" / "demo" / "a.txt.part").write_bytes(b"junk")
    result = fd.fetch_source(source, tmp_path / "raw", lock)
    assert result.status == "skipped"
    assert handler.hits["/a.txt"] == 1


def test_part_path_occupied_by_a_directory_fails_the_source_cleanly(tmp_path: Path, server) -> None:
    base, handler = server
    handler.routes["/a.txt"] = b"v1"
    (source,) = load(tmp_path, auto_src(url=f"{base}/a.txt"))
    (tmp_path / "raw" / "demo" / "a.txt.part").mkdir(parents=True)
    result = fd.fetch_source(source, tmp_path / "raw", {})
    assert result.status == "failed"
    assert "a.txt.part" in result.detail
    assert not (tmp_path / "raw" / "demo" / "a.txt").exists()


def test_stale_lock_part_does_not_block_record_checksums(tmp_path: Path, server, capsys) -> None:
    base, handler = server
    handler.routes["/free.txt"] = b"free"
    manifest = cli_manifest(tmp_path, base)
    lock_path = fd.lock_path_for(manifest)
    stale = lock_path.with_name(lock_path.name + ".part")
    stale.write_text("{ truncated")
    assert run_cli(manifest, tmp_path / "raw", "--only", "free", "--record-checksums") == 0
    assert fd.load_lock(lock_path) == {"free": {"free.txt": sha(b"free")}}
    assert not stale.exists()


def test_stale_bad_file_is_replaced_on_repeated_mismatch(tmp_path: Path, server) -> None:
    base, handler = server
    handler.routes["/a.txt"] = b"tampered v2"
    (source,) = load(tmp_path, auto_src(url=f"{base}/a.txt"))
    src_dir = tmp_path / "raw" / "demo"
    src_dir.mkdir(parents=True)
    (src_dir / "a.txt.bad").write_bytes(b"tampered v1")
    result = fd.fetch_source(source, tmp_path / "raw", {"demo": {"a.txt": sha(b"original")}})
    assert result.status == "failed" and "mismatch" in result.detail
    assert (src_dir / "a.txt.bad").read_bytes() == b"tampered v2"
    assert not (src_dir / "a.txt").exists()
    assert not (src_dir / "a.txt.part").exists()


# --------------------------------------------------------------------------- real manifest


REAL_SOURCES = fd.load_manifest(REAL_MANIFEST)


@pytest.mark.parametrize("source", REAL_SOURCES, ids=lambda s: s.id)
def test_real_source_has_a_known_licence_class(source: fd.Source) -> None:
    assert source.licence_class in fd.LICENCE_CLASSES
    assert source.licence.strip(), "licence text as stated must not be blank"


@pytest.mark.parametrize(
    "source", [s for s in REAL_SOURCES if s.access == "manual"], ids=lambda s: s.id
)
def test_real_manual_source_has_instructions_naming_its_folder(source: fd.Source) -> None:
    assert len(source.instructions.strip()) >= 40
    assert source.expected


@pytest.mark.parametrize(
    "url",
    [f.url for s in REAL_SOURCES if s.access == "auto" for f in s.files],
)
def test_real_auto_urls_are_https(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    assert parsed.scheme == "https" and parsed.netloc, url


def test_real_hf_sources_use_hf_and_are_pinned_or_known_unpinned() -> None:
    unpinned = {s.id for s in REAL_SOURCES if s.access == "hf_dataset" and not s.revision_pinned}
    # dailydialog is deliberately on 'main' (docs/decisions-pending/fetch-data.md); any new
    # unpinned hf source should be a conscious decision.
    assert unpinned <= {"dailydialog"}


def test_real_manifest_lists_works_offline_and_names_every_source(capsys, monkeypatch) -> None:
    monkeypatch.setattr(fd.urllib.request, "urlopen", no_network)
    assert fd.main(["--list"]) == 0
    out = capsys.readouterr().out
    for s in REAL_SOURCES:
        assert s.id in out
