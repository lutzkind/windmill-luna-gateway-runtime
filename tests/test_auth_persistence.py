"""Filesystem-level regression tests for the shared Codex auth contract."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "runtime-entrypoint.sh"

pytestmark = pytest.mark.skipif(
    os.geteuid() != 0 or shutil.which("setpriv") is None,
    reason="the runtime entrypoint must be exercised as root with setpriv",
)


def write_auth(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": f"access-{marker}",
                    "refresh_token": f"refresh-{marker}",
                }
            }
        ),
        encoding="utf-8",
    )
    os.chown(path, 0, 0)
    os.chmod(path, 0o600)


def atomic_replace_auth(path: Path, marker: str) -> int:
    replacement = path.with_name(f".{path.name}.{marker}.tmp")
    write_auth(replacement, marker)
    os.replace(replacement, path)
    return path.stat().st_ino


def runtime_env(source_home: Path, runtime_home: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CODEX_HOME": str(source_home),
            "LUNA_CODEX_HOME": str(runtime_home),
            "CODEX_AUTH_SOURCE": str(source_home / "auth.json"),
        }
    )
    return environment


def run_runtime(source_home: Path, runtime_home: Path, *command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sh", str(ENTRYPOINT), *command],
        env=runtime_env(source_home, runtime_home),
        capture_output=True,
        text=True,
        check=False,
    )


def auth_marker(path: Path) -> str:
    return json.loads(path.read_text(encoding="utf-8"))["tokens"]["access_token"]


def test_host_login_atomic_replacement_is_visible_to_running_sidecar(tmp_path: Path):
    source_home = tmp_path / "shared-codex"
    source_auth = source_home / "auth.json"
    write_auth(source_auth, "old")
    runtime_home = tmp_path / "runtime-home"
    script = (
        "import json, os, pathlib, sys; "
        "path = pathlib.Path(os.environ['CODEX_AUTH_SOURCE']); "
        "print(json.loads(path.read_text())['tokens']['access_token'], flush=True); "
        "sys.stdin.readline(); "
        "print(json.loads(path.read_text())['tokens']['access_token'], flush=True)"
    )
    process = subprocess.Popen(
        ["sh", str(ENTRYPOINT), "python3", "-u", "-c", script],
        env=runtime_env(source_home, runtime_home),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdin is not None
    assert process.stdout.readline().strip() == "access-old"
    old_inode = source_auth.stat().st_ino
    new_inode = atomic_replace_auth(source_auth, "login")
    assert new_inode != old_inode
    process.stdin.write("\n")
    process.stdin.flush()
    assert process.stdout.readline().strip() == "access-login"
    _stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, stderr


def test_refresh_rotation_persists_across_restart_redeploy_and_recreation(tmp_path: Path):
    source_home = tmp_path / "shared-codex"
    source_auth = source_home / "auth.json"
    write_auth(source_auth, "old")
    first = run_runtime(
        source_home,
        tmp_path / "runtime-first",
        "python3",
        "-c",
        "import os; print(open(os.environ['CODEX_AUTH_SOURCE']).read())",
    )
    assert first.returncode == 0
    assert auth_marker(source_auth) == "access-old"

    atomic_replace_auth(source_auth, "rotated")
    for runtime_name in ("runtime-restart", "runtime-redeploy", "runtime-recreated"):
        result = run_runtime(
            source_home,
            tmp_path / runtime_name,
            "python3",
            "-c",
            "import os; print(open(os.environ['CODEX_AUTH_SOURCE']).read())",
        )
        assert result.returncode == 0, result.stderr
        assert "access-rotated" in result.stdout
        assert "refresh-rotated" in result.stdout
    assert auth_marker(source_auth) == "access-rotated"


def test_stale_runtime_credential_cannot_overwrite_newer_host_auth(tmp_path: Path):
    source_home = tmp_path / "shared-codex"
    source_auth = source_home / "auth.json"
    write_auth(source_auth, "new-host")
    stale_runtime = tmp_path / "stale-runtime"
    stale_runtime.mkdir()
    (stale_runtime / "auth.json").write_text("stale-runtime-secret", encoding="utf-8")

    result = run_runtime(
        source_home,
        stale_runtime,
        "python3",
        "-c",
        "import os; print(open(os.environ['CODEX_AUTH_SOURCE']).read())",
    )
    assert result.returncode == 0, result.stderr
    assert "access-new-host" in result.stdout
    assert "stale-runtime-secret" not in result.stdout
    assert auth_marker(source_auth) == "access-new-host"
    assert (stale_runtime / "auth.json").read_text(encoding="utf-8") == "stale-runtime-secret"


def test_legacy_owner_and_permissions_are_normalized_before_drop(tmp_path: Path):
    source_home = tmp_path / "shared-codex"
    source_auth = source_home / "auth.json"
    write_auth(source_auth, "legacy")
    os.chown(source_auth, 10001, 10001)
    os.chmod(source_auth, 0o644)

    result = run_runtime(
        source_home,
        tmp_path / "runtime",
        "python3",
        "-c",
        "import os; print(os.getuid())",
    )
    assert result.returncode == 0, result.stderr
    metadata = source_auth.stat()
    assert (metadata.st_uid, metadata.st_gid) == (0, 0)
    assert metadata.st_mode & 0o777 == 0o600
    assert result.stdout.strip() == "0"


def test_missing_auth_fails_closed_without_stale_resurrection(tmp_path: Path):
    source_home = tmp_path / "shared-codex"
    source_home.mkdir()
    stale_runtime = tmp_path / "runtime"
    stale_runtime.mkdir()
    stale_auth = stale_runtime / "auth.json"
    stale_auth.write_text("stale-runtime-secret", encoding="utf-8")

    result = run_runtime(
        source_home,
        stale_runtime,
        "python3",
        "-c",
        "raise AssertionError('must not start')",
    )
    assert result.returncode == 78
    assert "stale-runtime-secret" not in result.stderr
    assert stale_auth.read_text(encoding="utf-8") == "stale-runtime-secret"


def test_symlinked_or_legacy_secret_file_source_fails_closed(tmp_path: Path):
    source_home = tmp_path / "shared-codex"
    source_home.mkdir()
    secret_file = tmp_path / "legacy-secret-file"
    write_auth(secret_file, "legacy")
    (source_home / "auth.json").symlink_to(secret_file)

    result = run_runtime(
        source_home,
        tmp_path / "runtime",
        "python3",
        "-c",
        "raise AssertionError('must not start')",
    )
    assert result.returncode == 78
    assert "legacy" not in result.stderr
