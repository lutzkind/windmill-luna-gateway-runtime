from __future__ import annotations

import asyncio
import os
import shlex
import time
from pathlib import Path

import pytest
from fastapi import HTTPException

from app import codex_upstream


def _pid_alive(pid: int) -> bool:
    """True only for a live (non-zombie) process."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - same-user processes only
        return True
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return False
    state = stat.rsplit(")", 1)[1].split()[0]
    return state not in {"Z", "X"}


def _group_alive(pgid: int) -> bool:
    if pgid <= 0:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_dead(pids: list[int], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(_pid_alive(pid) for pid in pids):
            return True
        time.sleep(0.02)
    return not any(_pid_alive(pid) for pid in pids)


def _wait_group_dead(pgid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _group_alive(pgid):
            return True
        time.sleep(0.02)
    return not _group_alive(pgid)


async def _spawn_group(
    pid_file: Path, *, ignore_term: bool = False
) -> tuple[asyncio.subprocess.Process, int]:
    """Start a session-leading shell that records its grandchild pid.

    ``ignore_term`` makes the recorded grandchild install a SIGTERM trap and
    keep the inherited stdout/stderr pipes open, forcing the SIGKILL
    escalation. The readiness marker ensures the trap is installed before the
    caller signals the group.
    """
    ready_file = pid_file.with_suffix(".ready")
    if ignore_term:
        inner = 'trap "" TERM; : > "$0"; while :; do sleep 1; done'
        script = (
            f"sh -c {shlex.quote(inner)} {shlex.quote(str(ready_file))} & "
            'echo $! > "$1"; wait'
        )
    else:
        script = 'sleep 300 & echo $! > "$1"; wait'
    process = await asyncio.create_subprocess_exec(
        "sh",
        "-c",
        script,
        "sh",
        str(pid_file),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            recorded = pid_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            recorded = ""
        if recorded and (not ignore_term or ready_file.exists()):
            return process, int(recorded)
        await asyncio.sleep(0.02)
    raise AssertionError("grandchild pid was never recorded")


def test_timeout_kills_the_full_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(codex_upstream, "PROCESS_TERMINATE_GRACE_SECONDS", 0.3)

    async def scenario() -> None:
        process, grandchild = await _spawn_group(tmp_path / "grandchild.pid")
        assert _pid_alive(process.pid)
        assert _pid_alive(grandchild)

        started = time.monotonic()
        with pytest.raises(HTTPException) as excinfo:
            await codex_upstream._communicate_with_timeout(process, b"", 0.3)
        elapsed = time.monotonic() - started

        assert excinfo.value.status_code == 504
        assert excinfo.value.detail == "codex execution timed out"
        assert elapsed < 5.0
        assert process.returncode is not None
        assert _wait_dead([process.pid, grandchild])
        assert _wait_group_dead(process.pid)

    asyncio.run(scenario())


def test_cancellation_kills_the_full_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(codex_upstream, "PROCESS_TERMINATE_GRACE_SECONDS", 0.3)

    async def scenario() -> None:
        process, grandchild = await _spawn_group(tmp_path / "grandchild.pid")
        task = asyncio.create_task(
            codex_upstream._communicate_with_timeout(process, b"", 60.0)
        )
        await asyncio.sleep(0.2)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert process.returncode is not None
        assert _wait_dead([process.pid, grandchild])
        assert _wait_group_dead(process.pid)

    asyncio.run(scenario())


def test_pipe_holding_survivor_cannot_wedge_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(codex_upstream, "PROCESS_TERMINATE_GRACE_SECONDS", 0.3)

    async def scenario() -> None:
        process, grandchild = await _spawn_group(
            tmp_path / "grandchild.pid", ignore_term=True
        )

        started = time.monotonic()
        await codex_upstream._terminate_process_group(process)
        elapsed = time.monotonic() - started

        # The grandchild survived SIGTERM while holding the inherited pipes,
        # so cleanup may only finish after the bounded SIGKILL escalation.
        assert 0.25 <= elapsed < 5.0
        assert process.returncode is not None
        assert _wait_dead([process.pid, grandchild])
        assert _wait_group_dead(process.pid)

    asyncio.run(scenario())


def test_repeated_timeouts_then_normal_command_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(codex_upstream, "PROCESS_TERMINATE_GRACE_SECONDS", 0.3)

    async def scenario() -> None:
        for index in range(2):
            process, grandchild = await _spawn_group(
                tmp_path / f"grandchild-{index}.pid"
            )
            with pytest.raises(HTTPException) as excinfo:
                await codex_upstream._communicate_with_timeout(process, b"", 0.3)
            assert excinfo.value.status_code == 504
            assert process.returncode is not None
            assert _wait_dead([process.pid, grandchild])
            assert _wait_group_dead(process.pid)

        healthy = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            "printf LUNA_RECOVERED",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        stdout, stderr = await codex_upstream._communicate_with_timeout(
            healthy, b"", 5.0
        )
        assert stdout == b"LUNA_RECOVERED"
        assert stderr == b""
        assert healthy.returncode == 0

    asyncio.run(scenario())


def _blocking_impl(
    running: asyncio.Event, blocked: asyncio.Event
):
    async def impl(*args: object, **kwargs: object) -> None:
        running.set()
        await blocked.wait()

    return impl


async def _acquire_all_semaphore_slots() -> None:
    for _ in range(codex_upstream.MAX_CONCURRENCY):
        async with codex_upstream.SEMAPHORE:
            pass


def test_timeout_and_cancellation_release_auth_sync_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_impl(*args: object, **kwargs: object) -> None:
        raise HTTPException(status_code=504, detail="codex execution timed out")

    monkeypatch.setattr(codex_upstream, "_run_codex_once_impl", failing_impl)

    async def scenario() -> None:
        with pytest.raises(HTTPException) as excinfo:
            await codex_upstream._run_codex_once("prompt", None, False)
        assert excinfo.value.status_code == 504
        assert codex_upstream.AUTH_SYNC_LOCK.locked() is False

        blocked = asyncio.Event()
        running = asyncio.Event()
        monkeypatch.setattr(
            codex_upstream,
            "_run_codex_once_impl",
            _blocking_impl(running, blocked),
        )
        task = asyncio.create_task(
            codex_upstream._run_codex_once("prompt", None, False)
        )
        await asyncio.wait_for(running.wait(), timeout=1.0)
        assert codex_upstream.AUTH_SYNC_LOCK.locked() is True
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert codex_upstream.AUTH_SYNC_LOCK.locked() is False
        await asyncio.wait_for(codex_upstream.AUTH_SYNC_LOCK.acquire(), timeout=1.0)
        codex_upstream.AUTH_SYNC_LOCK.release()

    asyncio.run(scenario())


def test_timeout_and_cancellation_release_semaphore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_once(*args: object, **kwargs: object) -> None:
        raise HTTPException(status_code=504, detail="codex execution timed out")

    monkeypatch.setattr(codex_upstream, "_run_codex_once", failing_once)

    async def scenario() -> None:
        with pytest.raises(HTTPException) as excinfo:
            await codex_upstream._run_codex("prompt")
        assert excinfo.value.status_code == 504
        await asyncio.wait_for(_acquire_all_semaphore_slots(), timeout=1.0)

        running = asyncio.Event()
        blocked = asyncio.Event()

        async def blocking_once(*args: object, **kwargs: object) -> None:
            running.set()
            await blocked.wait()

        monkeypatch.setattr(codex_upstream, "_run_codex_once", blocking_once)
        task = asyncio.create_task(codex_upstream._run_codex("prompt"))
        await asyncio.wait_for(running.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(_acquire_all_semaphore_slots(), timeout=1.0)

    asyncio.run(scenario())


def test_gateway_default_deadline_exceeds_sidecar_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main as gateway_main

    monkeypatch.delenv("PROVIDER_TIMEOUT_SECONDS", raising=False)

    settings = gateway_main.Settings.from_env()

    assert settings.timeout_seconds == 210.0
    assert codex_upstream.TIMEOUT_SECONDS == 180
    assert settings.timeout_seconds > codex_upstream.TIMEOUT_SECONDS


def test_run_codex_once_starts_a_new_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 4242
        returncode = 0
        stdin = None
        stdout = None
        stderr = None

    async def fake_exec(*command: str, **kwargs: object) -> FakeProcess:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    async def fake_communicate(
        process: object, prompt: bytes, timeout: float
    ) -> tuple[bytes, bytes]:
        return b"hello", b""

    monkeypatch.setattr(codex_upstream.shutil, "which", lambda _name: "/usr/bin/codex")
    monkeypatch.setattr(codex_upstream, "_prepare_runtime_home", lambda: None)
    monkeypatch.setattr(
        codex_upstream.asyncio, "create_subprocess_exec", fake_exec
    )
    monkeypatch.setattr(
        codex_upstream, "_communicate_with_timeout", fake_communicate
    )

    run = asyncio.run(codex_upstream._run_codex_once_impl("prompt", None, False))

    assert run.text == "hello"
    assert captured["command"][0] == codex_upstream.CODEX_BINARY
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["stdin"] is asyncio.subprocess.PIPE
    assert captured["kwargs"]["stdout"] is asyncio.subprocess.PIPE
    assert captured["kwargs"]["stderr"] is asyncio.subprocess.PIPE
