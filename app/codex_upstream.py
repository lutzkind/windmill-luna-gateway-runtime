from __future__ import annotations

import asyncio
import os
import shutil
import time
import uuid
from typing import Any

from fastapi import FastAPI, Header, HTTPException

app = FastAPI(title="Codex CLI OpenAI-compatible upstream", version="1.0.0")

API_KEY = os.environ.get("OPENAI_VIA_CODEX_API_KEY", "").strip()
CODEX_BINARY = os.environ.get("CODEX_BINARY", "codex").strip() or "codex"
CODEX_MODEL = os.environ.get("CODEX_MODEL", "").strip()
CODEX_HOME = os.environ.get("CODEX_HOME", "/root/.codex").strip() or "/root/.codex"
TIMEOUT_SECONDS = max(30, int(os.environ.get("CODEX_TIMEOUT_SECONDS", "180")))
MAX_CONCURRENCY = max(1, int(os.environ.get("CODEX_MAX_CONCURRENCY", "4")))
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENCY)


def _authorize(authorization: str | None) -> None:
    if not API_KEY:
        raise HTTPException(status_code=503, detail="upstream API key is not configured")
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="unauthorized")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("input_text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        text = content.get("text") or content.get("input_text") or content.get("content")
        return text if isinstance(text, str) else ""
    return ""


def _prompt_from_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    sections: list[str] = [
        "Complete this bounded language task without using shell, filesystem, network, or other tools. "
        "Use only the supplied messages. Return only the final requested answer."
    ]
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").upper()
        text = _content_text(message.get("content"))
        if text:
            sections.append(f"<{role}>\n{text}\n</{role}>")
    return "\n\n".join(sections)


def _prompt_from_responses_input(value: Any) -> str:
    if isinstance(value, str):
        return _prompt_from_messages([{"role": "user", "content": value}])
    if isinstance(value, list):
        return _prompt_from_messages(value)
    return _prompt_from_messages([{"role": "user", "content": _content_text(value)}])


async def _run_codex(prompt: str) -> str:
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="empty prompt")
    if shutil.which(CODEX_BINARY) is None:
        raise HTTPException(status_code=503, detail="codex binary unavailable")

    command = [
        CODEX_BINARY,
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--color",
        "never",
    ]
    if CODEX_MODEL:
        command.extend(["--model", CODEX_MODEL])

    env = os.environ.copy()
    env["CODEX_HOME"] = CODEX_HOME
    env.pop("OPENAI_API_KEY", None)

    async with SEMAPHORE:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd="/tmp",
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(prompt.encode("utf-8")),
                timeout=TIMEOUT_SECONDS,
            )
        except TimeoutError:
            process.kill()
            await process.communicate()
            raise HTTPException(status_code=504, detail="codex execution timed out")

    output = stdout.decode("utf-8", errors="replace").strip()
    error_text = stderr.decode("utf-8", errors="replace").strip()
    if process.returncode != 0:
        detail = error_text[-1200:] or output[-1200:] or f"codex exited {process.returncode}"
        raise HTTPException(status_code=502, detail=detail)
    if not output:
        raise HTTPException(status_code=502, detail="codex returned an empty response")
    return output


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "ok": shutil.which(CODEX_BINARY) is not None and os.path.isfile(os.path.join(CODEX_HOME, "auth.json")),
        "provider": "official-codex-cli",
        "binary": CODEX_BINARY,
        "max_concurrency": MAX_CONCURRENCY,
    }


@app.get("/v1/models")
async def models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(authorization)
    return {"object": "list", "data": [{"id": "gpt-5.6-luna", "object": "model", "owned_by": "codex-proxy"}]}


@app.post("/v1/chat/completions")
async def chat_completions(payload: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(authorization)
    requested_model = str(payload.get("model") or "gpt-5.6-luna")
    output = await _run_codex(_prompt_from_messages(payload.get("messages")))
    return {
        "id": f"chatcmpl_{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": output}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.post("/v1/responses")
async def responses(payload: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(authorization)
    requested_model = str(payload.get("model") or "gpt-5.6-luna")
    output = await _run_codex(_prompt_from_responses_input(payload.get("input")))
    response_id = f"resp_{uuid.uuid4().hex}"
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": requested_model,
        "output_text": output,
        "output": [{
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": output, "annotations": []}],
        }],
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }
