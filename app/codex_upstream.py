from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException

app = FastAPI(title="Codex CLI OpenAI-compatible upstream", version="1.2.0")

API_KEY = os.environ.get("OPENAI_VIA_CODEX_API_KEY", "").strip()
CODEX_BINARY = os.environ.get("CODEX_BINARY", "codex").strip() or "codex"
CODEX_MODEL = os.environ.get("CODEX_MODEL", "").strip()
SOURCE_CODEX_HOME = Path(os.environ.get("CODEX_HOME", "/root/.codex").strip() or "/root/.codex")
RUNTIME_CODEX_HOME = Path(os.environ.get("LUNA_CODEX_HOME", "/tmp/luna-codex-home").strip() or "/tmp/luna-codex-home")
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
        "Complete this bounded language task without using shell, filesystem, network, MCP, or other tools. "
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


def _chat_requires_json(payload: dict[str, Any]) -> bool:
    fmt = payload.get("response_format")
    return isinstance(fmt, dict) and str(fmt.get("type") or "").strip().lower() in {"json_object", "json_schema"}


def _chat_schema(payload: dict[str, Any]) -> dict[str, Any] | None:
    fmt = payload.get("response_format")
    if not isinstance(fmt, dict) or str(fmt.get("type") or "").strip().lower() != "json_schema":
        return None
    wrapper = fmt.get("json_schema")
    if isinstance(wrapper, dict) and isinstance(wrapper.get("schema"), dict):
        return wrapper["schema"]
    if isinstance(fmt.get("schema"), dict):
        return fmt["schema"]
    return None


def _responses_format(payload: dict[str, Any]) -> dict[str, Any] | None:
    text = payload.get("text")
    fmt = text.get("format") if isinstance(text, dict) else None
    return fmt if isinstance(fmt, dict) else None


def _responses_requires_json(payload: dict[str, Any]) -> bool:
    fmt = _responses_format(payload)
    return isinstance(fmt, dict) and str(fmt.get("type") or "").strip().lower() in {"json", "json_object", "json_schema"}


def _responses_schema(payload: dict[str, Any]) -> dict[str, Any] | None:
    fmt = _responses_format(payload)
    if not isinstance(fmt, dict) or str(fmt.get("type") or "").strip().lower() != "json_schema":
        return None
    if isinstance(fmt.get("schema"), dict):
        return fmt["schema"]
    wrapper = fmt.get("json_schema")
    if isinstance(wrapper, dict) and isinstance(wrapper.get("schema"), dict):
        return wrapper["schema"]
    return None


def _prepare_runtime_home() -> None:
    source_auth = SOURCE_CODEX_HOME / "auth.json"
    if not source_auth.is_file():
        raise HTTPException(status_code=503, detail="codex authentication is unavailable")
    RUNTIME_CODEX_HOME.mkdir(parents=True, exist_ok=True)
    target_auth = RUNTIME_CODEX_HOME / "auth.json"
    shutil.copy2(source_auth, target_auth)
    target_auth.chmod(0o600)
    # Intentionally do not copy config.toml. Completion jobs must never load MCP tools.


async def _run_codex_once(prompt: str, schema: dict[str, Any] | None, require_json: bool) -> str:
    _prepare_runtime_home()
    with tempfile.TemporaryDirectory(prefix="luna-codex-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        output_path = tmp_path / "final.txt"
        command = [
            CODEX_BINARY,
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--output-last-message",
            str(output_path),
        ]
        if CODEX_MODEL:
            command.extend(["--model", CODEX_MODEL])
        if schema is not None:
            schema_path = tmp_path / "schema.json"
            schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
            command.extend(["--output-schema", str(schema_path)])
            prompt = (
                "Your final response must be exactly one strict JSON object satisfying the supplied schema. "
                "Do not return markdown, prose outside the object, or code fences.\n\n" + prompt
            )
        elif require_json:
            prompt = (
                "Your final response must be exactly one valid JSON object. Do not return markdown, prose outside "
                "the object, or code fences. Preserve the keys and structure requested by the user.\n\n" + prompt
            )

        env = os.environ.copy()
        env["CODEX_HOME"] = str(RUNTIME_CODEX_HOME)
        env.pop("OPENAI_API_KEY", None)

        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=tmp_dir,
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

        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        error_text = stderr.decode("utf-8", errors="replace").strip()
        output = output_path.read_text(encoding="utf-8").strip() if output_path.is_file() else stdout_text
        if process.returncode != 0:
            detail = error_text[-1200:] or stdout_text[-1200:] or f"codex exited {process.returncode}"
            raise HTTPException(status_code=502, detail=detail)
        if not output:
            raise HTTPException(status_code=502, detail="codex returned an empty response")
        return output


async def _run_codex(
    prompt: str,
    schema: dict[str, Any] | None = None,
    require_json: bool = False,
) -> str:
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="empty prompt")
    if shutil.which(CODEX_BINARY) is None:
        raise HTTPException(status_code=503, detail="codex binary unavailable")

    async with SEMAPHORE:
        output = await _run_codex_once(prompt, schema, require_json)
        if not require_json:
            return output
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            schema_instruction = " satisfying the supplied schema" if schema is not None else ""
            repair_prompt = (
                "The prior attempt was not valid JSON. Redo the original task from scratch and return exactly one "
                f"valid JSON object{schema_instruction}. No markdown or surrounding prose.\n\n"
                f"ORIGINAL TASK:\n{prompt}\n\nINVALID PRIOR OUTPUT:\n{output[:6000]}"
            )
            output = await _run_codex_once(repair_prompt, schema, True)
            try:
                parsed = json.loads(output)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=502, detail=f"codex returned invalid structured output: {exc.msg}")
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=502, detail="codex structured output was not a JSON object")
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "ok": shutil.which(CODEX_BINARY) is not None and (SOURCE_CODEX_HOME / "auth.json").is_file(),
        "provider": "official-codex-cli",
        "binary": CODEX_BINARY,
        "max_concurrency": MAX_CONCURRENCY,
        "structured_output": True,
        "mcp_tools_loaded": False,
    }


@app.get("/v1/models")
async def models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(authorization)
    return {"object": "list", "data": [{"id": "gpt-5.6-luna", "object": "model", "owned_by": "codex-proxy"}]}


@app.post("/v1/chat/completions")
async def chat_completions(payload: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(authorization)
    requested_model = str(payload.get("model") or "gpt-5.6-luna")
    output = await _run_codex(
        _prompt_from_messages(payload.get("messages")),
        _chat_schema(payload),
        _chat_requires_json(payload),
    )
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
    output = await _run_codex(
        _prompt_from_responses_input(payload.get("input")),
        _responses_schema(payload),
        _responses_requires_json(payload),
    )
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
