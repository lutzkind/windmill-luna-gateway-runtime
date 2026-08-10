from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import shutil
import tempfile
import time
from urllib.parse import urlparse
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException

app = FastAPI(title="Codex CLI OpenAI-compatible upstream", version="1.3.0")

API_KEY = os.environ.get("OPENAI_VIA_CODEX_API_KEY", "").strip()
CODEX_BINARY = os.environ.get("CODEX_BINARY", "codex").strip() or "codex"
CODEX_MODEL = os.environ.get("CODEX_MODEL", "").strip()
SOURCE_CODEX_HOME = Path(os.environ.get("CODEX_HOME", "/root/.codex").strip() or "/root/.codex")
CODEX_AUTH_SOURCE = Path(
    os.environ.get("CODEX_AUTH_SOURCE", str(SOURCE_CODEX_HOME / "auth.json")).strip()
    or str(SOURCE_CODEX_HOME / "auth.json")
)
RUNTIME_CODEX_HOME = Path(os.environ.get("LUNA_CODEX_HOME", "/tmp/luna-codex-home").strip() or "/tmp/luna-codex-home")
TIMEOUT_SECONDS = max(30, int(os.environ.get("CODEX_TIMEOUT_SECONDS", "180")))
MAX_CONCURRENCY = max(1, int(os.environ.get("CODEX_MAX_CONCURRENCY", "4")))
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENCY)
CODEX_REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high", "xhigh"})
WEB_SEARCH_TOOL_TYPES = frozenset({"web_search"})
MAX_IMAGE_INPUTS = max(1, min(8, int(os.environ.get("CODEX_MAX_IMAGE_INPUTS", "8"))))
MAX_IMAGE_BYTES = max(1_048_576, min(20 * 1024 * 1024, int(os.environ.get("CODEX_MAX_IMAGE_BYTES", str(15 * 1024 * 1024)))))
IMAGE_FETCH_TIMEOUT_SECONDS = max(5.0, min(60.0, float(os.environ.get("CODEX_IMAGE_FETCH_TIMEOUT_SECONDS", "30"))))
IMAGE_MIME_SUFFIXES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
IMAGE_MAGIC_TYPES = (
    (b"\xff\xd8\xff", "image/jpeg", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", "image/png", ".png"),
    (b"RIFF", "image/webp", ".webp"),
    (b"GIF87a", "image/gif", ".gif"),
    (b"GIF89a", "image/gif", ".gif"),
)


@dataclass
class CodexRun:
    text: str
    web_search_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


def _authorize(authorization: str | None) -> None:
    if not API_KEY:
        raise HTTPException(status_code=503, detail="upstream API key is not configured")
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="unauthorized")


def _image_source_from_part(item: dict[str, Any]) -> str | None:
    item_type = str(item.get("type") or "").strip().lower()
    if item_type not in {"image_url", "input_image"}:
        return None
    value: Any = item.get("image_url") or item.get("url") or item.get("file_url")
    if isinstance(value, dict):
        value = value.get("url") or value.get("image_url") or value.get("uri")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if item_type == "input_image" and item.get("file_id"):
        raise HTTPException(status_code=400, detail="file_id image inputs are unsupported; provide image_url")
    return None


def _content_text_and_images(content: Any) -> tuple[str, list[str]]:
    if isinstance(content, str):
        return content, []
    if isinstance(content, list):
        parts: list[str] = []
        images: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("input_text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
                image_source = _image_source_from_part(item)
                if image_source:
                    images.append(image_source)
        return "\n".join(part for part in parts if part), images
    if isinstance(content, dict):
        text = content.get("text") or content.get("input_text") or content.get("content")
        image_source = _image_source_from_part(content)
        return text if isinstance(text, str) else "", [image_source] if image_source else []
    return "", []


def _content_text(content: Any) -> str:
    return _content_text_and_images(content)[0]


def _web_search_requested(payload: dict[str, Any]) -> bool:
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return False
    unsupported = [
        tool.get("type")
        for tool in tools
        if isinstance(tool, dict) and tool.get("type") not in WEB_SEARCH_TOOL_TYPES
    ]
    if unsupported:
        raise HTTPException(status_code=400, detail="unsupported_tool")
    return any(isinstance(tool, dict) and tool.get("type") == "web_search" for tool in tools)


WEB_SEARCH_INSTRUCTION = """The caller explicitly requested web search. Use Codex's built-in web search tool before answering.
Do not use shell commands, local files, MCP tools, or arbitrary network access.
When returning JSON, include a top-level web_search_sources array when the requested format permits it. Each source must contain url, title, and snippet; snippet must be a verbatim excerpt from the search result that supports the answer. Do not invent source URLs or excerpts."""


def _prompt_and_images_from_messages(messages: Any, *, web_search: bool = False) -> tuple[str, list[str]]:
    if not isinstance(messages, list):
        return "", []
    sections: list[str] = [WEB_SEARCH_INSTRUCTION if web_search else "Use only the supplied messages. Do not use tools."]
    image_inputs: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").upper()
        text, message_images = _content_text_and_images(message.get("content"))
        if message_images:
            image_inputs.extend(message_images)
            image_note = "\n".join(
                f"[Image attachment {len(image_inputs) - len(message_images) + index + 1} is supplied to the vision model.]"
                for index in range(len(message_images))
            )
            text = f"{text}\n{image_note}" if text else image_note
        if text:
            sections.append(f"<{role}>\n{text}\n</{role}>")
    if len(image_inputs) > MAX_IMAGE_INPUTS:
        raise HTTPException(status_code=400, detail="too_many_image_inputs")
    return "\n\n".join(sections), image_inputs


def _prompt_from_messages(messages: Any, *, web_search: bool = False) -> str:
    return _prompt_and_images_from_messages(messages, web_search=web_search)[0]


def _prompt_and_images_from_responses_input(
    value: Any, *, instructions: Any = None, web_search: bool = False
) -> tuple[str, list[str]]:
    sections: list[str] = [WEB_SEARCH_INSTRUCTION if web_search else "Use only the supplied messages. Do not use tools."]
    image_inputs: list[str] = []
    instruction_text = _content_text(instructions)
    if instruction_text:
        sections.append(f"<INSTRUCTIONS>\n{instruction_text}\n</INSTRUCTIONS>")
    messages = value if isinstance(value, list) else [{"role": "user", "content": value}]
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").upper()
        text, message_images = _content_text_and_images(message.get("content"))
        if message_images:
            image_inputs.extend(message_images)
            image_note = "\n".join(
                f"[Image attachment {len(image_inputs) - len(message_images) + index + 1} is supplied to the vision model.]"
                for index in range(len(message_images))
            )
            text = f"{text}\n{image_note}" if text else image_note
        if text:
            sections.append(f"<{role}>\n{text}\n</{role}>")
    if len(image_inputs) > MAX_IMAGE_INPUTS:
        raise HTTPException(status_code=400, detail="too_many_image_inputs")
    return "\n\n".join(sections), image_inputs


def _prompt_from_responses_input(
    value: Any, *, instructions: Any = None, web_search: bool = False
) -> str:
    return _prompt_and_images_from_responses_input(
        value, instructions=instructions, web_search=web_search
    )[0]


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
    RUNTIME_CODEX_HOME.mkdir(parents=True, exist_ok=True)
    target_auth = RUNTIME_CODEX_HOME / "auth.json"
    if not target_auth.is_file():
        source_auth = CODEX_AUTH_SOURCE
        if not source_auth.is_file():
            raise HTTPException(status_code=503, detail="codex authentication is unavailable")
        shutil.copy2(source_auth, target_auth)
    target_auth.chmod(0o600)
    # Intentionally do not copy config.toml. Completion jobs must never load MCP tools.


def _codex_reasoning_effort(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    normalized = str(value).strip().lower()
    if normalized == "none":
        return "minimal"
    if normalized not in CODEX_REASONING_EFFORTS:
        raise HTTPException(status_code=400, detail="invalid_reasoning_effort")
    return normalized


def _build_codex_command(
    output_path: str,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
    schema_path: Path | None = None,
    json_events: bool = False,
    image_paths: list[Path] | None = None,
) -> list[str]:
    command = [
        CODEX_BINARY,
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "-c",
        'approval_policy="never"',
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "--output-last-message",
        output_path,
    ]
    if json_events:
        command.append("--json")
    selected_model = str(model or CODEX_MODEL).strip()
    if selected_model:
        command.extend(["--model", selected_model])
    selected_effort = _codex_reasoning_effort(reasoning_effort)
    if selected_effort:
        command.extend(["-c", f'model_reasoning_effort="{selected_effort}"'])
    if schema_path is not None:
        command.extend(["--output-schema", str(schema_path)])
    for image_path in image_paths or []:
        command.extend(["--image", str(image_path)])
    return command


def _decode_data_image(source: str) -> tuple[bytes, str, str]:
    header, separator, encoded = source.partition(",")
    if not separator or not header.startswith("data:") or ";base64" not in header.lower():
        raise HTTPException(status_code=400, detail="invalid_data_image_input")
    mime = header[5:].split(";", 1)[0].strip().lower()
    if mime not in IMAGE_MIME_SUFFIXES:
        raise HTTPException(status_code=400, detail="unsupported_image_type")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="invalid_data_image_input") from exc
    return content, mime, IMAGE_MIME_SUFFIXES[mime]


def _image_type_from_bytes(content: bytes, content_type: str = "") -> tuple[str, str] | None:
    normalized_type = content_type.split(";", 1)[0].strip().lower()
    if normalized_type in IMAGE_MIME_SUFFIXES:
        return normalized_type, IMAGE_MIME_SUFFIXES[normalized_type]
    for magic, mime, suffix in IMAGE_MAGIC_TYPES:
        if content.startswith(magic) or (mime == "image/webp" and content.startswith(b"RIFF") and b"WEBP" in content[:16]):
            return mime, suffix
    return None


async def _fetch_image_url(source: str) -> tuple[bytes, str, str]:
    parsed = urlparse(source)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="unsupported_image_url")
    try:
        async with httpx.AsyncClient(
            timeout=IMAGE_FETCH_TIMEOUT_SECONDS,
            follow_redirects=True,
        ) as client:
            async with client.stream("GET", source) as response:
                if response.status_code < 200 or response.status_code >= 300:
                    raise HTTPException(status_code=400, detail="image_url_fetch_failed")
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_IMAGE_BYTES:
                        raise HTTPException(status_code=413, detail="image_input_too_large")
                    chunks.append(chunk)
                content = b"".join(chunks)
                image_type = _image_type_from_bytes(content, response.headers.get("content-type", ""))
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=400, detail="image_url_fetch_failed") from exc
    if image_type is None:
        raise HTTPException(status_code=400, detail="unsupported_image_type")
    return content, image_type[0], image_type[1]


async def _materialize_image_inputs(sources: list[str], tmp_path: Path) -> list[Path]:
    if len(sources) > MAX_IMAGE_INPUTS:
        raise HTTPException(status_code=400, detail="too_many_image_inputs")
    paths: list[Path] = []
    for index, source in enumerate(sources, start=1):
        if source.startswith("data:"):
            content, _mime, suffix = _decode_data_image(source)
        else:
            content, _mime, suffix = await _fetch_image_url(source)
        if not content or len(content) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="image_input_too_large")
        image_path = tmp_path / f"input-image-{index}{suffix}"
        image_path.write_bytes(content)
        paths.append(image_path)
    return paths


def _parse_codex_events(stdout_text: str) -> tuple[list[dict[str, Any]], dict[str, int], str]:
    web_search_calls: list[dict[str, Any]] = []
    usage: dict[str, int] = {}
    agent_message = ""
    for line in stdout_text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = {
                str(key): int(value)
                for key, value in event["usage"].items()
                if isinstance(value, (int, float))
            }
        if event.get("type") != "item.completed" or not isinstance(event.get("item"), dict):
            continue
        item = event["item"]
        if item.get("type") == "web_search":
            web_search_calls.append({
                "id": item.get("id") or f"websearch_{uuid.uuid4().hex}",
                "type": "web_search_call",
                "status": "completed",
                "action": item.get("action") if isinstance(item.get("action"), dict) else {},
            })
        elif item.get("type") == "agent_message" and isinstance(item.get("text"), str):
            agent_message = item["text"]
    return web_search_calls, usage, agent_message


def _source_records_from_output(text: str) -> list[dict[str, str]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, dict):
        return []

    candidates: list[Any] = []
    for key in ("web_search_sources", "sources", "citations"):
        if isinstance(value.get(key), list):
            candidates.extend(value[key])
    if value.get("source_url"):
        candidates.append(value)

    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        url = str(candidate.get("url") or candidate.get("source_url") or "").strip()
        title = str(candidate.get("title") or candidate.get("source_title") or "").strip()
        snippet = str(
            candidate.get("snippet")
            or candidate.get("source_excerpt")
            or candidate.get("excerpt")
            or ""
        ).strip()
        if not (url.startswith("https://") or url.startswith("http://")) or not title or not snippet:
            continue
        if url in seen:
            continue
        seen.add(url)
        records.append({"url": url[:2000], "title": title[:1000], "snippet": snippet[:5000]})
    return records[:20]


async def _run_codex_once(
    prompt: str,
    schema: dict[str, Any] | None,
    require_json: bool,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
    web_search: bool = False,
    image_inputs: list[str] | None = None,
) -> CodexRun:
    _prepare_runtime_home()
    with tempfile.TemporaryDirectory(prefix="luna-codex-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        output_path = tmp_path / "final.txt"
        schema_path: Path | None = None
        if schema is not None:
            schema_path = tmp_path / "schema.json"
            schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
            prompt = (
                "Return one JSON object matching the supplied schema.\n\n" + prompt
            )
        elif require_json:
            prompt = (
                "Return one valid JSON object only.\n\n" + prompt
            )
        image_paths = await _materialize_image_inputs(image_inputs or [], tmp_path)
        command = _build_codex_command(
            str(output_path),
            model=model,
            reasoning_effort=reasoning_effort,
            schema_path=schema_path,
            json_events=web_search,
            image_paths=image_paths,
        )
        print(json.dumps({
            "event": "codex_invocation",
            "model": str(model or CODEX_MODEL).strip() or None,
            "reasoning_effort": _codex_reasoning_effort(reasoning_effort),
            "sandbox": "read-only",
            "approval_policy": "never",
            "image_count": len(image_paths),
        }, sort_keys=True))

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
        web_search_calls, usage, agent_message = _parse_codex_events(stdout_text) if web_search else ([], {}, "")
        output = output_path.read_text(encoding="utf-8").strip() if output_path.is_file() else agent_message or stdout_text
        if process.returncode != 0:
            detail = error_text[-1200:] or stdout_text[-1200:] or f"codex exited {process.returncode}"
            raise HTTPException(status_code=502, detail=detail)
        if not output:
            raise HTTPException(status_code=502, detail="codex returned an empty response")
        if web_search and not web_search_calls:
            raise HTTPException(status_code=502, detail="codex did not execute the requested web search")
        sources = _source_records_from_output(output) if web_search else []
        for call in web_search_calls:
            call["results"] = sources
        return CodexRun(text=output, web_search_calls=web_search_calls, usage=usage)


def _as_codex_run(value: CodexRun | str) -> CodexRun:
    return value if isinstance(value, CodexRun) else CodexRun(text=str(value))


async def _run_codex(
    prompt: str,
    schema: dict[str, Any] | None = None,
    require_json: bool = False,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
    web_search: bool = False,
    image_inputs: list[str] | None = None,
) -> CodexRun:
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="empty prompt")
    if shutil.which(CODEX_BINARY) is None:
        raise HTTPException(status_code=503, detail="codex binary unavailable")

    async with SEMAPHORE:
        output = _as_codex_run(await _run_codex_once(
            prompt,
            schema,
            require_json,
            model=model,
            reasoning_effort=reasoning_effort,
            web_search=web_search,
            image_inputs=image_inputs,
        ))
        if not require_json:
            return output
        try:
            parsed = json.loads(output.text)
        except json.JSONDecodeError:
            schema_instruction = " satisfying the supplied schema" if schema is not None else ""
            repair_prompt = (
                "The prior attempt was not valid JSON. Redo the original task from scratch and return exactly one "
                f"valid JSON object{schema_instruction}. No markdown or surrounding prose.\n\n"
                f"ORIGINAL TASK:\n{prompt}\n\nINVALID PRIOR OUTPUT:\n{output.text[:6000]}"
            )
            output = _as_codex_run(await _run_codex_once(
                repair_prompt,
                schema,
                True,
                model=model,
                reasoning_effort=reasoning_effort,
                web_search=web_search,
                image_inputs=image_inputs,
            ))
            try:
                parsed = json.loads(output.text)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=502, detail=f"codex returned invalid structured output: {exc.msg}")
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=502, detail="codex structured output was not a JSON object")
        output.text = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        return output


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "ok": shutil.which(CODEX_BINARY) is not None and CODEX_AUTH_SOURCE.is_file(),
        "provider": "official-codex-cli",
        "binary": CODEX_BINARY,
        "max_concurrency": MAX_CONCURRENCY,
        "structured_output": True,
        "mcp_tools_loaded": False,
        "reasoning_forwarding": True,
        "model_forwarding": True,
        "web_search_forwarding": True,
        "image_input_forwarding": True,
        "image_transport": "codex_exec_image_flags",
        "max_image_inputs": MAX_IMAGE_INPUTS,
        "max_image_bytes": MAX_IMAGE_BYTES,
        "sandbox": "read-only",
    }


@app.get("/v1/models")
async def models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(authorization)
    return {"object": "list", "data": [{"id": "gpt-5.6-luna", "object": "model", "owned_by": "codex-proxy"}]}


@app.post("/v1/chat/completions")
async def chat_completions(payload: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(authorization)
    requested_model = str(payload.get("model") or "gpt-5.6-luna")
    web_search = _web_search_requested(payload)
    prompt, image_inputs = _prompt_and_images_from_messages(
        payload.get("messages"), web_search=web_search
    )
    run = await _run_codex(
        prompt,
        _chat_schema(payload),
        _chat_requires_json(payload),
        model=requested_model,
        reasoning_effort=payload.get("reasoning_effort"),
        web_search=web_search,
        image_inputs=image_inputs,
    )
    return {
        "id": f"chatcmpl_{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": run.text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": run.usage.get("input_tokens", 0),
            "completion_tokens": run.usage.get("output_tokens", 0),
            "total_tokens": run.usage.get("input_tokens", 0) + run.usage.get("output_tokens", 0),
            "prompt_tokens_details": {"cached_tokens": run.usage.get("cached_input_tokens", 0)},
        },
    }


@app.post("/v1/responses")
async def responses(payload: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(authorization)
    requested_model = str(payload.get("model") or "gpt-5.6-luna")
    web_search = _web_search_requested(payload)
    prompt, image_inputs = _prompt_and_images_from_responses_input(
        payload.get("input"),
        instructions=payload.get("instructions"),
        web_search=web_search,
    )
    run = await _run_codex(
        prompt,
        _responses_schema(payload),
        _responses_requires_json(payload),
        model=requested_model,
        reasoning_effort=(payload.get("reasoning") or {}).get("effort")
        if isinstance(payload.get("reasoning"), dict)
        else None,
        web_search=web_search,
        image_inputs=image_inputs,
    )
    response_id = f"resp_{uuid.uuid4().hex}"
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": requested_model,
        "output_text": run.text,
        "output": [*run.web_search_calls, {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": run.text, "annotations": [
                {"type": "url_citation", "url": result["url"], "title": result["title"], "start_index": 0, "end_index": 0}
                for call in run.web_search_calls for result in call.get("results", [])
            ]}],
        }],
        "usage": {
            "input_tokens": run.usage.get("input_tokens", 0),
            "output_tokens": run.usage.get("output_tokens", 0),
            "total_tokens": run.usage.get("input_tokens", 0) + run.usage.get("output_tokens", 0),
            "input_tokens_details": {"cached_tokens": run.usage.get("cached_input_tokens", 0)},
        },
    }
