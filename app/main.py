from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from jsonschema import ValidationError, validate as validate_json_schema
LOGGER = logging.getLogger(__name__)

EndpointKind = Literal["chat", "responses"]
ProviderName = Literal["codex", "openai"]
API_PASSTHROUGH_METHODS: dict[str, frozenset[str]] = {
    "audio/transcriptions": frozenset({"POST"}),
    "audio/translations": frozenset({"POST"}),
    "audio/speech": frozenset({"POST"}),
    "embeddings": frozenset({"POST"}),
    "images/edits": frozenset({"POST"}),
    "models": frozenset({"GET"}),
}
MAX_PASSTHROUGH_BODY_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_BODY_BYTES = 16 * 1024 * 1024
REASONING_EFFORTS = frozenset({"none", "low", "medium", "high"})
HIGH_REASONING_MARKERS = (
    "learning optimizer",
    "conversation learning",
    "outreach optimizer",
    "strategy optimizer",
    "cross-conversation",
    "root cause analysis",
    "compare competing",
    "multi-step decision",
)
MEDIUM_REASONING_MARKERS = (
    "research",
    "website review",
    "review the website",
    "social topic",
    "social post",
    "journal",
    "weekly summary",
    "synthesize",
    "synthesis",
    "call summary",
    "transcription summary",
    "content engine",
)



@dataclass(frozen=True)
class Settings:
    allowed_api_key_sha256s: frozenset[str]
    codex_url: str
    codex_api_key: str
    openai_url: str
    server_openai_api_key: str
    allowed_models: frozenset[str]
    model_aliases: dict[str, str]
    timeout_seconds: float
    max_body_bytes: int
    max_concurrency: int
    transient_failure_threshold: int
    transient_failure_window_seconds: int
    transient_open_seconds: int
    quota_open_seconds: int
    auth_open_seconds: int
    enable_test_controls: bool

    @classmethod
    def from_env(cls) -> "Settings":
        aliases_raw = os.getenv(
            "MODEL_ALIASES_JSON",
            '{"luna-auto":"gpt-5.6-luna","gpt-5.6-luna":"gpt-5.6-luna"}',
        )
        aliases = json.loads(aliases_raw)
        if not isinstance(aliases, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in aliases.items()
        ):
            raise RuntimeError(
                "MODEL_ALIASES_JSON must be a string-to-string JSON object"
            )

        allowed = frozenset(
            part.strip()
            for part in os.getenv(
                "ALLOWED_MODELS", "gpt-5.6-luna,luna-auto"
            ).split(",")
            if part.strip()
        )
        configured_key_hashes = os.getenv("ALLOWED_API_KEY_SHA256S", "")
        additional_key_hashes = os.getenv(
            "ADDITIONAL_ALLOWED_API_KEY_SHA256S",
            "68471dcada6d4d3b0468b20ad6ea6fc2ea21451d4d7a28b7fdb63be763679817,f777774c7a4100fc25022f34d27483a9080679aed01a0fca54ced407ca09df9f",
        )
        allowed_key_hashes = frozenset(
            part.strip().lower()
            for part in f"{configured_key_hashes},{additional_key_hashes}".split(",")
            if part.strip()
        )
        return cls(
            allowed_api_key_sha256s=allowed_key_hashes,
            codex_url=os.getenv(
                "CODEX_UPSTREAM_URL", "http://codex-upstream:18080/v1"
            ).rstrip("/"),
            codex_api_key=os.getenv(
                "CODEX_UPSTREAM_API_KEY", "internal-codex-sidecar-v1"
            ).strip(),
            openai_url=os.getenv(
                "OPENAI_API_BASE_URL", "https://api.openai.com/v1"
            ).rstrip("/"),
            server_openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            allowed_models=allowed,
            model_aliases=aliases,
            timeout_seconds=float(
                os.getenv("PROVIDER_TIMEOUT_SECONDS", "180")
            ),
            max_body_bytes=int(
                os.getenv("MAX_BODY_BYTES", str(DEFAULT_MAX_BODY_BYTES))
            ),
            max_concurrency=max(
                1, int(os.getenv("MAX_CONCURRENCY", "4"))
            ),
            transient_failure_threshold=max(
                1, int(os.getenv("TRANSIENT_FAILURE_THRESHOLD", "3"))
            ),
            transient_failure_window_seconds=max(
                1, int(os.getenv("TRANSIENT_FAILURE_WINDOW_SECONDS", "300"))
            ),
            transient_open_seconds=max(
                1, int(os.getenv("TRANSIENT_OPEN_SECONDS", "900"))
            ),
            quota_open_seconds=max(
                1, int(os.getenv("QUOTA_OPEN_SECONDS", "1800"))
            ),
            auth_open_seconds=max(
                1, int(os.getenv("AUTH_OPEN_SECONDS", "300"))
            ),
            enable_test_controls=os.getenv(
                "ENABLE_TEST_CONTROLS", "false"
            ).strip().lower()
            in {"1", "true", "yes", "on"},
        )


@dataclass
class CircuitBreaker:
    settings: Settings
    open_until_epoch: float = 0.0
    open_reason: str | None = None
    transient_failures: deque[float] = field(default_factory=deque)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def snapshot(self) -> dict[str, Any]:
        async with self.lock:
            now = time.time()
            is_open = self.open_until_epoch > now
            return {
                "open": is_open,
                "reason": self.open_reason if is_open else None,
                "open_until": self.open_until_epoch if is_open else None,
            }

    async def should_skip(self) -> tuple[bool, str | None]:
        async with self.lock:
            now = time.time()
            if self.open_until_epoch > now:
                return True, self.open_reason
            self.open_until_epoch = 0.0
            self.open_reason = None
            return False, None

    async def success(self) -> None:
        async with self.lock:
            self.transient_failures.clear()
            self.open_until_epoch = 0.0
            self.open_reason = None

    async def failure(self, reason: str) -> None:
        async with self.lock:
            now = time.time()
            if reason == "quota":
                self.open_until_epoch = now + self.settings.quota_open_seconds
                self.open_reason = reason
                self.transient_failures.clear()
                return
            if reason == "auth":
                self.open_until_epoch = now + self.settings.auth_open_seconds
                self.open_reason = reason
                self.transient_failures.clear()
                return
            if reason not in {
                "timeout",
                "network",
                "upstream_5xx",
                "invalid_success",
                "rate_limit",
            }:
                return

            cutoff = now - self.settings.transient_failure_window_seconds
            while (
                self.transient_failures
                and self.transient_failures[0] < cutoff
            ):
                self.transient_failures.popleft()
            self.transient_failures.append(now)
            if (
                len(self.transient_failures)
                >= self.settings.transient_failure_threshold
            ):
                self.open_until_epoch = (
                    now + self.settings.transient_open_seconds
                )
                self.open_reason = reason
                self.transient_failures.clear()


@dataclass
class ProviderResult:
    response: httpx.Response | None
    error_reason: str | None = None
    error_detail: str | None = None


def _prompt_text(value: Any, *, limit: int = 50000) -> str:
    parts: list[str] = []
    length = 0

    def visit(item: Any) -> None:
        nonlocal length
        if length >= limit:
            return
        if isinstance(item, str):
            remaining = limit - length
            part = item[:remaining]
            parts.append(part)
            length += len(part)
        elif isinstance(item, dict):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    if isinstance(value, dict):
        for key in ("instructions", "input", "messages"):
            visit(value.get(key))
    return "\n".join(parts).lower()


def _explicit_reasoning_effort(
    kind: EndpointKind, payload: dict[str, Any]
) -> str | None:
    effort: Any = None
    if kind == "chat":
        effort = payload.get("reasoning_effort")
    else:
        reasoning = payload.get("reasoning")
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort")
    if effort is None:
        return None
    if not isinstance(effort, str) or effort not in REASONING_EFFORTS:
        raise HTTPException(
            status_code=400, detail="invalid_reasoning_effort"
        )
    return effort


def select_reasoning_effort(
    kind: EndpointKind, payload: dict[str, Any]
) -> tuple[str, str]:
    explicit = _explicit_reasoning_effort(kind, payload)
    if explicit:
        return explicit, "explicit"

    text = _prompt_text(payload)
    if any(marker in text for marker in HIGH_REASONING_MARKERS):
        return "high", "adaptive"

    tools = payload.get("tools")
    has_web_search = isinstance(tools, list) and any(
        isinstance(tool, dict) and tool.get("type") == "web_search"
        for tool in tools
    )
    messages = payload.get("messages")
    message_count = len(messages) if isinstance(messages, list) else 0
    if (
        has_web_search
        or any(marker in text for marker in MEDIUM_REASONING_MARKERS)
        or len(text) > 12000
        or message_count > 6
    ):
        return "medium", "adaptive"
    return "low", "adaptive"


def apply_reasoning_policy(
    kind: EndpointKind, payload: dict[str, Any]
) -> tuple[dict[str, Any], str, str]:
    effort, source = select_reasoning_effort(kind, payload)
    normalized = dict(payload)
    if kind == "chat":
        normalized["reasoning_effort"] = effort
    else:
        reasoning = normalized.get("reasoning")
        updated = dict(reasoning) if isinstance(reasoning, dict) else {}
        updated["effort"] = effort
        normalized["reasoning"] = updated
    return normalized, effort, source


class Gateway:
    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.settings = settings
        self.circuit = CircuitBreaker(settings)
        self.image_circuit = CircuitBreaker(settings)
        self.semaphore = asyncio.Semaphore(settings.max_concurrency)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.timeout_seconds),
            transport=transport,
            follow_redirects=False,
        )

    async def close(self) -> None:
        await self.client.aclose()

    def normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        model = payload.get("model")
        if not isinstance(model, str) or not model.strip():
            raise HTTPException(status_code=400, detail="model is required")
        model = model.strip()
        normalized = dict(payload)
        normalized["model"] = self.settings.model_aliases.get(model, model)
        if normalized.get("stream") is True:
            raise HTTPException(
                status_code=400, detail="streaming_not_supported"
            )
        if normalized.get("store") is True:
            raise HTTPException(
                status_code=400, detail="persistent_storage_not_supported"
            )
        if normalized.get("previous_response_id") is not None:
            raise HTTPException(
                status_code=400, detail="previous_response_id_not_supported"
            )
        return normalized

    async def request(
        self,
        kind: EndpointKind,
        payload: dict[str, Any],
        request_id: str,
        fallback_api_key: str,
        forced_failure: str | None = None,
    ) -> Response:
        normalized = self.normalize_payload(payload)
        codex_models = frozenset(self.settings.model_aliases.values())
        if normalized["model"] not in codex_models:
            return await self._api_only_json(
                kind=kind,
                payload=normalized,
                request_id=request_id,
                api_key=fallback_api_key,
            )

        skip, skip_reason = await self.circuit.should_skip()
        normalized, reasoning_effort, reasoning_source = (
            apply_reasoning_policy(kind, normalized)
        )
        skip, skip_reason = await self.circuit.should_skip()
        async with self.semaphore:
            if forced_failure:
                reason = forced_failure
                await self.circuit.failure(reason)
                fallback_reason = f"forced_test:{reason}"
            elif not skip:
                codex_result = await self._call_provider(
                    provider="codex",
                    kind=kind,
                    payload=normalized,
                    request_id=request_id,
                    api_key=self.settings.codex_api_key,
                )
                if (
                    codex_result.response is not None
                    and 200 <= codex_result.response.status_code < 300
                ):
                    invalid = validate_success(
                        kind, normalized, codex_result.response
                    )
                    if invalid is None:
                        await self.circuit.success()
                        return relay_response(
                            codex_result.response,
                            provider="codex",
                            fallback_used=False,
                            fallback_reason=None,
                            request_id=request_id,
                        )
                    codex_result = ProviderResult(
                        response=codex_result.response,
                        error_reason="invalid_success",
                        error_detail=invalid,
                    )

                reason = classify_failure(codex_result)
                if reason is None:
                    assert codex_result.response is not None
                    return relay_response(
                        codex_result.response,
                        provider="codex",
                        fallback_used=False,
                        fallback_reason=None,
                        request_id=request_id,
                    )
                await self.circuit.failure(reason)
                fallback_reason = reason
            else:
                fallback_reason = (
                    f"circuit_open:{skip_reason or 'unknown'}"
                )

            if not fallback_api_key:
                return json_error(
                    status_code=502,
                    message=(
                        "Codex was unavailable and no OpenAI API fallback "
                        "credential was supplied."
                    ),
                    code="fallback_key_missing",
                    request_id=request_id,
                    fallback_reason=fallback_reason,
                )

            api_result = await self._call_provider(
                provider="openai",
                kind=kind,
                payload=normalized,
                request_id=request_id,
                api_key=fallback_api_key,
            )
            if api_result.response is None:
                return json_error(
                    status_code=502,
                    message="Both Codex and OpenAI API providers were unavailable.",
                    code="all_providers_unavailable",
                    request_id=request_id,
                    fallback_reason=fallback_reason,
                )
            return relay_response(
                api_result.response,
                provider="openai-api",
                fallback_used=True,
                fallback_reason=fallback_reason,
                request_id=request_id,
            )

    async def _api_only_json(
        self,
        *,
        kind: EndpointKind,
        payload: dict[str, Any],
        request_id: str,
        api_key: str,
    ) -> Response:
        if not api_key:
            return json_error(status_code=502, message="No OpenAI API credential was supplied.", code="api_key_missing", request_id=request_id, fallback_reason="api_only_model")
        async with self.semaphore:
            result = await self._call_provider(provider="openai", kind=kind, payload=payload, request_id=request_id, api_key=api_key)
        if result.response is None:
            return json_error(status_code=502, message="The OpenAI API provider was unavailable.", code="api_provider_unavailable", request_id=request_id, fallback_reason=result.error_reason or "api_only_model")
        return relay_response(result.response, provider="openai-api", fallback_used=False, fallback_reason=None, request_id=request_id)

    async def generate_image(
        self,
        *,
        payload: dict[str, Any],
        request_id: str,
        api_key: str,
    ) -> Response:
        skip, skip_reason = await self.image_circuit.should_skip()
        fallback_reason: str | None = None
        async with self.semaphore:
            if not skip:
                codex_response: httpx.Response | None = None
                reason: str | None = None
                try:
                    codex_response = await self.client.post(
                        f"{self.settings.codex_url}/images/generations",
                        headers={
                            "Authorization": f"Bearer {self.settings.codex_api_key}",
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                            "X-Request-ID": request_id,
                        },
                        json=payload,
                    )
                except httpx.TimeoutException:
                    reason = "timeout"
                except httpx.HTTPError:
                    reason = "network"

                if codex_response is not None:
                    if 200 <= codex_response.status_code < 300:
                        if validate_image_success(codex_response):
                            await self.image_circuit.success()
                            return relay_response(
                                codex_response,
                                provider="codex-image",
                                fallback_used=False,
                                fallback_reason=None,
                                request_id=request_id,
                            )
                        reason = "invalid_success"
                    else:
                        reason = classify_image_failure_response(codex_response)
                        if reason is None:
                            return relay_response(
                                codex_response,
                                provider="codex-image",
                                fallback_used=False,
                                fallback_reason=None,
                                request_id=request_id,
                            )

                reason = reason or "network"
                await self.image_circuit.failure(reason)
                fallback_reason = f"image_{reason}"
            else:
                fallback_reason = f"image_circuit_open:{skip_reason or 'unknown'}"

            if not api_key:
                return json_error(
                    status_code=502,
                    message="Codex image generation was unavailable and no OpenAI API fallback credential was supplied.",
                    code="image_fallback_key_missing",
                    request_id=request_id,
                    fallback_reason=fallback_reason,
                )
            try:
                api_response = await self.client.post(
                    f"{self.settings.openai_url}/images/generations",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "X-Request-ID": request_id,
                    },
                    json=payload,
                )
            except httpx.TimeoutException:
                return json_error(
                    status_code=504,
                    message="The OpenAI Images API fallback timed out.",
                    code="image_api_timeout",
                    request_id=request_id,
                    fallback_reason=fallback_reason,
                )
            except httpx.HTTPError:
                return json_error(
                    status_code=502,
                    message="The OpenAI Images API fallback request failed.",
                    code="image_api_network",
                    request_id=request_id,
                    fallback_reason=fallback_reason,
                )
            return relay_response(
                api_response,
                provider="openai-api",
                fallback_used=True,
                fallback_reason=fallback_reason,
                request_id=request_id,
            )

    async def proxy_openai_api(
        self,
        *,
        path: str,
        method: str,
        body: bytes,
        request_headers: dict[str, str],
        request_id: str,
        api_key: str,
    ) -> Response:
        if not api_key:
            return json_error(status_code=502, message="No OpenAI API credential was supplied.", code="api_key_missing", request_id=request_id, fallback_reason="api_passthrough")
        headers = {"Authorization": f"Bearer {api_key}", "Accept": request_headers.get("accept", "application/json"), "X-Request-ID": request_id}
        for name in ("content-type", "openai-organization", "openai-project", "idempotency-key"):
            value = request_headers.get(name)
            if value:
                headers[name] = value
        try:
            async with self.semaphore:
                response = await self.client.request(method, f"{self.settings.openai_url}/{path}", headers=headers, content=body if method != "GET" else None)
        except httpx.TimeoutException:
            return json_error(status_code=504, message="The OpenAI API request timed out.", code="api_timeout", request_id=request_id, fallback_reason="api_passthrough")
        except httpx.HTTPError:
            return json_error(status_code=502, message="The OpenAI API request failed.", code="api_network", request_id=request_id, fallback_reason="api_passthrough")
        return relay_response(response, provider="openai-api", fallback_used=False, fallback_reason=None, request_id=request_id)

    async def _call_provider(
        self,
        *,
        provider: ProviderName,
        kind: EndpointKind,
        payload: dict[str, Any],
        request_id: str,
        api_key: str,
    ) -> ProviderResult:
        if not api_key:
            return ProviderResult(
                None, "auth", f"{provider}_api_key_missing"
            )

        path = "/chat/completions" if kind == "chat" else "/responses"
        base = (
            self.settings.codex_url
            if provider == "codex"
            else self.settings.openai_url
        )
        try:
            response = await self.client.post(
                f"{base}{path}",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-Request-ID": request_id,
                },
                json=payload,
            )
            return ProviderResult(response=response)
        except httpx.TimeoutException as exc:
            LOGGER.warning(
                "provider timeout provider=%s request_id=%s",
                provider,
                request_id,
            )
            return ProviderResult(
                None, "timeout", exc.__class__.__name__
            )
        except httpx.HTTPError as exc:
            LOGGER.warning(
                "provider network error provider=%s request_id=%s",
                provider,
                request_id,
            )
            return ProviderResult(
                None, "network", exc.__class__.__name__
            )


def validate_image_success(response: httpx.Response) -> bool:
    try:
        data = response.json()
    except ValueError:
        return False
    if not isinstance(data, dict):
        return False
    rows = data.get("data")
    return isinstance(rows, list) and bool(rows) and isinstance(rows[0], dict) and bool(rows[0].get("b64_json"))


def classify_image_failure_response(response: httpx.Response) -> str | None:
    status = response.status_code
    text = response.text[:8000].lower()
    if status == 429:
        quota_terms = ("image_gen", "usage limit", "quota", "plan limit", "insufficient_quota")
        return "quota" if any(term in text for term in quota_terms) else "rate_limit"
    if status in {401, 403}:
        return "auth"
    if status == 404:
        return "capability"
    if status >= 500:
        return "upstream_5xx"
    return None


def classify_failure(result: ProviderResult) -> str | None:
    if result.error_reason:
        reason = result.error_reason
        status = 504 if reason == "timeout" else 503 if reason == "auth" else 502
        detail = result.error_detail or f"Codex failed with {reason}"
        result.response = httpx.Response(
            status, json={"error": {"message": detail, "type": "codex_provider_error", "code": reason}}
        )
        return None
    if result.response is None:
        result.response = httpx.Response(
            502, json={"error": {"message": "Codex provider returned no response", "type": "codex_provider_error", "code": "network"}}
        )
        return None
    if result.response.status_code == 429:
        text = result.response.text[:4000].lower()
        quota_terms = (
            "usage limit",
            "quota",
            "plan limit",
            "insufficient_quota",
            "codex usage",
            "weekly limit",
            "weighted tokens left",
        )
        if any(term in text for term in quota_terms):
            return "quota"
    return None


def validate_success(
    kind: EndpointKind,
    request_payload: dict[str, Any],
    response: httpx.Response,
) -> str | None:
    try:
        data = response.json()
    except ValueError:
        return "provider returned non-JSON success"
    if not isinstance(data, dict):
        return "provider returned a non-object success"

    if kind == "chat":
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return "chat success has no choices"
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        if not isinstance(message, dict):
            return "chat success has no message"
        text = extract_chat_text(message)
        if (
            text is None
            and not message.get("tool_calls")
            and not message.get("function_call")
        ):
            return "chat success has neither text nor tool calls"
        return validate_structured_output(request_payload, text)

    output = data.get("output")
    if not isinstance(output, list):
        return "responses success has no output list"
    text = extract_response_text(data)
    if not output and not text:
        return "responses success is empty"
    return validate_structured_output(request_payload, text)


def extract_chat_text(message: dict[str, Any]) -> str | None:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(
                item.get("text"), str
            ):
                parts.append(item["text"])
        return "".join(parts) if parts else None
    return None


def extract_response_text(data: dict[str, Any]) -> str | None:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]

    parts: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts) if parts else None


def validate_structured_output(
    payload: dict[str, Any], text: str | None
) -> str | None:
    schema: dict[str, Any] | None = None
    requires_json = False

    response_format = payload.get("response_format")
    if isinstance(response_format, dict):
        format_type = response_format.get("type")
        if format_type in {"json_object", "json_schema"}:
            requires_json = True
        if format_type == "json_schema":
            json_schema = response_format.get("json_schema")
            if (
                isinstance(json_schema, dict)
                and isinstance(json_schema.get("schema"), dict)
            ):
                schema = json_schema["schema"]

    text_config = payload.get("text")
    if (
        isinstance(text_config, dict)
        and isinstance(text_config.get("format"), dict)
    ):
        text_format = text_config["format"]
        format_type = text_format.get("type")
        if format_type in {"json_object", "json_schema"}:
            requires_json = True
        if (
            format_type == "json_schema"
            and isinstance(text_format.get("schema"), dict)
        ):
            schema = text_format["schema"]

    if not requires_json:
        return None
    if not isinstance(text, str) or not text.strip():
        return "structured output is empty"

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return "structured output is not valid JSON"

    if schema is not None:
        try:
            validate_json_schema(parsed, schema)
        except ValidationError as exc:
            return (
                "structured output failed schema validation: "
                f"{exc.message}"
            )
    return None


def gateway_headers(
    *,
    provider: str,
    fallback_used: bool,
    fallback_reason: str | None,
    request_id: str,
) -> dict[str, str]:
    headers = {
        "X-Luna-Gateway-Provider": provider,
        "X-Luna-Gateway-Fallback": (
            "true" if fallback_used else "false"
        ),
        "X-Request-ID": request_id,
        "Cache-Control": "no-store",
    }
    if fallback_reason:
        headers["X-Luna-Gateway-Fallback-Reason"] = (
            fallback_reason[:200]
        )
    return headers


def relay_response(
    upstream: httpx.Response,
    *,
    provider: str,
    fallback_used: bool,
    fallback_reason: str | None,
    request_id: str,
) -> Response:
    headers = gateway_headers(
        provider=provider,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        request_id=request_id,
    )
    headers["Content-Type"] = upstream.headers.get(
        "content-type", "application/json"
    )
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=headers,
    )


def json_error(
    *,
    status_code: int,
    message: str,
    code: str,
    request_id: str,
    fallback_reason: str | None,
) -> Response:
    payload = {
        "error": {
            "message": message,
            "type": "gateway_provider_error",
            "code": code,
        }
    }
    return Response(
        content=json.dumps(payload),
        status_code=status_code,
        media_type="application/json",
        headers=gateway_headers(
            provider="none",
            fallback_used=True,
            fallback_reason=fallback_reason,
            request_id=request_id,
        ),
    )


def require_gateway_auth(request: Request, settings: Settings) -> None:
    if not settings.allowed_api_key_sha256s:
        raise HTTPException(
            status_code=503, detail="gateway_not_configured"
        )
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    supplied = token.strip() if scheme.lower() == "bearer" else ""
    if not supplied:
        raise HTTPException(status_code=401, detail="unauthorized")
    fingerprint = hashlib.sha256(supplied.encode("utf-8")).hexdigest()
    if not any(
        hmac.compare_digest(fingerprint, allowed)
        for allowed in settings.allowed_api_key_sha256s
    ):
        raise HTTPException(status_code=401, detail="unauthorized")


def fallback_api_key(request: Request, settings: Settings) -> str:
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    supplied = token.strip() if scheme.lower() == "bearer" else ""
    if supplied:
        fingerprint = hashlib.sha256(supplied.encode("utf-8")).hexdigest()
        # Windmill's shared proxy resource uses an internal gateway bearer,
        # not an OpenAI API key. API-only passthroughs such as audio/speech
        # must use the gateway's server-side OpenAI fallback credential.
        if (
            settings.server_openai_api_key
            and any(hmac.compare_digest(fingerprint, allowed) for allowed in settings.allowed_api_key_sha256s)
        ):
            return settings.server_openai_api_key
        return supplied
    return settings.server_openai_api_key


def forced_failure_reason(
    request: Request, settings: Settings
) -> str | None:
    requested = request.headers.get(
        "x-luna-gateway-force-fallback", ""
    ).strip().lower()
    if not requested:
        return None
    if not settings.enable_test_controls:
        raise HTTPException(
            status_code=403, detail="test_controls_disabled"
        )
    if requested != "quota":
        raise HTTPException(
            status_code=400, detail="invalid_forced_failure"
        )
    return requested


def create_app(
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    selected = settings or Settings.from_env()
    gateway = Gateway(selected, transport=transport)
    app = FastAPI(title="Windmill Luna Gateway", version="1.1.0")
    app.state.settings = selected
    app.state.gateway = gateway

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await gateway.close()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        circuit = await gateway.circuit.snapshot()
        image_circuit = await gateway.image_circuit.snapshot()
        return {
            "status": "ok",
            "gateway_configured": bool(selected.allowed_api_key_sha256s),
            "windmill_caller_allowed": "f777774c7a4100fc25022f34d27483a9080679aed01a0fca54ced407ca09df9f" in selected.allowed_api_key_sha256s,
            "allowed_caller_count": len(selected.allowed_api_key_sha256s),
            "max_body_bytes": selected.max_body_bytes,
            "codex_configured": bool(selected.codex_api_key),
            "api_fallback": (
                "caller_bearer_or_server"
                if selected.server_openai_api_key
                else "caller_bearer"
            ),
            "test_controls": selected.enable_test_controls,
            "circuit": circuit,
            "image_circuit": image_circuit,
            "image_generation": "codex-primary-api-fallback",
        }

    async def handle(
        request: Request, kind: EndpointKind
    ) -> Response:
        require_gateway_auth(request, selected)
        payload = await read_json_body(
            request, selected.max_body_bytes
        )
        request_id = (
            request.headers.get("x-request-id") or str(uuid.uuid4())
        )
        return await gateway.request(
            kind=kind,
            payload=payload,
            request_id=request_id,
            fallback_api_key=selected.server_openai_api_key,
            forced_failure=forced_failure_reason(request, selected),
        )
    @app.post("/v1/chat/completions")
    @app.post("/chat/completions", include_in_schema=False)
    async def chat_completions(request: Request) -> Response:
        return await handle(request, "chat")

    @app.post("/v1/responses")
    @app.post("/responses", include_in_schema=False)
    async def responses(request: Request) -> Response:
        return await handle(request, "responses")

    @app.post("/v1/images/generations")
    @app.post("/images/generations", include_in_schema=False)
    async def image_generations(request: Request) -> Response:
        require_gateway_auth(request, selected)
        payload = await read_json_body(request, selected.max_body_bytes)
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        return await gateway.generate_image(
            payload=payload,
            request_id=request_id,
            api_key=selected.server_openai_api_key,
        )

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    @app.api_route("/{path:path}", methods=["GET", "POST"], include_in_schema=False)
    async def openai_passthrough(path: str, request: Request) -> Response:
        require_gateway_auth(request, selected)
        if path.startswith("v1/"):
            path = path[3:]
        method = request.method.upper()
        allowed_methods = API_PASSTHROUGH_METHODS.get(path)
        if allowed_methods is None or method not in allowed_methods:
            raise HTTPException(status_code=404, detail="endpoint_not_allowed")
        body = await request.body()
        if len(body) > MAX_PASSTHROUGH_BODY_BYTES:
            raise HTTPException(status_code=413, detail="request_too_large")
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        return await gateway.proxy_openai_api(
            path=path,
            method=method,
            body=body,
            request_headers={key.lower(): value for key, value in request.headers.items()},
            request_id=request_id,
            api_key=fallback_api_key(request, selected),
        )

    return app


async def read_json_body(
    request: Request, max_body_bytes: int
) -> dict[str, Any]:
    raw = await request.body()
    if len(raw) > max_body_bytes:
        raise HTTPException(
            status_code=413, detail="request_too_large"
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="invalid_json"
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail="request_body_must_be_object",
        )
    return payload


app = create_app()
