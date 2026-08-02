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

LOGGER = logging.getLogger("windmill_luna_gateway")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

EndpointKind = Literal["chat", "responses"]
ProviderName = Literal["codex", "openai"]


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
        allowed_key_hashes = frozenset(
            part.strip().lower()
            for part in os.getenv("ALLOWED_API_KEY_SHA256S", "").split(",")
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
                os.getenv("MAX_BODY_BYTES", str(2 * 1024 * 1024))
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


@dataclass(frozen=True)
class ProviderResult:
    response: httpx.Response | None
    error_reason: str | None = None
    error_detail: str | None = None


class Gateway:
    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.settings = settings
        self.circuit = CircuitBreaker(settings)
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
        if model not in self.settings.allowed_models:
            raise HTTPException(status_code=400, detail="model_not_allowed")

        normalized = dict(payload)
        normalized["model"] = self.settings.model_aliases.get(model, model)
        if normalized.get("stream") is True:
            raise HTTPException(
                status_code=400, detail="streaming_not_supported"
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


def classify_failure(result: ProviderResult) -> str | None:
    if result.error_reason:
        return result.error_reason
    if result.response is None:
        return "network"

    status = result.response.status_code
    text = result.response.text[:4000].lower()
    if status in {401, 403}:
        return "auth"
    if status == 429:
        quota_terms = (
            "usage limit",
            "quota",
            "plan limit",
            "limit reached",
            "insufficient_quota",
            "codex usage",
            "weekly limit",
        )
        return (
            "quota"
            if any(term in text for term in quota_terms)
            else "rate_limit"
        )
    if status in {408, 500, 502, 503, 504}:
        return "timeout" if status == 408 else "upstream_5xx"
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
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
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
    if requested not in {
        "quota",
        "auth",
        "timeout",
        "network",
        "upstream_5xx",
        "rate_limit",
        "invalid_success",
    }:
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
        return {
            "status": "ok",
            "gateway_configured": bool(selected.allowed_api_key_sha256s),
            "codex_configured": bool(selected.codex_api_key),
            "api_fallback": (
                "caller_bearer_or_server"
                if selected.server_openai_api_key
                else "caller_bearer"
            ),
            "test_controls": selected.enable_test_controls,
            "circuit": circuit,
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
            fallback_api_key=fallback_api_key(request, selected),
            forced_failure=forced_failure_reason(request, selected),
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await handle(request, "chat")

    @app.post("/v1/responses")
    async def responses(request: Request) -> Response:
        return await handle(request, "responses")

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
