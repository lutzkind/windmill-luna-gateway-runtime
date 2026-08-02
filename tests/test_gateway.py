from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from app.main import Settings, create_app


def settings(**overrides):
    base = dict(
        gateway_token="gateway-secret",
        codex_url="https://codex.test/v1",
        codex_api_key="codex-secret",
        openai_url="https://api.test/v1",
        server_openai_api_key="",
        allowed_models=frozenset({"gpt-5.6-luna", "luna-auto"}),
        model_aliases={
            "luna-auto": "gpt-5.6-luna",
            "gpt-5.6-luna": "gpt-5.6-luna",
        },
        timeout_seconds=10,
        max_body_bytes=1024 * 1024,
        max_concurrency=2,
        transient_failure_threshold=2,
        transient_failure_window_seconds=300,
        transient_open_seconds=900,
        quota_open_seconds=1800,
        auth_open_seconds=300,
        enable_test_controls=False,
    )
    base.update(overrides)
    return Settings(**base)


def client_for(handler, **setting_overrides):
    transport = httpx.MockTransport(handler)
    return TestClient(
        create_app(settings(**setting_overrides), transport=transport)
    )


def headers(*, api_key: str = "api-secret", force: str | None = None):
    result = {
        "X-Luna-Gateway-Token": "gateway-secret",
        "Authorization": f"Bearer {api_key}",
    }
    if force:
        result["X-Luna-Gateway-Force-Fallback"] = force
    return result


def chat_payload():
    return {
        "model": "luna-auto",
        "messages": [{"role": "user", "content": "hi"}],
    }


def chat_success(content: str = "hello"):
    return {
        "id": "chatcmpl_1",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                }
            }
        ],
    }


def test_primary_codex_success_uses_sidecar_key():
    seen = []

    def handler(request: httpx.Request):
        seen.append(
            (
                str(request.url),
                request.headers.get("authorization"),
            )
        )
        return httpx.Response(200, json=chat_success())

    with client_for(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=headers(),
            json=chat_payload(),
        )

    assert response.status_code == 200
    assert response.headers["x-luna-gateway-provider"] == "codex"
    assert response.headers["x-luna-gateway-fallback"] == "false"
    assert seen == [
        (
            "https://codex.test/v1/chat/completions",
            "Bearer codex-secret",
        )
    ]


def test_quota_429_falls_back_with_caller_api_key_and_opens_circuit():
    seen = []

    def handler(request: httpx.Request):
        seen.append(
            (
                str(request.url),
                request.headers.get("authorization"),
            )
        )
        if request.url.host == "codex.test":
            return httpx.Response(
                429,
                json={
                    "error": {
                        "message": "Codex usage limit reached"
                    }
                },
            )
        return httpx.Response(200, json=chat_success("fallback"))

    with client_for(handler) as client:
        first = client.post(
            "/v1/chat/completions",
            headers=headers(api_key="caller-api-key"),
            json=chat_payload(),
        )
        second = client.post(
            "/v1/chat/completions",
            headers=headers(api_key="caller-api-key"),
            json=chat_payload(),
        )

    assert first.status_code == 200
    assert first.headers["x-luna-gateway-provider"] == "openai-api"
    assert first.headers["x-luna-gateway-fallback-reason"] == "quota"
    assert second.headers["x-luna-gateway-provider"] == "openai-api"
    assert second.headers[
        "x-luna-gateway-fallback-reason"
    ].startswith("circuit_open:quota")
    assert seen == [
        (
            "https://codex.test/v1/chat/completions",
            "Bearer codex-secret",
        ),
        (
            "https://api.test/v1/chat/completions",
            "Bearer caller-api-key",
        ),
        (
            "https://api.test/v1/chat/completions",
            "Bearer caller-api-key",
        ),
    ]


def test_forced_live_fallback_control_requires_enablement():
    def handler(request: httpx.Request):
        return httpx.Response(200, json=chat_success("fallback"))

    with client_for(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=headers(force="quota"),
            json=chat_payload(),
        )
    assert response.status_code == 403

    with client_for(handler, enable_test_controls=True) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=headers(force="quota"),
            json=chat_payload(),
        )
    assert response.status_code == 200
    assert response.headers["x-luna-gateway-provider"] == "openai-api"
    assert (
        response.headers["x-luna-gateway-fallback-reason"]
        == "forced_test:quota"
    )


def test_missing_fallback_key_fails_closed():
    def handler(request: httpx.Request):
        return httpx.Response(
            429,
            json={"error": {"message": "Codex quota reached"}},
        )

    with client_for(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Luna-Gateway-Token": "gateway-secret"},
            json=chat_payload(),
        )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "fallback_key_missing"


def test_server_fallback_key_is_used_when_caller_bearer_is_absent():
    seen = []

    def handler(request: httpx.Request):
        seen.append(request.headers.get("authorization"))
        if request.url.host == "codex.test":
            return httpx.Response(
                429,
                json={"error": {"message": "Codex quota reached"}},
            )
        return httpx.Response(200, json=chat_success("fallback"))

    with client_for(
        handler, server_openai_api_key="server-api-key"
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Luna-Gateway-Token": "gateway-secret"},
            json=chat_payload(),
        )

    assert response.status_code == 200
    assert seen == [
        "Bearer codex-secret",
        "Bearer server-api-key",
    ]


def test_non_retryable_400_does_not_fallback():
    calls = []

    def handler(request: httpx.Request):
        calls.append(str(request.url))
        return httpx.Response(
            400, json={"error": {"message": "bad request"}}
        )

    with client_for(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=headers(),
            json=chat_payload(),
        )

    assert response.status_code == 400
    assert response.headers["x-luna-gateway-provider"] == "codex"
    assert calls == ["https://codex.test/v1/chat/completions"]


def test_invalid_json_mode_success_falls_back():
    def handler(request: httpx.Request):
        if request.url.host == "codex.test":
            return httpx.Response(
                200,
                json=chat_success("not json"),
            )
        return httpx.Response(
            200,
            json=chat_success(json.dumps({"ok": True})),
        )

    payload = chat_payload()
    payload["response_format"] = {"type": "json_object"}

    with client_for(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=headers(),
            json=payload,
        )

    assert response.status_code == 200
    assert response.headers["x-luna-gateway-provider"] == "openai-api"
    assert (
        response.headers["x-luna-gateway-fallback-reason"]
        == "invalid_success"
    )


def test_json_schema_failure_falls_back():
    schema = {
        "type": "object",
        "properties": {"classification": {"type": "string"}},
        "required": ["classification"],
        "additionalProperties": False,
    }

    def handler(request: httpx.Request):
        if request.url.host == "codex.test":
            content = json.dumps({"wrong": "field"})
        else:
            content = json.dumps(
                {"classification": "interested"}
            )
        return httpx.Response(200, json=chat_success(content))

    payload = chat_payload()
    payload["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": "classification",
            "schema": schema,
        },
    }

    with client_for(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=headers(),
            json=payload,
        )

    assert response.status_code == 200
    assert response.headers["x-luna-gateway-provider"] == "openai-api"


def test_responses_endpoint_preserves_payload():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "done",
                            }
                        ],
                    }
                ],
            },
        )

    payload = {
        "model": "luna-auto",
        "input": "research this",
        "tools": [{"type": "web_search"}],
        "tool_choice": "required",
    }
    with client_for(handler) as client:
        response = client.post(
            "/v1/responses",
            headers=headers(),
            json=payload,
        )

    assert response.status_code == 200
    assert seen["url"] == "https://codex.test/v1/responses"
    assert seen["payload"]["model"] == "gpt-5.6-luna"
    assert seen["payload"]["tools"] == [{"type": "web_search"}]


def test_auth_model_streaming_and_size_guards():
    def handler(request: httpx.Request):
        raise AssertionError("provider should not be called")

    with client_for(handler, max_body_bytes=20) as client:
        assert (
            client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer api-secret"},
                json=chat_payload(),
            ).status_code
            == 401
        )

    with client_for(handler) as client:
        invalid_model = client.post(
            "/v1/chat/completions",
            headers=headers(),
            json={"model": "other", "messages": []},
        )
        streaming = client.post(
            "/v1/chat/completions",
            headers=headers(),
            json={
                "model": "gpt-5.6-luna",
                "messages": [],
                "stream": True,
            },
        )

    assert invalid_model.status_code == 400
    assert streaming.status_code == 400
