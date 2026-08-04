from __future__ import annotations

import hashlib
import json

import httpx
from fastapi.testclient import TestClient

from app.main import Settings, create_app

CALLER_KEY = "api-secret"
CALLER_HASH = hashlib.sha256(CALLER_KEY.encode()).hexdigest()


def settings(**overrides):
    values = dict(
        allowed_api_key_sha256s=frozenset({CALLER_HASH}),
        codex_url="https://codex.test/v1",
        codex_api_key="internal-codex-sidecar-v1",
        openai_url="https://api.test/v1",
        server_openai_api_key="",
        allowed_models=frozenset({"gpt-5.6-luna", "luna-auto"}),
        model_aliases={"luna-auto": "gpt-5.6-luna", "gpt-5.6-luna": "gpt-5.6-luna"},
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
    values.update(overrides)
    return Settings(**values)


def client_for(handler, **overrides):
    return TestClient(create_app(settings(**overrides), transport=httpx.MockTransport(handler)))


def headers(api_key=CALLER_KEY, force=None):
    result = {"Authorization": f"Bearer {api_key}"}
    if force:
        result["X-Luna-Gateway-Force-Fallback"] = force
    return result


def payload():
    return {"model": "luna-auto", "messages": [{"role": "user", "content": "hi"}]}


def success(content="hello"):
    return {"id": "c1", "choices": [{"message": {"role": "assistant", "content": content}}]}


def test_codex_primary_uses_internal_sidecar_key():
    seen = []
    def handler(request):
        seen.append((str(request.url), request.headers.get("authorization")))
        return httpx.Response(200, json=success())
    with client_for(handler) as client:
        response = client.post("/v1/chat/completions", headers=headers(), json=payload())
    assert response.status_code == 200
    assert response.headers["x-luna-gateway-provider"] == "codex"
    assert response.headers["x-luna-gateway-fallback"] == "false"
    assert seen == [("https://codex.test/v1/chat/completions", "Bearer internal-codex-sidecar-v1")]


def test_missing_or_unknown_bearer_is_rejected():
    def handler(request):
        raise AssertionError("provider called")
    with client_for(handler) as client:
        assert client.post("/v1/chat/completions", json=payload()).status_code == 401
        assert client.post("/v1/chat/completions", headers=headers("wrong"), json=payload()).status_code == 401


def test_quota_fallback_and_circuit_use_caller_key():
    seen = []
    def handler(request):
        seen.append((str(request.url), request.headers.get("authorization")))
        if request.url.host == "codex.test":
            return httpx.Response(429, json={"error": {"message": "Codex usage limit reached"}})
        return httpx.Response(200, json=success("fallback"))
    with client_for(handler) as client:
        first = client.post("/v1/chat/completions", headers=headers(), json=payload())
        second = client.post("/v1/chat/completions", headers=headers(), json=payload())
    assert first.headers["x-luna-gateway-provider"] == "openai-api"
    assert first.headers["x-luna-gateway-fallback-reason"] == "quota"
    assert second.headers["x-luna-gateway-fallback-reason"].startswith("circuit_open:quota")
    assert seen == [
        ("https://codex.test/v1/chat/completions", "Bearer internal-codex-sidecar-v1"),
        ("https://api.test/v1/chat/completions", f"Bearer {CALLER_KEY}"),
        ("https://api.test/v1/chat/completions", f"Bearer {CALLER_KEY}"),
    ]


def test_forced_fallback_requires_test_controls():
    def handler(request):
        return httpx.Response(200, json=success("fallback"))
    with client_for(handler) as client:
        disabled = client.post("/v1/chat/completions", headers=headers(force="quota"), json=payload())
    assert disabled.status_code == 403
    with client_for(handler, enable_test_controls=True) as client:
        enabled = client.post("/v1/chat/completions", headers=headers(force="quota"), json=payload())
    assert enabled.status_code == 200
    assert enabled.headers["x-luna-gateway-provider"] == "openai-api"
    assert enabled.headers["x-luna-gateway-fallback-reason"] == "forced_test:quota"


def test_non_retryable_400_does_not_fallback():
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(400, json={"error": {"message": "bad"}})
    with client_for(handler) as client:
        response = client.post("/v1/chat/completions", headers=headers(), json=payload())
    assert response.status_code == 400
    assert response.headers["x-luna-gateway-provider"] == "codex"
    assert calls == ["https://codex.test/v1/chat/completions"]


def test_invalid_json_success_falls_back():
    def handler(request):
        if request.url.host == "codex.test":
            return httpx.Response(200, json=success("not json"))
        return httpx.Response(200, json=success(json.dumps({"ok": True})))
    request_payload = payload()
    request_payload["response_format"] = {"type": "json_object"}
    with client_for(handler) as client:
        response = client.post("/v1/chat/completions", headers=headers(), json=request_payload)
    assert response.headers["x-luna-gateway-provider"] == "openai-api"
    assert response.headers["x-luna-gateway-fallback-reason"] == "invalid_success"


def test_responses_endpoint_preserves_web_search_tool():
    seen = {}
    def handler(request):
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "r1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
            },
        )
    request_payload = {"model": "luna-auto", "input": "research", "tools": [{"type": "web_search"}], "tool_choice": "required"}
    with client_for(handler) as client:
        response = client.post("/v1/responses", headers=headers(), json=request_payload)
    assert response.status_code == 200
    assert seen["url"] == "https://codex.test/v1/responses"
    assert seen["payload"]["model"] == "gpt-5.6-luna"
    assert seen["payload"]["tools"] == [{"type": "web_search"}]


def test_api_only_model_and_media_passthrough():
    seen = []
    def handler(request):
        seen.append((str(request.url), request.headers.get("authorization"), request.content))
        if request.url.path.endswith("/audio/speech"):
            return httpx.Response(200, content=b"audio-bytes", headers={"content-type": "audio/mpeg"})
        return httpx.Response(200, json=success("api-only"))
    with client_for(handler) as client:
        model_response = client.post("/v1/chat/completions", headers=headers(), json={"model": "gpt-5-mini", "messages": []})
        speech_response = client.post("/v1/audio/speech", headers={**headers(), "Content-Type": "application/json"}, content=b'{"model":"gpt-4o-mini-tts","input":"hi"}')
        blocked = client.post("/v1/files", headers=headers(), content=b"x")
    assert model_response.status_code == 200
    assert model_response.headers["x-luna-gateway-provider"] == "openai-api"
    assert model_response.headers["x-luna-gateway-fallback"] == "false"
    assert speech_response.status_code == 200
    assert speech_response.content == b"audio-bytes"
    assert speech_response.headers["content-type"].startswith("audio/mpeg")
    assert blocked.status_code == 404
    assert seen[0][0] == "https://api.test/v1/chat/completions"
    assert seen[1][0] == "https://api.test/v1/audio/speech"
    assert all(item[1] == f"Bearer {CALLER_KEY}" for item in seen)


def test_streaming_size_guards_and_health():
    def handler(request):
        raise AssertionError("provider called")
    with client_for(handler, max_body_bytes=20) as client:
        assert client.post("/v1/chat/completions", headers=headers(), json=payload()).status_code == 413
    with client_for(handler, enable_test_controls=True) as client:
        assert client.post("/v1/chat/completions", headers=headers(), json={"model": "gpt-5.6-luna", "messages": [], "stream": True}).status_code == 400
        health = client.get("/health").json()
    assert health["gateway_configured"] is True
    assert health["codex_configured"] is True
    assert health["api_fallback"] == "caller_bearer"
    assert health["test_controls"] is True
    assert health["test_controls"] is True


def test_adaptive_reasoning_levels_and_explicit_override():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        if request.url.path.endswith("/responses"):
            return httpx.Response(
                200,
                json={
                    "id": "r1",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "done"}
                            ],
                        }
                    ],
                },
            )
        return httpx.Response(200, json=success())

    with client_for(handler) as client:
        simple = client.post(
            "/v1/chat/completions", headers=headers(), json=payload()
        )
        research = client.post(
            "/v1/responses",
            headers=headers(),
            json={
                "model": "luna-auto",
                "input": "Research this website and synthesize the findings.",
                "tools": [{"type": "web_search"}],
            },
        )
        strategic = client.post(
            "/v1/chat/completions",
            headers=headers(),
            json={
                "model": "luna-auto",
                "messages": [
                    {
                        "role": "user",
                        "content": "Run the conversation learning optimizer.",
                    }
                ],
            },
        )
        explicit = client.post(
            "/v1/chat/completions",
            headers=headers(),
            json={
                "model": "luna-auto",
                "reasoning_effort": "medium",
                "messages": [{"role": "user", "content": "classify"}],
            },
        )

    assert all(
        response.status_code == 200
        for response in (simple, research, strategic, explicit)
    )
    assert seen[0]["reasoning_effort"] == "low"
    assert seen[1]["reasoning"] == {"effort": "medium"}
    assert seen[2]["reasoning_effort"] == "high"
    assert seen[3]["reasoning_effort"] == "medium"


def test_quota_fallback_preserves_reasoning_effort():
    seen = []

    def handler(request):
        seen.append((request.url.host, json.loads(request.content)))
        if request.url.host == "codex.test":
            return httpx.Response(
                429,
                json={"error": {"message": "Codex usage limit reached"}},
            )
        return httpx.Response(200, json=success("fallback"))

    with client_for(handler) as client:
        response = client.post(
            "/v1/chat/completions", headers=headers(), json=payload()
        )

    assert response.status_code == 200
    assert response.headers["x-luna-gateway-provider"] == "openai-api"
    assert seen[0][1]["reasoning_effort"] == "low"
    assert seen[1][1]["reasoning_effort"] == "low"


def test_invalid_explicit_reasoning_is_rejected():
    def handler(request):
        raise AssertionError("provider called")

    with client_for(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=headers(),
            json={
                "model": "luna-auto",
                "reasoning_effort": "extreme",
                "messages": [],
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid_reasoning_effort"