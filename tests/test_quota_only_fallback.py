from __future__ import annotations

import hashlib

import httpx
from fastapi.testclient import TestClient

from app.main import Settings, create_app


CALLER_KEY = "internal-windmill-bearer"
CALLER_HASH = hashlib.sha256(CALLER_KEY.encode()).hexdigest()


def settings(**overrides):
    values = dict(
        allowed_api_key_sha256s=frozenset({CALLER_HASH}),
        codex_url="https://codex.test/v1",
        codex_api_key="internal-codex-sidecar-v1",
        openai_url="https://api.test/v1",
        server_openai_api_key="server-openai-key",
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
    values.update(overrides)
    return Settings(**values)


def client_for(handler, **overrides):
    return TestClient(
        create_app(settings(**overrides), transport=httpx.MockTransport(handler))
    )


def headers(force: str | None = None):
    result = {"Authorization": f"Bearer {CALLER_KEY}"}
    if force:
        result["X-Luna-Gateway-Force-Fallback"] = force
    return result


def payload():
    return {
        "model": "luna-auto",
        "messages": [{"role": "user", "content": "hi"}],
    }


def success(content: str = "ok"):
    return {
        "id": "c1",
        "choices": [
            {"message": {"role": "assistant", "content": content}}
        ],
    }


def test_explicit_quota_uses_server_key_and_quota_circuit():
    seen = []

    def handler(request: httpx.Request):
        seen.append((str(request.url), request.headers.get("authorization")))
        if request.url.host == "codex.test":
            return httpx.Response(
                429,
                json={"error": {"message": "Codex weekly usage limit reached"}},
            )
        return httpx.Response(200, json=success("api-fallback"))

    with client_for(handler) as client:
        first = client.post("/v1/chat/completions", headers=headers(), json=payload())
        second = client.post("/v1/chat/completions", headers=headers(), json=payload())

    assert first.status_code == 200
    assert first.headers["x-luna-gateway-provider"] == "openai-api"
    assert first.headers["x-luna-gateway-fallback"] == "true"
    assert first.headers["x-luna-gateway-fallback-reason"] == "quota"
    assert second.status_code == 200
    assert second.headers["x-luna-gateway-provider"] == "openai-api"
    assert second.headers["x-luna-gateway-fallback-reason"].startswith(
        "circuit_open:quota"
    )
    assert seen == [
        (
            "https://codex.test/v1/chat/completions",
            "Bearer internal-codex-sidecar-v1",
        ),
        ("https://api.test/v1/chat/completions", "Bearer server-openai-key"),
        ("https://api.test/v1/chat/completions", "Bearer server-openai-key"),
    ]
    assert all(value != f"Bearer {CALLER_KEY}" for _, value in seen)


def test_response_bearing_nonquota_failures_never_fallback():
    cases = [
        (502, "Selected model is at capacity"),
        (502, "Bad Gateway"),
        (429, "rate limit reached"),
        (401, "Codex authentication failed"),
    ]

    for status_code, message in cases:
        seen = []

        def handler(request: httpx.Request, status=status_code, detail=message):
            seen.append(str(request.url))
            return httpx.Response(status, json={"error": {"message": detail}})

        with client_for(handler) as client:
            response = client.post(
                "/v1/chat/completions", headers=headers(), json=payload()
            )

        assert response.status_code == status_code
        assert response.headers["x-luna-gateway-provider"] == "codex"
        assert response.headers["x-luna-gateway-fallback"] == "false"
        assert "x-luna-gateway-fallback-reason" not in response.headers
        assert seen == ["https://codex.test/v1/chat/completions"]


def test_timeout_never_falls_back():
    seen = []

    def handler(request: httpx.Request):
        seen.append(str(request.url))
        raise httpx.ReadTimeout("Codex timed out", request=request)

    with client_for(handler) as client:
        response = client.post(
            "/v1/chat/completions", headers=headers(), json=payload()
        )

    assert response.status_code == 504
    assert response.headers["x-luna-gateway-provider"] == "codex"
    assert response.headers["x-luna-gateway-fallback"] == "false"
    assert seen == ["https://codex.test/v1/chat/completions"]


def test_nonquota_forced_fallback_control_is_rejected():
    def handler(request: httpx.Request):
        raise AssertionError("provider should not be called")

    with client_for(handler, enable_test_controls=True) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=headers(force="timeout"),
            json=payload(),
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid_forced_failure"
