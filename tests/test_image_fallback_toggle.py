from __future__ import annotations

import hashlib

import httpx
from fastapi.testclient import TestClient

from app import main as core
from app import runtime_app


CALLER_KEY = "internal-windmill-bearer"
CALLER_HASH = hashlib.sha256(CALLER_KEY.encode()).hexdigest()


def settings(**overrides):
    values = dict(
        allowed_api_key_sha256s=frozenset({CALLER_HASH}),
        codex_url="https://codex.test/v1",
        codex_api_key="internal-codex-sidecar-v1",
        openai_url="https://api.test/v1",
        server_openai_api_key="server-openai-key",
        allowed_models=frozenset({"gpt-6-luna", "luna-auto"}),
        model_aliases={"luna-auto": "gpt-6-luna", "gpt-6-luna": "gpt-6-luna"},
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
    return core.Settings(**values)


def headers():
    return {"Authorization": f"Bearer {CALLER_KEY}"}


def client_for(handler):
    return TestClient(core.create_app(settings(), transport=httpx.MockTransport(handler)))


def image_payload():
    return {"model": "gpt-image-2", "prompt": "restaurant lighting", "size": "1024x1536"}


def test_paid_image_fallback_defaults_off(monkeypatch):
    monkeypatch.delenv("ALLOW_PAID_OPENAI_IMAGE_FALLBACK", raising=False)
    assert runtime_app.paid_image_fallback_enabled() is False


def test_paid_image_fallback_can_be_enabled(monkeypatch):
    monkeypatch.setenv("ALLOW_PAID_OPENAI_IMAGE_FALLBACK", "true")
    assert runtime_app.paid_image_fallback_enabled() is True


def test_image_quota_falls_back_to_paid_api_only_when_enabled(monkeypatch):
    monkeypatch.setenv("ALLOW_PAID_OPENAI_IMAGE_FALLBACK", "true")
    seen = []

    def handler(request: httpx.Request):
        seen.append((str(request.url), request.headers.get("authorization")))
        if request.url.host == "codex.test":
            return httpx.Response(
                429,
                json={"error": {"message": "image_gen usage limit reached", "limit_id": "image_gen"}},
            )
        return httpx.Response(200, json={"data": [{"b64_json": "aW1hZ2U="}]})

    with client_for(handler) as client:
        response = client.post("/v1/images/generations", headers=headers(), json=image_payload())

    assert response.status_code == 200
    assert response.headers["x-luna-gateway-provider"] == "openai-api-image"
    assert response.headers["x-luna-gateway-fallback"] == "true"
    assert response.headers["x-luna-gateway-fallback-reason"] == "image_quota"
    assert seen == [
        ("https://codex.test/v1/images/generations", "Bearer internal-codex-sidecar-v1"),
        ("https://api.test/v1/images/generations", "Bearer server-openai-key"),
    ]


def test_image_quota_stays_fail_closed_when_disabled(monkeypatch):
    monkeypatch.setenv("ALLOW_PAID_OPENAI_IMAGE_FALLBACK", "false")
    seen = []

    def handler(request: httpx.Request):
        seen.append(str(request.url))
        return httpx.Response(
            429,
            json={"error": {"message": "image_gen usage limit reached", "limit_id": "image_gen"}},
        )

    with client_for(handler) as client:
        response = client.post("/v1/images/generations", headers=headers(), json=image_payload())

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "image_quota_unavailable"
    assert response.headers["x-luna-gateway-fallback"] == "false"
    assert seen == ["https://codex.test/v1/images/generations"]


def test_nonquota_image_failure_never_uses_paid_fallback(monkeypatch):
    monkeypatch.setenv("ALLOW_PAID_OPENAI_IMAGE_FALLBACK", "true")
    seen = []

    def handler(request: httpx.Request):
        seen.append(str(request.url))
        return httpx.Response(400, json={"error": {"message": "bad image request"}})

    with client_for(handler) as client:
        response = client.post("/v1/images/generations", headers=headers(), json=image_payload())

    assert response.status_code == 400
    assert response.headers["x-luna-gateway-provider"] == "codex-image"
    assert response.headers["x-luna-gateway-fallback"] == "false"
    assert seen == ["https://codex.test/v1/images/generations"]


def test_image_edits_remain_blocked(monkeypatch):
    monkeypatch.setenv("ALLOW_PAID_OPENAI_IMAGE_FALLBACK", "true")

    def handler(request: httpx.Request):
        raise AssertionError(f"provider must not be called: {request.url}")

    with client_for(handler) as client:
        response = client.post("/v1/images/edits", headers=headers(), content=b"blocked")

    assert response.status_code == 404
