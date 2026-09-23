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


def test_quota_failure_does_not_send_caller_key_to_direct_api():
    seen = []
    def handler(request):
        seen.append((str(request.url), request.headers.get("authorization")))
        return httpx.Response(429, json={"error": {"message": "Codex usage limit reached"}})
    with client_for(handler) as client:
        first = client.post("/v1/chat/completions", headers=headers(), json=payload())
        second = client.post("/v1/chat/completions", headers=headers(), json=payload())
    assert first.status_code == 502
    assert first.headers["x-luna-gateway-provider"] == "none"
    assert first.headers["x-luna-gateway-fallback-reason"] == "quota"
    assert second.status_code == 502
    assert second.headers["x-luna-gateway-fallback-reason"].startswith("circuit_open:quota")
    assert seen == [("https://codex.test/v1/chat/completions", "Bearer internal-codex-sidecar-v1")]


def test_forced_fallback_requires_test_controls():
    def handler(request):
        return httpx.Response(200, json=success("fallback"))
    with client_for(handler) as client:
        disabled = client.post("/v1/chat/completions", headers=headers(force="quota"), json=payload())
    assert disabled.status_code == 403
    with client_for(handler, enable_test_controls=True) as client:
        enabled = client.post("/v1/chat/completions", headers=headers(force="quota"), json=payload())
        assert enabled.status_code == 502
    assert enabled.headers["x-luna-gateway-provider"] == "none"
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


def test_invalid_json_success_does_not_fallback():
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, json=success("not json"))
    request_payload = payload()
    request_payload["response_format"] = {"type": "json_object"}
    with client_for(handler) as client:
        response = client.post("/v1/chat/completions", headers=headers(), json=request_payload)
    assert response.status_code == 502
    assert response.headers["x-luna-gateway-provider"] == "codex"
    assert response.headers["x-luna-gateway-fallback"] == "false"
    assert "x-luna-gateway-fallback-reason" not in response.headers
    assert calls == ["https://codex.test/v1/chat/completions"]


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
    assert seen["payload"]["model"] == "gpt-6-luna"
    assert seen["payload"]["tools"] == [{"type": "web_search"}]


def test_api_only_model_and_media_passthrough():
    seen = []
    def handler(request):
        seen.append((str(request.url), request.headers.get("authorization"), request.content))
        if request.url.path.endswith("/audio/speech"):
            return httpx.Response(200, content=b"audio-bytes", headers={"content-type": "audio/mpeg"})
        return httpx.Response(200, json=success("api-only"))
    with client_for(handler, allowed_models=frozenset({"gpt-6-luna", "luna-auto", "gpt-5-mini"})) as client:
        model_response = client.post("/v1/chat/completions", headers=headers(), json={"model": "gpt-5-mini", "messages": []})
        speech_response = client.post("/v1/audio/speech", headers={**headers(), "Content-Type": "application/json"}, content=b'{"model":"gpt-4o-mini-tts","input":"hi"}')
        blocked = client.post("/v1/files", headers=headers(), content=b"x")
    assert model_response.status_code == 502
    assert model_response.headers["x-luna-gateway-provider"] == "none"
    assert model_response.headers["x-luna-gateway-fallback"] == "true"
    assert speech_response.status_code == 200
    assert speech_response.content == b"audio-bytes"
    assert speech_response.headers["content-type"].startswith("audio/mpeg")
    assert blocked.status_code == 404
    assert seen[0][0] == "https://api.test/v1/audio/speech"
    assert all(item[1] == f"Bearer {CALLER_KEY}" for item in seen)


def test_internal_windmill_bearer_uses_server_key_for_api_only_media():
    seen = []

    def handler(request):
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, content=b"audio-bytes", headers={"content-type": "audio/mpeg"})

    with client_for(handler, server_openai_api_key="server-api-secret") as client:
        response = client.post(
            "/v1/audio/speech",
            headers={**headers(), "Content-Type": "application/json"},
            content=b'{"model":"gpt-4o-mini-tts","input":"hi"}',
        )

    assert response.status_code == 200
    assert seen == ["Bearer server-api-secret"]


def test_streaming_size_guards_and_health():
    def handler(request):
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True})
        raise AssertionError("provider called")
    with client_for(handler, max_body_bytes=20) as client:
        assert client.post("/v1/chat/completions", headers=headers(), json=payload()).status_code == 413
    with client_for(handler, enable_test_controls=True) as client:
        assert client.post("/v1/chat/completions", headers=headers(), json={"model": "gpt-6-luna", "messages": [], "stream": True}).status_code == 400
        health_response = client.get("/health")
        health = health_response.json()
    assert health_response.status_code == 200
    assert health["gateway_configured"] is True
    assert health["codex_configured"] is True
    assert health["codex_upstream"] == {"ok": True, "status_code": 200}
    assert health["api_fallback"] == "caller_bearer"
    assert health["test_controls"] is True


def test_health_fails_when_codex_upstream_is_unhealthy():
    def handler(request):
        if request.url.path == "/healthz":
            return httpx.Response(503, json={"ok": False})
        raise AssertionError("provider called")

    with client_for(handler) as client:
        response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["codex_upstream"] == {"ok": False, "status_code": 503}



def test_default_model_body_limit_supports_multimodal_requests(monkeypatch):
    monkeypatch.delenv("MAX_BODY_BYTES", raising=False)
    settings = Settings.from_env()
    assert settings.max_body_bytes == 16 * 1024 * 1024


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
        return httpx.Response(429, json={"error": {"message": "Codex usage limit reached"}})

    with client_for(handler) as client:
        response = client.post(
            "/v1/chat/completions", headers=headers(), json=payload()
        )

    assert response.status_code == 502
    assert response.headers["x-luna-gateway-provider"] == "none"
    assert seen[0][1]["reasoning_effort"] == "low"


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


def test_payload_only_context_guards_and_no_implicit_tools():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=success())

    with client_for(handler) as client:
        clean = client.post(
            "/v1/chat/completions", headers=headers(), json=payload()
        )
        stored = client.post(
            "/v1/chat/completions",
            headers=headers(),
            json={**payload(), "store": True},
        )
        chained = client.post(
            "/v1/responses",
            headers=headers(),
            json={
                "model": "luna-auto",
                "input": "continue",
                "previous_response_id": "resp_previous",
            },
        )

    assert clean.status_code == 200
    assert len(seen) == 1
    assert "tools" not in seen[0]
    assert "previous_response_id" not in seen[0]
    assert stored.status_code == 400
    assert stored.json()["detail"] == "persistent_storage_not_supported"
    assert chained.status_code == 400
    assert chained.json()["detail"] == "previous_response_id_not_supported"


def test_codex_image_primary_is_not_api_passthrough():
    seen = []
    def handler(request):
        seen.append(str(request.url))
        if request.url.host == "codex.test" and request.url.path.endswith("/images/generations"):
            return httpx.Response(200, json={"data": [{"b64_json": "aW1hZ2U="}], "size": "1024x1536"})
        raise AssertionError(f"unexpected provider call: {request.url}")
    with client_for(handler, server_openai_api_key="server-api-secret") as client:
        response = client.post("/v1/images/generations", headers=headers(), json={"model": "gpt-image-2", "prompt": "restaurant lighting", "size": "1024x1536"})
    assert response.status_code == 200
    assert response.headers["x-luna-gateway-provider"] == "codex-image"
    assert response.headers["x-luna-gateway-fallback"] == "false"
    assert seen == ["https://codex.test/v1/images/generations"]


def test_image_quota_falls_back_without_opening_text_circuit():
    seen = []
    def handler(request):
        seen.append(str(request.url))
        if request.url.host == "codex.test" and request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True})
        if request.url.host == "codex.test" and request.url.path.endswith("/images/generations"):
            return httpx.Response(429, json={"error": {"message": "image_gen usage limit reached", "limit_id": "image_gen"}})
        if request.url.host == "api.test" and request.url.path.endswith("/images/generations"):
            return httpx.Response(200, json={"data": [{"b64_json": "ZmFsbGJhY2s="}], "size": "1024x1536"})
        if request.url.host == "codex.test" and request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json=success("text-still-codex"))
        raise AssertionError(f"unexpected provider call: {request.url}")
    with client_for(handler, server_openai_api_key="server-api-secret") as client:
        image = client.post("/v1/images/generations", headers=headers(), json={"model": "gpt-image-2", "prompt": "restaurant lighting", "size": "1024x1536"})
        text = client.post("/v1/chat/completions", headers=headers(), json=payload())
        health = client.get("/health").json()
    assert image.status_code == 200
    assert image.headers["x-luna-gateway-provider"] == "openai-api"
    assert image.headers["x-luna-gateway-fallback"] == "true"
    assert image.headers["x-luna-gateway-fallback-reason"] == "image_quota"
    assert text.status_code == 200
    assert text.headers["x-luna-gateway-provider"] == "codex"
    assert health["image_circuit"]["open"] is True
    assert health["circuit"]["open"] is False


def test_model_allowlist_rejects_unlisted_models_before_provider_calls():
    def handler(request):
        raise AssertionError("provider called")

    with client_for(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=headers(),
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        responses_response = client.post(
            "/v1/responses",
            headers=headers(),
            json={"model": "gpt-4o", "input": "hi"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "model_not_allowed"
    assert responses_response.status_code == 400
    assert responses_response.json()["detail"] == "model_not_allowed"


def test_model_allowlist_accepts_alias_targets_and_canonical_names():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json=success())

    with client_for(handler, allowed_models=frozenset({"gpt-6-luna"})) as client:
        aliased = client.post("/v1/chat/completions", headers=headers(), json=payload())
        canonical = client.post(
            "/v1/chat/completions",
            headers=headers(),
            json={"model": "gpt-6-luna", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert aliased.status_code == 200
    assert canonical.status_code == 200
    assert seen == [
        "https://codex.test/v1/chat/completions",
        "https://codex.test/v1/chat/completions",
    ]


def test_model_allowlist_fails_closed_for_unlisted_alias_targets():
    def handler(request):
        raise AssertionError("provider called")

    with client_for(
        handler,
        allowed_models=frozenset({"gpt-6-luna"}),
        model_aliases={"luna-auto": "gpt-4o"},
    ) as client:
        response = client.post("/v1/chat/completions", headers=headers(), json=payload())

    assert response.status_code == 400
    assert response.json()["detail"] == "model_not_allowed"


def test_empty_model_allowlist_disables_enforcement():
    seen = []

    def handler(request):
        if request.url.host == "codex.test" and request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True})
        seen.append(str(request.url))
        return httpx.Response(200, json=success("api-only"))

    with client_for(
        handler,
        allowed_models=frozenset(),
        server_openai_api_key="server-api-secret",
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=headers(),
            json={"model": "gpt-5-mini", "messages": [{"role": "user", "content": "hi"}]},
        )
        health = client.get("/health").json()

    assert response.status_code == 200
    assert response.headers["x-luna-gateway-provider"] == "openai-api"
    assert seen == ["https://api.test/v1/chat/completions"]
    assert health["model_allowlist_enforced"] is False


def test_model_allowlist_defaults_and_health_flag(monkeypatch):
    monkeypatch.delenv("ALLOWED_MODELS", raising=False)
    assert Settings.from_env().allowed_models == frozenset({"gpt-6-luna", "luna-auto"})
    monkeypatch.setenv("ALLOWED_MODELS", "")
    assert Settings.from_env().allowed_models == frozenset()

    def handler(request):
        if request.url.host == "codex.test" and request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True})
        raise AssertionError("provider called")

    with client_for(handler) as client:
        health = client.get("/health").json()
    assert health["model_allowlist_enforced"] is True


def test_codex_image_client_error_does_not_use_paid_fallback():
    seen = []
    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(400, json={"error": {"message": "bad image request"}})
    with client_for(handler, server_openai_api_key="server-api-secret") as client:
        response = client.post("/v1/images/generations", headers=headers(), json={"model": "gpt-image-2", "prompt": "restaurant lighting"})
    assert response.status_code == 400
    assert response.headers["x-luna-gateway-provider"] == "codex-image"
    assert response.headers["x-luna-gateway-fallback"] == "false"
    assert seen == ["https://codex.test/v1/images/generations"]


def test_health_reports_canonical_luna_auto_mapping():
    def handler(request):
        if request.url.host == "codex.test" and request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True})
        raise AssertionError("provider called")

    with client_for(handler) as client:
        health = client.get("/health").json()

    assert health["luna_auto_model"] == "gpt-6-luna"
    assert health["model_aliases"] == {"gpt-6-luna": "gpt-6-luna", "luna-auto": "gpt-6-luna"}
    assert health["reasoning_efforts"] == ["high", "low", "medium", "none"]
    assert health["model_validation_enabled"] is False


def test_luna_auto_and_concrete_model_produce_identical_reasoning_effort():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=success())

    messages = [{"role": "user", "content": "hi"}]
    with client_for(handler) as client:
        assert client.post(
            "/v1/chat/completions",
            headers=headers(),
            json={"model": "luna-auto", "messages": messages},
        ).status_code == 200
        assert client.post(
            "/v1/chat/completions",
            headers=headers(),
            json={"model": "gpt-6-luna", "messages": messages},
        ).status_code == 200

    assert len(seen) == 2
    assert seen[0]["model"] == seen[1]["model"] == "gpt-6-luna"
    assert seen[0]["reasoning_effort"] == seen[1]["reasoning_effort"]


def test_explicit_reasoning_effort_is_forwarded_unchanged_for_every_model_alias():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=success())

    with client_for(handler) as client:
        for model in ("luna-auto", "gpt-6-luna"):
            response = client.post(
                "/v1/chat/completions",
                headers=headers(),
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "reasoning_effort": "high",
                },
            )
            assert response.status_code == 200

    assert [entry["reasoning_effort"] for entry in seen] == ["high", "high"]
    assert [entry["model"] for entry in seen] == ["gpt-6-luna", "gpt-6-luna"]


def test_validate_model_endpoint_is_hidden_until_enabled():
    def handler(request):
        raise AssertionError("provider called")

    with client_for(handler) as client:
        response = client.post(
            "/admin/validate-model",
            headers=headers(),
            json={"model": "gpt-6-luna"},
        )
    assert response.status_code == 404


def test_validate_model_endpoint_requires_allowlisted_caller():
    def handler(request):
        raise AssertionError("provider called")

    with client_for(handler, enable_model_validation=True) as client:
        unauthorized = client.post("/admin/validate-model", json={"model": "gpt-6-luna"})
        wrong = client.post(
            "/admin/validate-model",
            headers=headers("wrong"),
            json={"model": "gpt-6-luna"},
        )
    assert unauthorized.status_code == 401
    assert wrong.status_code == 401


def test_validate_model_checks_reasoning_levels_and_smoke_without_touching_aliases():
    seen = []

    def handler(request):
        if request.url.host == "codex.test" and request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True})
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(
            200,
            json=success("LUNA_SMOKE_OK"),
        )

    with client_for(handler, enable_model_validation=True) as client:
        response = client.post(
            "/admin/validate-model",
            headers=headers(),
            json={
                "model": "gpt-7-luna",
                "reasoning_efforts": ["low", "high"],
                "smoke": True,
            },
        )
        health = client.get("/health").json()

    assert response.status_code == 200
    result = response.json()
    assert result["ok"] is True
    assert result["model"] == "gpt-7-luna"
    assert result["checks"]["reasoning_efforts"]["low"]["ok"] is True
    assert result["checks"]["reasoning_efforts"]["high"]["ok"] is True
    assert result["checks"]["smoke"]["ok"] is True
    assert [entry.get("reasoning_effort") for entry in seen] == ["low", "high", None]
    assert {entry["model"] for entry in seen} == {"gpt-7-luna"}
    # Validation must never advance the active routing mapping.
    assert health["luna_auto_model"] == "gpt-6-luna"


def test_validate_model_reports_failure_and_keeps_mapping():
    def handler(request):
        if request.url.host == "codex.test" and request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(400, json={"error": {"message": "model is not supported"}})

    with client_for(handler, enable_model_validation=True) as client:
        response = client.post(
            "/admin/validate-model",
            headers=headers(),
            json={"model": "gpt-9-luna", "reasoning_efforts": ["low"], "smoke": True},
        )
        health = client.get("/health").json()

    assert response.status_code == 200
    result = response.json()
    assert result["ok"] is False
    assert "reasoning_effort_failed:low" in result["errors"]
    assert "smoke_test_failed" in result["errors"]
    assert health["luna_auto_model"] == "gpt-6-luna"


def test_validate_model_rejects_invalid_candidates():
    def handler(request):
        if request.url.host == "codex.test" and request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True})
        raise AssertionError("provider called")

    with client_for(handler, enable_model_validation=True) as client:
        bad_model = client.post("/admin/validate-model", headers=headers(), json={"model": "../etc/passwd"})
        bad_effort = client.post(
            "/admin/validate-model",
            headers=headers(),
            json={"model": "gpt-6-luna", "reasoning_efforts": ["extreme"]},
        )
    assert bad_model.status_code == 400
    assert bad_effort.status_code == 400


def test_validate_model_accepts_codex_cli_reasoning_levels():
    seen = []

    def handler(request):
        if request.url.host == "codex.test" and request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True})
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=success("LUNA_SMOKE_OK"))

    with client_for(handler, enable_model_validation=True) as client:
        response = client.post(
            "/admin/validate-model",
            headers=headers(),
            json={"model": "gpt-7-luna", "reasoning_efforts": ["default", "xhigh", "max"], "smoke": False},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["ok"] is True
    assert [entry.get("reasoning_effort") for entry in seen] == [None, "xhigh", "max"]


def test_validate_model_still_rejects_unknown_reasoning_levels():
    def handler(request):
        if request.url.host == "codex.test" and request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True})
        raise AssertionError("provider called")

    with client_for(handler, enable_model_validation=True) as client:
        response = client.post(
            "/admin/validate-model",
            headers=headers(),
            json={"model": "gpt-7-luna", "reasoning_efforts": ["extreme"]},
        )
    assert response.status_code == 400
