from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from app import codex_upstream


def test_codex_command_forwards_model_and_reasoning_effort():
    command = codex_upstream._build_codex_command(
        output_path="/tmp/final.txt",
        model="gpt-5.6-luna",
        reasoning_effort="low",
    )

    assert command[command.index("--model") + 1] == "gpt-5.6-luna"
    assert 'model_reasoning_effort="low"' in command
    assert "--sandbox" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "-c" in command
    assert 'approval_policy="never"' in command
    assert "--ignore-user-config" in command


def test_codex_command_forwards_image_paths():
    command = codex_upstream._build_codex_command(
        output_path="/tmp/final.txt",
        model="gpt-5.6-luna",
        image_paths=[Path("/tmp/source.jpg"), Path("/tmp/card.png")],
    )

    assert command[command.index("--image") + 1] == "/tmp/source.jpg"
    assert command[command.index("--image", command.index("--image") + 1) + 1] == "/tmp/card.png"


def test_chat_prompt_preserves_image_inputs():
    image_url = "data:image/jpeg;base64," + base64.b64encode(b"fake-image").decode()
    prompt, images = codex_upstream._prompt_and_images_from_messages(
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect the card."},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}},
            ],
        }]
    )

    assert images == [image_url]
    assert "Inspect the card." in prompt
    assert "Image attachment 1 is supplied" in prompt


def test_responses_prompt_preserves_input_images():
    image_url = "data:image/png;base64," + base64.b64encode(b"fake-image").decode()
    prompt, images = codex_upstream._prompt_and_images_from_responses_input(
        [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Inspect the artwork."},
                {"type": "input_image", "image_url": image_url},
            ],
        }]
    )

    assert images == [image_url]
    assert "Inspect the artwork." in prompt
    assert "Image attachment 1 is supplied" in prompt


def test_data_image_is_materialized_with_safe_extension(tmp_path: Path):
    image_url = "data:image/jpeg;base64," + base64.b64encode(b"jpeg-bytes").decode()
    paths = asyncio.run(codex_upstream._materialize_image_inputs([image_url], tmp_path))

    assert paths == [tmp_path / "input-image-1.jpg"]
    assert paths[0].read_bytes() == b"jpeg-bytes"


def test_image_input_limit_is_bounded():
    image_url = "data:image/jpeg;base64," + base64.b64encode(b"x").decode()
    with pytest.raises(HTTPException, match="too_many_image_inputs"):
        asyncio.run(codex_upstream._materialize_image_inputs(
            [image_url] * (codex_upstream.MAX_IMAGE_INPUTS + 1),
            Path("/tmp"),
        ))


def test_web_search_command_enables_json_event_capture():
    command = codex_upstream._build_codex_command(
        output_path="/tmp/final.txt",
        model="gpt-5.6-luna",
        json_events=True,
    )

    assert "--json" in command


def test_web_search_prompt_allows_only_codex_search():
    prompt = codex_upstream._prompt_from_responses_input(
        "Find the official source.",
        instructions="Return JSON with a Topics array.",
        web_search=True,
    )

    assert "Use Codex's built-in web search tool" in prompt
    assert "Do not use shell commands, local files, MCP tools" in prompt
    assert "<INSTRUCTIONS>\nReturn JSON with a Topics array.\n</INSTRUCTIONS>" in prompt
    assert "<USER>\nFind the official source.\n</USER>" in prompt
    assert "Do not use tools." not in prompt


def test_web_search_event_parser_and_source_extraction():
    stdout = "\n".join(
        [
            json.dumps({
                "type": "item.completed",
                "item": {
                    "id": "search-1",
                    "type": "web_search",
                    "action": {"type": "search", "query": "official source"},
                },
            }),
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 4, "output_tokens": 5}}),
        ]
    )
    calls, usage, agent_message = codex_upstream._parse_codex_events(stdout)
    assert calls[0]["type"] == "web_search_call"
    assert calls[0]["action"]["query"] == "official source"
    assert usage == {"input_tokens": 10, "cached_input_tokens": 4, "output_tokens": 5}
    assert agent_message == ""

    sources = codex_upstream._source_records_from_output(
        json.dumps({
            "source_url": "https://example.com/source",
            "source_title": "Source",
            "source_excerpt": "An exact source excerpt.",
        })
    )
    assert sources == [{
        "url": "https://example.com/source",
        "title": "Source",
        "snippet": "An exact source excerpt.",
    }]


def test_none_reasoning_maps_to_codex_minimal():
    assert codex_upstream._codex_reasoning_effort("none") == "minimal"
    assert codex_upstream._codex_reasoning_effort("low") == "low"


def test_runtime_home_uses_bootstrapped_auth_without_reading_source_mount(monkeypatch: pytest.MonkeyPatch, tmp_path):
    runtime_home = tmp_path / "runtime-home"
    runtime_home.mkdir()
    target_auth = runtime_home / "auth.json"
    target_auth.write_text('{"bootstrapped":true}', encoding="utf-8")

    monkeypatch.setattr(codex_upstream, "RUNTIME_CODEX_HOME", runtime_home)
    monkeypatch.setattr(codex_upstream, "CODEX_AUTH_SOURCE", tmp_path / "root-owned-source.json")

    codex_upstream._prepare_runtime_home()

    assert target_auth.read_text(encoding="utf-8") == '{"bootstrapped":true}'


def test_rotated_runtime_auth_is_persisted_to_shared_server_session(monkeypatch: pytest.MonkeyPatch, tmp_path):
    source_auth = tmp_path / "shared" / "auth.json"
    source_auth.parent.mkdir()
    source_auth.write_text('{"token":"old"}', encoding="utf-8")
    runtime_home = tmp_path / "runtime"
    runtime_home.mkdir()
    (runtime_home / "auth.json").write_text('{"token":"rotated"}', encoding="utf-8")

    monkeypatch.setattr(codex_upstream, "CODEX_AUTH_SOURCE", source_auth)
    monkeypatch.setattr(codex_upstream, "RUNTIME_CODEX_HOME", runtime_home)

    codex_upstream._sync_runtime_auth_to_source()

    assert source_auth.read_text(encoding="utf-8") == '{"token":"rotated"}'


def test_open_json_object_contract_is_validated_without_cli_schema():
    assert codex_upstream._chat_schema({"response_format": {"type": "json_object"}}) is None
    assert codex_upstream._responses_schema({"text": {"format": {"type": "json"}}}) is None
    assert codex_upstream._chat_requires_json({"response_format": {"type": "json_object"}})
    assert codex_upstream._responses_requires_json({"text": {"format": {"type": "json"}}})


def test_json_object_output_is_repaired_without_passing_invalid_schema(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, object]] = []

    async def fake_run(prompt: str, schema: object, require_json: bool, **kwargs: object) -> str:
        calls.append((prompt, schema, require_json, kwargs))
        return "not json" if len(calls) == 1 else '{"ok":true}'

    monkeypatch.setattr(codex_upstream, "_run_codex_once", fake_run)
    result = asyncio.run(codex_upstream._run_codex("return an object", require_json=True))

    assert result.text == '{"ok":true}'
    assert calls[0][1] is None
    assert calls[0][2] is True
    assert len(calls) == 2


def test_json_object_parser_rejects_non_object():
    async def fake_run(prompt: str, schema: object, require_json: bool, **kwargs: object) -> str:
        return "[]"

    original = codex_upstream._run_codex_once
    codex_upstream._run_codex_once = fake_run
    try:
        with pytest.raises(HTTPException, match="not a JSON object"):
            asyncio.run(codex_upstream._run_codex("return an object", require_json=True))
    finally:
        codex_upstream._run_codex_once = original


def test_codex_image_payload_normalizes_social_portrait_size():
    from app.codex_image import normalize_generation_payload
    normalized = normalize_generation_payload({"model": "gpt-image-2", "prompt": "premium restaurant table lighting", "size": "1200x1500", "quality": "medium", "output_format": "jpeg"})
    assert normalized == {"model": "gpt-image-2", "prompt": "premium restaurant table lighting", "size": "1024x1536", "quality": "medium", "background": "auto"}
    assert "output_format" not in normalized


def test_codex_image_auth_uses_chatgpt_account_header():
    from app.codex_image import authorization_headers
    headers = authorization_headers({"tokens": {"access_token": "access-token", "refresh_token": "refresh-token", "account_id": "account-123"}})
    assert headers["Authorization"] == "Bearer access-token"
    assert headers["ChatGPT-Account-ID"] == "account-123"
    assert headers["originator"] == "codex_cli_rs"
    assert headers["User-Agent"].startswith("codex_cli_rs/")
    assert headers["X-Codex-Image-Turn-Id"]
