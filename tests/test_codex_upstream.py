from __future__ import annotations

import asyncio
import json

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


def test_web_search_command_enables_json_event_capture():
    command = codex_upstream._build_codex_command(
        output_path="/tmp/final.txt",
        model="gpt-5.6-luna",
        json_events=True,
    )

    assert "--json" in command


def test_web_search_prompt_allows_only_codex_search():
    prompt = codex_upstream._prompt_from_responses_input(
        "Find the official source.", web_search=True
    )

    assert "Use Codex's built-in web search tool" in prompt
    assert "Do not use shell commands, local files, MCP tools" in prompt
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
