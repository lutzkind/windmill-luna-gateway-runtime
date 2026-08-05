from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from app import codex_upstream


def test_open_json_object_contract_is_validated_without_cli_schema():
    assert codex_upstream._chat_schema({"response_format": {"type": "json_object"}}) is None
    assert codex_upstream._responses_schema({"text": {"format": {"type": "json"}}}) is None
    assert codex_upstream._chat_requires_json({"response_format": {"type": "json_object"}})
    assert codex_upstream._responses_requires_json({"text": {"format": {"type": "json"}}})


def test_json_object_output_is_repaired_without_passing_invalid_schema(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, object]] = []

    async def fake_run(prompt: str, schema: object, require_json: bool) -> str:
        calls.append((prompt, schema, require_json))
        return "not json" if len(calls) == 1 else '{"ok":true}'

    monkeypatch.setattr(codex_upstream, "_run_codex_once", fake_run)
    result = asyncio.run(codex_upstream._run_codex("return an object", require_json=True))

    assert result == '{"ok":true}'
    assert calls[0][1] is None
    assert calls[0][2] is True
    assert len(calls) == 2


def test_json_object_parser_rejects_non_object():
    async def fake_run(prompt: str, schema: object, require_json: bool) -> str:
        return "[]"

    original = codex_upstream._run_codex_once
    codex_upstream._run_codex_once = fake_run
    try:
        with pytest.raises(HTTPException, match="not a JSON object"):
            asyncio.run(codex_upstream._run_codex("return an object", require_json=True))
    finally:
        codex_upstream._run_codex_once = original
