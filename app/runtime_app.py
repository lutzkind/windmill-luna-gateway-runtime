from __future__ import annotations

import json
import os

import httpx

from app import main as core


_ORIGINAL_GENERATE_IMAGE = core.Gateway.generate_image
_TRUE_VALUES = {"1", "true", "yes", "on"}


def paid_image_fallback_enabled() -> bool:
    return os.getenv("ALLOW_PAID_OPENAI_IMAGE_FALLBACK", "false").strip().lower() in _TRUE_VALUES


def _is_image_quota_unavailable(response: core.Response) -> bool:
    if response.status_code != 429:
        return False
    try:
        payload = json.loads(response.body)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    error = payload.get("error")
    return isinstance(error, dict) and error.get("code") == "image_quota_unavailable"


async def generate_image_with_optional_paid_fallback(
    self: core.Gateway,
    *,
    payload: dict,
    request_id: str,
    api_key: str,
) -> core.Response:
    response = await _ORIGINAL_GENERATE_IMAGE(
        self,
        payload=payload,
        request_id=request_id,
        api_key=api_key,
    )
    if not paid_image_fallback_enabled() or not _is_image_quota_unavailable(response):
        return response

    if not api_key:
        return core.json_error(
            status_code=502,
            message="Codex image quota is exhausted and the paid OpenAI image fallback credential is unavailable.",
            code="image_fallback_key_missing",
            request_id=request_id,
            fallback_reason="image_quota",
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
        return core.json_error(
            status_code=504,
            message="The paid OpenAI image fallback timed out.",
            code="image_api_timeout",
            request_id=request_id,
            fallback_reason="image_quota",
        )
    except httpx.HTTPError:
        return core.json_error(
            status_code=502,
            message="The paid OpenAI image fallback request failed.",
            code="image_api_network",
            request_id=request_id,
            fallback_reason="image_quota",
        )

    return core.relay_response(
        api_response,
        provider="openai-api-image",
        fallback_used=True,
        fallback_reason="image_quota",
        request_id=request_id,
    )


core.Gateway.generate_image = generate_image_with_optional_paid_fallback
app = core.create_app()
