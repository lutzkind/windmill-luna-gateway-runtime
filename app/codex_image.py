from __future__ import annotations

import base64
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

CHATGPT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CHATGPT_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
CHATGPT_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
IMAGE_MODEL = "gpt-image-2"
IMAGE_SIZES = frozenset({"auto", "1024x1024", "1024x1536", "1536x1024"})
IMAGE_QUALITIES = frozenset({"auto", "low", "medium", "high"})
IMAGE_BACKGROUNDS = frozenset({"auto", "opaque", "transparent"})
CODEX_ORIGINATOR = "codex_cli_rs"
CODEX_USER_AGENT = os.environ.get("CODEX_IMAGE_USER_AGENT", "codex_cli_rs/0.0.0 (Linux x86_64; server) windmill-luna-gateway").strip()


def _jwt_payload(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    encoded = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        value = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _access_token_expiring(token: str, *, skew_seconds: int = 300) -> bool:
    exp = _jwt_payload(token).get("exp")
    return isinstance(exp, (int, float)) and float(exp) <= time.time() + skew_seconds


def _canonical_size(value: Any) -> str:
    size = str(value or "auto").strip().lower()
    if size in IMAGE_SIZES:
        return size
    try:
        width_text, height_text = size.split("x", 1)
        width, height = int(width_text), int(height_text)
    except (ValueError, TypeError):
        return "auto"
    if width == height:
        return "1024x1024"
    return "1024x1536" if height > width else "1536x1024"


def normalize_generation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("image prompt is required")
    result: dict[str, Any] = {
        "model": IMAGE_MODEL,
        "prompt": prompt,
        "size": _canonical_size(payload.get("size")),
    }
    quality = str(payload.get("quality") or "auto").strip().lower()
    result["quality"] = quality if quality in IMAGE_QUALITIES else "auto"
    background = str(payload.get("background") or "auto").strip().lower()
    result["background"] = background if background in IMAGE_BACKGROUNDS else "auto"
    return result


def _load_auth(auth_path: Path) -> dict[str, Any]:
    try:
        value = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("codex image authentication is unavailable") from exc
    if not isinstance(value, dict):
        raise RuntimeError("codex image authentication is invalid")
    return value


def _tokens(auth: dict[str, Any]) -> dict[str, Any]:
    tokens = auth.get("tokens")
    if not isinstance(tokens, dict):
        raise RuntimeError("codex image authentication has no ChatGPT tokens")
    if not str(tokens.get("access_token") or "").strip():
        raise RuntimeError("codex image authentication has no access token")
    return tokens


def authorization_headers(auth: dict[str, Any]) -> dict[str, str]:
    tokens = _tokens(auth)
    headers = {
        "Authorization": f"Bearer {str(tokens['access_token']).strip()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "originator": CODEX_ORIGINATOR,
        "User-Agent": CODEX_USER_AGENT,
        "X-Codex-Image-Turn-Id": str(uuid.uuid4()),
    }
    account_id = str(tokens.get("account_id") or "").strip()
    if account_id:
        headers["ChatGPT-Account-ID"] = account_id
    return headers


def auth_diagnostics(auth: dict[str, Any]) -> dict[str, Any]:
    tokens = auth.get("tokens") if isinstance(auth.get("tokens"), dict) else {}
    access_token = str(tokens.get("access_token") or "")
    payload = _jwt_payload(access_token)
    exp = payload.get("exp")
    return {
        "access_token_is_jwt": bool(payload),
        "access_token_exp": int(exp) if isinstance(exp, (int, float)) else None,
        "access_token_expiring": _access_token_expiring(access_token),
        "account_id_present": bool(str(tokens.get("account_id") or "").strip()),
        "refresh_token_present": bool(str(tokens.get("refresh_token") or "").strip()),
    }


def _persist_auth(auth_path: Path, auth: dict[str, Any]) -> None:
    tmp_path = auth_path.with_name(f".{auth_path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(auth, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.chmod(tmp_path, 0o600)
    os.replace(tmp_path, auth_path)


async def _refresh_auth(client: httpx.AsyncClient, auth_path: Path, auth: dict[str, Any]) -> dict[str, Any]:
    tokens = _tokens(auth)
    refresh_token = str(tokens.get("refresh_token") or "").strip()
    if not refresh_token:
        raise RuntimeError("codex image authentication has no refresh token")
    response = await client.post(
        CHATGPT_OAUTH_TOKEN_URL,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        json={
            "client_id": CHATGPT_OAUTH_CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
    )
    print(json.dumps({"event": "codex_image_token_refresh", "status": response.status_code}, sort_keys=True))
    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError(f"codex image token refresh failed with HTTP {response.status_code}")
    data = response.json()
    if not isinstance(data, dict) or not str(data.get("access_token") or "").strip():
        raise RuntimeError("codex image token refresh returned no access token")
    updated = dict(auth)
    updated_tokens = dict(tokens)
    for key in ("access_token", "refresh_token", "id_token"):
        if str(data.get(key) or "").strip():
            updated_tokens[key] = data[key]
    updated["tokens"] = updated_tokens
    updated["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _persist_auth(auth_path, updated)
    return updated


async def generate_codex_image(
    payload: dict[str, Any],
    auth_path: Path,
    *,
    timeout_seconds: float = 180,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.Response:
    normalized = normalize_generation_payload(payload)
    auth = _load_auth(auth_path)
    timeout = httpx.Timeout(timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout, transport=transport, follow_redirects=False) as client:
        access_token = str(_tokens(auth).get("access_token") or "")
        if _access_token_expiring(access_token):
            try:
                auth = await _refresh_auth(client, auth_path, auth)
            except RuntimeError:
                pass
        for attempt in range(2):
            response = await client.post(
                f"{CHATGPT_CODEX_BASE_URL}/images/generations",
                headers=authorization_headers(auth),
                json=normalized,
            )
            print(json.dumps({"event": "codex_image_request", "attempt": attempt + 1, "status": response.status_code, **auth_diagnostics(auth)}, sort_keys=True))
            if response.status_code != 401 or attempt:
                return response
            try:
                auth = await _refresh_auth(client, auth_path, auth)
            except RuntimeError:
                return response
    raise RuntimeError("codex image request failed unexpectedly")
