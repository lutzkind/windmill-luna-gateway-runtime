# Windmill Luna Gateway

Standalone OpenAI-compatible gateway for Windmill. It is unrelated to and fully separated from the Etsy renderer.

## Routing

- Primary: pinned `openai-api-server-via-codex` sidecar using the host's Codex authentication.
- Multimodal Chat Completions `image_url` and Responses `input_image` inputs are materialized into bounded temporary files and forwarded to Codex CLI with repeated `--image` flags. Image inputs are never reduced to text-only placeholders.
- Fallback: official OpenAI API on Codex quota, rate limit, authentication, network, timeout, upstream failure, or invalid structured output.
- Circuit breaking prevents repeated Codex attempts while an outage or quota condition is active.

## Authentication

Windmill sends its existing OpenAI API key as `Authorization: Bearer ...`. The gateway authorizes callers using a SHA-256 fingerprint; the raw key is not stored in this repository. The bearer is forwarded only when official-API fallback is required.

## Endpoints

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /health`

The Codex upstream health response exposes `image_input_forwarding`, the transport
(`codex_exec_image_flags`), and the bounded image count/size limits. Image URLs are
accepted only over HTTP(S); data URLs must be base64-encoded raster images.

`ENABLE_TEST_CONTROLS` is enabled only for live fallback verification and is disabled immediately afterward.
