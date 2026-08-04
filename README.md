# Windmill Luna Gateway

Standalone OpenAI-compatible gateway for Windmill. It is unrelated to and fully separated from the Etsy renderer.

## Routing

- Primary: pinned `openai-api-server-via-codex` sidecar using the host's Codex authentication.
- Fallback: official OpenAI API on Codex quota, rate limit, authentication, network, timeout, upstream failure, or invalid structured output.
- Circuit breaking prevents repeated Codex attempts while an outage or quota condition is active.

## Authentication

Windmill sends its existing OpenAI API key as `Authorization: Bearer ...`. The gateway authorizes callers using a SHA-256 fingerprint; the raw key is not stored in this repository. The bearer is forwarded only when official-API fallback is required.

## Endpoints

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /health`

`ENABLE_TEST_CONTROLS` is enabled only for live fallback verification and is disabled immediately afterward.
