# Windmill Luna Gateway

Standalone OpenAI-compatible gateway for Windmill.

- Codex/ChatGPT authentication is the primary provider through a pinned `openai-api-server-via-codex` sidecar.
- The official OpenAI API is used only when Codex is unavailable, rate-limited, or out of quota.
- The gateway and provider credentials are supplied only as runtime environment variables.
- The Etsy renderer is a completely separate repository and deployment.

Supported endpoints:

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /health`

Clients authenticate with `X-Luna-Gateway-Token`. Their existing OpenAI bearer token is retained only for the fallback request.
