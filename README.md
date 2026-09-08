# Windmill Luna Gateway

Standalone OpenAI-compatible gateway for Windmill. It is unrelated to and fully separated from the Etsy renderer.

## Routing

- Primary: pinned `openai-api-server-via-codex` sidecar using the host's Codex authentication.
- Multimodal Chat Completions `image_url` and Responses `input_image` inputs are materialized into bounded temporary files and forwarded to Codex CLI with repeated `--image` flags. Image inputs are never reduced to text-only placeholders.
- Fallback: official OpenAI API only when Codex explicitly reports quota or usage-limit exhaustion. Capacity, rate-limit, authentication, network, timeout, upstream, and invalid structured-output failures do not fall back.
- The quota circuit prevents repeated Codex attempts during a confirmed quota-exhaustion window; subsequent Luna requests use the API fallback until that circuit expires.

## Authentication

Windmill authenticates to the gateway with an internal bearer [REDACTED] SHA-256 fingerprint is allowlisted. For Luna quota fallback, the gateway uses its server-side `OPENAI_API_KEY`; the internal Windmill bearer [REDACTED] never forwarded to OpenAI.

The Codex sidecar uses the host's canonical `/root/.codex` directory through
the `/run/codex-session` directory bind. `CODEX_HOME` points directly at that
directory and `CODEX_AUTH_SOURCE` is its `auth.json`; no Coolify-managed
single-file secret mount, startup copy, symlink, or runtime-to-host auth sync
is allowed. This is required because Codex login and refresh replace
`auth.json` atomically. The sidecar normalizes the canonical file to
root-owned `0600` before dropping all Linux capabilities, keeps disposable
non-auth bootstrap files under `LUNA_CODEX_HOME`, and fails closed when the
shared file is missing, malformed, non-canonical, or has unsafe ownership or
permissions. Restart, redeploy, and container recreation therefore reopen the
same host-backed path rather than restoring a stored credential snapshot.

The `/healthz` response reports only non-secret auth presence, ownership,
permissions, canonical-path, writability, and JSON-validity facts. It never
prints token contents.

## Endpoints

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /health`

The Codex upstream health response exposes `image_input_forwarding`, the transport
(`codex_exec_image_flags`), and the bounded image count/size limits. Image URLs are
accepted only over HTTP(S); data URLs must be base64-encoded raster images.
The gateway accepts multimodal JSON bodies up to 16 MiB by default; the larger
passthrough limit does not apply to model completion requests.

`ENABLE_TEST_CONTROLS` is enabled only for live fallback verification and is disabled immediately afterward.
