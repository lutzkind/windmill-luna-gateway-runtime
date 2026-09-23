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
prints token contents. If a root-owned canonical `auth.json` is atomically
replaced with broader mode bits, the sidecar tightens it back to `0600` during
health/request validation without accepting unsafe ownership or non-canonical
paths.

The outer gateway `/health` endpoint is a readiness check: it verifies the
Codex sidecar's `/healthz` and returns HTTP 503 when the primary upstream is
not healthy. This prevents a healthy gateway process from masking an unusable
Luna proxy path.

## Model policy

`ALLOWED_MODELS` is enforced on every completion request when it is non-empty. A request
whose caller-supplied model, or the model its `MODEL_ALIASES_JSON` alias resolves to, is not
listed is rejected with HTTP 400 `model_not_allowed` before any provider call. Aliases can
therefore only route between allowlisted names; they cannot introduce a new model.

When `ALLOWED_MODELS` is unset the default `gpt-6-luna,luna-auto` applies. An explicitly
empty value disables the allowlist; it exists only as a migration escape hatch and is not
recommended. `GET /health` reports `model_allowlist_enforced` so the active mode is
observable without exposing the list.

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
