# Windmill Luna Gateway

Standalone OpenAI-compatible gateway for Windmill. It is unrelated to and fully separated from the Etsy renderer.

## Routing

- Primary: pinned `openai-api-server-via-codex` sidecar using the host's Codex authentication.
- Multimodal Chat Completions `image_url` and Responses `input_image` inputs are materialized into bounded temporary files and forwarded to Codex CLI with repeated `--image` flags. Image inputs are never reduced to text-only placeholders.
- Fallback: official OpenAI API only when Codex explicitly reports quota or usage-limit exhaustion. Capacity, rate-limit, authentication, network, timeout, upstream, and invalid structured-output failures do not fall back.
- The quota circuit prevents repeated Codex attempts during a confirmed quota-exhaustion window; subsequent Luna requests use the API fallback until that circuit expires.

## Authentication

Windmill authenticates to the gateway with an internal bearer whose SHA-256 fingerprint is allowlisted. For Luna quota fallback, the gateway uses its server-side `OPENAI_API_KEY`; the internal Windmill bearer is never forwarded to OpenAI.

The Codex sidecar uses the host's canonical `/root/.codex` directory through
the `/run/codex-session` directory bind. `CODEX_HOME` points directly at that
directory and `CODEX_AUTH_SOURCE` is its `auth.json`; no Coolify-managed
single-file secret mount, startup copy, symlink, or runtime-to-host auth sync
is allowed. This is required because Codex login and refresh replace
`auth.json` atomically. The sidecar normalizes the canonical file to
root-owned, shared-runtime-group-writable `0660` before dropping all Linux
capabilities, keeps disposable non-auth bootstrap files under
`LUNA_CODEX_HOME`, and fails closed when the shared file is missing, malformed,
non-canonical, or has unsafe ownership or permissions. A lightweight root
supervisor re-normalizes ownership and mode if another writer (host Codex
login, the Etsy renderer auth sync, or the Codex executor child) replaces the
file, so every consumer that shares the canonical file keeps read access.
Restart, redeploy, and container recreation therefore reopen the same
host-backed path rather than restoring a stored credential snapshot.

The `/healthz` response reports only non-secret auth presence, ownership,
group, permissions, canonical-path, writability, and JSON-validity facts. It
never prints token contents. If the canonical `auth.json` is atomically
replaced with broader mode bits, the sidecar tightens it back to `0660` during
health/request validation; ownership and group drift is repaired by the
supervisor loop, and the health check reports the repair state rather than
accepting unsafe ownership or non-canonical paths.

The outer gateway `/health` endpoint is a readiness check: it verifies the
Codex sidecar's `/healthz` and returns HTTP 503 when the primary upstream is
not healthy. This prevents a healthy gateway process from masking an unusable
Luna proxy path. `/health` also reports the canonical model-routing facts:
`luna_auto_model` (the concrete model `luna-auto` currently resolves to),
`model_aliases`, and `reasoning_efforts`.

## Model policy

`LUNA_AUTO_MODEL` is the single canonical selected Luna model for the whole
platform. The compose file expands it into `ALLOWED_MODELS` and
`MODEL_ALIASES_JSON`, so callers only ever need to send the caller-facing alias
`luna-auto`; the concrete version stays behind the gateway. Advancing Luna
generations is a change to `LUNA_AUTO_MODEL` alone, which the scheduled
`f/admins/luna-model-upgrade-check` Windmill job performs after validating a
candidate. Model routing is deliberately independent of reasoning: `luna-auto`
selects only the model version and never rewrites, infers, or normalizes the
caller's reasoning effort.

`ALLOWED_MODELS` is enforced on every completion request when it is non-empty. A request
whose caller-supplied model, or the model its `MODEL_ALIASES_JSON` alias resolves to, is not
listed is rejected with HTTP 400 `model_not_allowed` before any provider call. Aliases can
therefore only route between allowlisted names; they cannot introduce a new model.

`LUNA_AUTO_MODEL` is required at startup and is shared with the Codex sidecar. When
`ALLOWED_MODELS` is unset the default `luna-auto,<LUNA_AUTO_MODEL>` applies.
An explicitly empty value disables the allowlist; it exists only as a migration escape hatch
and is not recommended. `GET /health` reports `model_allowlist_enforced`, `luna_auto_model`,
`model_aliases`, and `reasoning_efforts` so the active routing mode is observable without
exposing credentials.

## Candidate validation

When `ENABLE_MODEL_VALIDATION=true`, the existing upgrade job can call
`GET /admin/discover-models` to read only the Luna-family IDs visible to the configured
OpenAI account. Discovery never changes the active mapping. The job still requires
candidate validation before advancing `LUNA_AUTO_MODEL`, so an account-visible model name
alone cannot trigger an upgrade.

Allowlisted callers may `POST /admin/validate-model`
with `{"model": "<candidate>", "reasoning_efforts": ["low", "medium", "high"], "smoke": true}`.
The endpoint runs the candidate through the Codex upstream, checking upstream readiness,
every requested reasoning level, and a deterministic smoke prompt. It never changes the
active routing mapping, so the scheduled upgrade job can validate a candidate before it
touches `LUNA_AUTO_MODEL`. The endpoint returns HTTP 404 when validation is disabled.

## Endpoints

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /health`
- `GET /admin/discover-models` (only when `ENABLE_MODEL_VALIDATION=true`)
- `POST /admin/validate-model` (only when `ENABLE_MODEL_VALIDATION=true`)

The Codex upstream health response exposes `image_input_forwarding`, the transport
(`codex_exec_image_flags`), and the bounded image count/size limits. Image URLs are
accepted only over HTTP(S); data URLs must be base64-encoded raster images.
The gateway accepts multimodal JSON bodies up to 16 MiB by default; the larger
passthrough limit does not apply to model completion requests.

`ENABLE_TEST_CONTROLS` is enabled only for live fallback verification and is disabled immediately afterward.
