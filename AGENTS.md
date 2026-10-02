# AGENTS.md — windmill-luna-gateway-runtime

Purpose: TODO — one line describing this repository.
GitHub: `lutzkind/windmill-luna-gateway-runtime` · Canonical checkout: TODO — add it to `/root/REPO_MAP.md`
· Default branch: `main`.

## Start here

- Docs: TODO — link this repository's canonical docs (`README.md`, `ARCHITECTURE.md`,
  `ROADMAP.md`, `DECISIONS.md`, `HANDOFF.md`, `docs/`).
- Existing project rules: TODO — link any pre-existing agent or contribution docs.
- Host map: `/root/REPO_MAP.md` (canonical checkouts, duplicates, production).
- `/root/mcp-shared/chatgpt/**` is continuity/history evidence, not the source of truth.

## Branch / state rule

- Local checkout is not proof of `origin/main`; `origin/main` is not proof of
  production. Check `git status -sb`, compare with `git rev-parse origin/main`,
  and inspect live production read-only when the task depends on deployed state.
- Do not switch, reset, pull, merge, or clean without task authorization.

## Commands

| Purpose | Command |
|---|---|
| Install | TODO |
| Targeted test | TODO |
| Full suite | TODO |
| Lint/typecheck/build | TODO or `none configured` |

TODO: if this repository has no tests, state that explicitly here instead of
inventing commands.

## Production

- TODO: name the Coolify app UUID/branch or the Windmill owner from
  `/root/windmill-production/OWNERS.yaml`, or state `No production deployment`
  explicitly. Never infer production from Git state alone.

## Traps

- TODO — known footguns, generated files, deploy hazards.

## Do not read/search by default

- `/root/agent-tmp/**` (disposable scratch), `/root/mcp-shared/chatgpt/**`
  (history, not source of truth), `/root/backups/**` (retired copies).
