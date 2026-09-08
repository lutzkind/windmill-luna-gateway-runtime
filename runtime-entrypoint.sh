#!/bin/sh
set -eu

runtime_uid=10001
auth_source="${CODEX_AUTH_SOURCE:-}"
codex_home="${CODEX_HOME:-/run/codex-session}"
runtime_home="${LUNA_CODEX_HOME:-/tmp/luna-codex-home}"
auth_target="$codex_home/auth.json"

if [ "$(id -u)" -eq 0 ]; then
    mkdir -p "$runtime_home"
    chown "$runtime_uid:0" "$runtime_home"
    chmod 0770 "$runtime_home"
    if [ -z "$auth_source" ] || [ "$auth_source" != "$auth_target" ] || [ ! -f "$auth_source" ] || [ -L "$auth_source" ]; then
        echo "codex authentication must be the regular file inside the shared Codex directory" >&2
        exit 78
    fi
    # Host `codex login` and token refreshes replace auth.json atomically.
    # Normalize legacy ownership once, then keep the sidecar root-owned with
    # no Linux capabilities so every newly replaced canonical inode remains
    # readable and writable without a private credential copy.
    chown 0:0 "$auth_source"
    chmod 0600 "$auth_source"
    test -r "$auth_source"
    exec setpriv \
        --reuid=0 \
        --regid=0 \
        --clear-groups \
        --bounding-set=-all \
        --inh-caps=-all \
        --ambient-caps=-all \
        --nnp \
        "$@"
fi

exec "$@"
