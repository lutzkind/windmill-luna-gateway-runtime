#!/bin/sh
set -eu

runtime_uid=10001
runtime_gid=10001
auth_source="${CODEX_AUTH_SOURCE:-}"
codex_home="${CODEX_HOME:-/tmp/luna-codex-home}"
auth_target="$codex_home/auth.json"

if [ "$(id -u)" -eq 0 ]; then
    mkdir -p "$codex_home"
    chown "$runtime_uid:0" "$codex_home"
    chmod 0770 "$codex_home"
    if [ -n "$auth_source" ]; then
        test -r "$auth_source"
        if [ "$auth_source" != "$auth_target" ]; then
            cp "$auth_source" "$auth_target"
        fi
        chown "$runtime_uid:$runtime_gid" "$auth_target"
        chmod 0600 "$auth_target"
    fi
    exec setpriv \
        --reuid="$runtime_uid" \
        --regid="$runtime_gid" \
        --clear-groups \
        --bounding-set=-all \
        --inh-caps=-all \
        --ambient-caps=-all \
        --nnp \
        "$@"
fi

exec "$@"
