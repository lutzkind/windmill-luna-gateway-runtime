#!/bin/sh
set -eu

runtime_uid=10001
runtime_gid=10001
auth_source="${CODEX_AUTH_SOURCE:-}"
codex_home="${CODEX_HOME:-/tmp/luna-codex-home}"

if [ "$(id -u)" -eq 0 ]; then
    if [ -n "$auth_source" ]; then
        test -r "$auth_source"
        mkdir -p "$codex_home"
        cp "$auth_source" "$codex_home/auth.json"
        chown "$runtime_uid:$runtime_gid" "$codex_home/auth.json"
        chmod 0600 "$codex_home/auth.json"
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
