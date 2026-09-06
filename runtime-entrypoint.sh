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
            # The host auth file is the single credential source. Older
            # runtimes copied it into tmpfs, so refresh-token rotation was
            # lost on restart and sibling Codex consumers diverged.
            chown "$runtime_uid:$runtime_gid" "$auth_source"
            chmod 0600 "$auth_source"
            rm -f "$auth_target"
            ln -s "$auth_source" "$auth_target"
        fi
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
