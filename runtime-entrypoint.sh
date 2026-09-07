#!/bin/sh
set -eu

runtime_uid=10001
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
            # Keep the host auth file as the single credential source. Host
            # `codex login` replaces auth.json atomically as root:root 0600;
            # an unprivileged long-running sidecar would then lose access to
            # the new inode until restart. The Codex upstream therefore keeps
            # uid/gid 0 but drops every Linux capability and sets
            # no-new-privileges before starting. This survives both Codex
            # token rotation and future interactive login replacement without
            # copying or snapshotting credentials.
            chmod 0600 "$auth_source"
            rm -f "$auth_target"
            ln -s "$auth_source" "$auth_target"
        fi
    fi
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
