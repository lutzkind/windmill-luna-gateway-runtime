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
            # `codex login` replaces auth.json atomically as root:root 0600.
            # Normalize any legacy UID-10001 inode left by older runtimes back
            # to that canonical ownership before dropping all capabilities.
            # The Codex upstream then keeps uid/gid 0 with no Linux capabilities
            # and no-new-privileges, so both token rotation and future login
            # replacements remain readable without restart or credential copies.
            chown 0:0 "$auth_source"
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
