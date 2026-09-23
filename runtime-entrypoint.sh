#!/bin/sh
set -eu

runtime_uid=10001
runtime_gid=10001
auth_source="${CODEX_AUTH_SOURCE:-}"
codex_home="${CODEX_HOME:-/run/codex-session}"
runtime_home="${LUNA_CODEX_HOME:-/tmp/luna-codex-home}"
auth_target="$codex_home/auth.json"
shared_auth_mode=660

normalize_auth_file() {
    # The canonical auth.json is shared by several consumers: host Codex
    # login, the Codex executor child (uid 10001), the Etsy renderer auth
    # sync, and this sidecar. Keep it a root-owned regular file with the
    # shared runtime group and group-writable mode so every consumer can read
    # it and the dropped-privilege consumers can refresh tokens through it.
    [ -f "$auth_source" ] || return 0
    [ -L "$auth_source" ] && return 0
    current_owner="$(stat -c '%u:%g' "$auth_source" 2>/dev/null || echo '')"
    current_mode="$(stat -c '%a' "$auth_source" 2>/dev/null || echo '')"
    if [ "$current_owner" != "0:$runtime_gid" ] || [ "$current_mode" != "$shared_auth_mode" ]; then
        chown "0:$runtime_gid" "$auth_source" 2>/dev/null || true
        chmod "$shared_auth_mode" "$auth_source" 2>/dev/null || true
    fi
}

# Repair ownership and mode drift left behind by any other writer of the
# canonical file. Runs as a detached root process that exits with the
# sidecar, closes every inherited descriptor so it cannot hold the container's
# stdio pipes open, and never blocks request handling.
supervise_auth_file() {
    LUNA_SUPERVISOR_MAIN_PID="$1" \
    CODEX_AUTH_SOURCE="$auth_source" \
    LUNA_SHARED_RUNTIME_GID="$runtime_gid" \
    LUNA_SHARED_AUTH_MODE="$shared_auth_mode" \
    python3 -c '
import os, stat, time

main_pid = int(os.environ["LUNA_SUPERVISOR_MAIN_PID"])
auth = os.environ["CODEX_AUTH_SOURCE"]
gid = int(os.environ["LUNA_SHARED_RUNTIME_GID"])
mode = int(os.environ["LUNA_SHARED_AUTH_MODE"], 8)

for fd in range(3, 256):
    try:
        os.close(fd)
    except OSError:
        pass


def sidecar_alive(pid):
    try:
        with open("/proc/%d/stat" % pid, encoding="utf-8") as handle:
            return handle.read().split()[2] != "Z"
    except OSError:
        return False


while sidecar_alive(main_pid):
    time.sleep(2)
    try:
        metadata = os.lstat(auth)
    except OSError:
        continue
    if not stat.S_ISREG(metadata.st_mode):
        continue
    if (
        metadata.st_uid == 0
        and metadata.st_gid == gid
        and stat.S_IMODE(metadata.st_mode) == mode
    ):
        continue
    try:
        os.chown(auth, 0, gid)
        os.chmod(auth, mode)
    except OSError:
        pass
' </dev/null >/dev/null 2>&1 &
}

if [ "$(id -u)" -eq 0 ]; then
    main_pid=$$
    mkdir -p "$runtime_home"
    chown "$runtime_uid:0" "$runtime_home"
    chmod 0770 "$runtime_home"
    if [ -z "$auth_source" ] || [ "$auth_source" != "$auth_target" ] || [ ! -f "$auth_source" ] || [ -L "$auth_source" ]; then
        echo "codex authentication must be the regular file inside the shared Codex directory" >&2
        exit 78
    fi
    # Host `codex login` and token refreshes replace auth.json atomically.
    # Normalize the shared file before dropping capabilities and keep a root
    # supervisor watching it so any writer that replaces ownership or mode
    # cannot starve the other consumers of the same canonical credential.
    normalize_auth_file
    test -r "$auth_source"
    umask 077
    supervise_auth_file "$main_pid"
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
