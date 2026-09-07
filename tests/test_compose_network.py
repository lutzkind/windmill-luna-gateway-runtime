from pathlib import Path


COMPOSE = Path(__file__).parents[1] / "docker-compose.yaml"


def test_windmill_gateway_has_private_windmill_network_path():
    text = COMPOSE.read_text(encoding="utf-8")

    gateway_block = text.split("  windmill-luna-gateway:\n", 1)[1].split(
        "\nnetworks:\n", 1
    )[0]
    assert "networks: [gateway-private, coolify, windmill]" in gateway_block
    assert "  windmill:\n    external: true\n" in text
    assert "name: ${WINDMILL_NETWORK_NAME:-m9qaud6gadgni5bxty30bkdl}" in text


def test_gateway_containers_are_least_privilege():
    text = COMPOSE.read_text(encoding="utf-8")

    assert "read_only: true" in text
    assert "no-new-privileges:true" in text
    assert "cap_drop: [ALL]" in text
    assert "CODEX_HOME: /tmp/luna-codex-home" in text
    assert "LUNA_CODEX_HOME: /tmp/luna-codex-home" in text
    assert "CODEX_AUTH_SOURCE: /run/codex-session/auth.json" in text
    assert "/root/.codex:/run/codex-session:rw" in text
    dockerfile_text = (Path(__file__).parents[1] / "Dockerfile").read_text(encoding="utf-8")
    assert "chown 10001:0 /tmp/luna-codex-home" in dockerfile_text
    assert "chmod 0770 /tmp/luna-codex-home" in dockerfile_text
    entrypoint_text = (Path(__file__).parents[1] / "runtime-entrypoint.sh").read_text(encoding="utf-8")
    assert 'chown "$runtime_uid:0" "$codex_home"' in entrypoint_text
    assert 'chmod 0770 "$codex_home"' in entrypoint_text
    assert "--bounding-set=-all" in entrypoint_text
    assert "--inh-caps=-all" in entrypoint_text
    assert "--ambient-caps=-all" in entrypoint_text
    assert "--nnp" in entrypoint_text


def test_codex_auth_refreshes_persist_without_coolify_file_snapshots():
    text = COMPOSE.read_text(encoding="utf-8")
    entrypoint = (Path(__file__).parents[1] / "runtime-entrypoint.sh").read_text(encoding="utf-8")

    assert "CODEX_HOME: /tmp/luna-codex-home" in text
    assert "LUNA_CODEX_HOME: /tmp/luna-codex-home" in text
    assert "CODEX_AUTH_SOURCE: /run/codex-session/auth.json" in text
    assert "/root/.codex:/run/codex-session:rw" in text
    assert "/root/.codex/auth.json:/run/secrets/codex-auth.json:rw" not in text
    assert "/root/.codex-gateway/auth.json" not in text
    assert 'chown "$runtime_uid:$runtime_gid" "$auth_source"' not in entrypoint
    assert 'ln -s "$auth_source" "$auth_target"' in entrypoint
    assert 'cp "$auth_source" "$auth_target"' not in entrypoint


def test_codex_auth_survives_future_host_login_atomic_replacement():
    entrypoint = (Path(__file__).parents[1] / "runtime-entrypoint.sh").read_text(encoding="utf-8")

    # Interactive host Codex login atomically replaces auth.json as root-owned.
    # The long-running upstream must therefore retain uid 0 rather than depend
    # on a one-time chown that becomes stale after the next replacement.
    assert "--reuid=0" in entrypoint
    assert "--regid=0" in entrypoint
    assert 'chown "$runtime_uid:$runtime_gid" "$auth_source"' not in entrypoint
    # Root identity is retained only after all capabilities are removed and
    # no-new-privileges is enabled.
    assert "--bounding-set=-all" in entrypoint
    assert "--inh-caps=-all" in entrypoint
    assert "--ambient-caps=-all" in entrypoint
    assert "--nnp" in entrypoint
