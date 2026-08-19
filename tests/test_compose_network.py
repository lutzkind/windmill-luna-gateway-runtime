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


def test_codex_auth_refreshes_persist_across_restarts():
    text = COMPOSE.read_text(encoding="utf-8")

    assert "CODEX_HOME: /run/secrets" in text
    assert "LUNA_CODEX_HOME: /run/secrets" in text
    assert "CODEX_AUTH_SOURCE: /run/secrets/auth.json" in text
    assert "/root/.codex/auth.json:/run/secrets/auth.json:rw" in text
    assert "/root/.codex-gateway/auth.json" not in text
    assert "/run/secrets/auth.json:ro" not in text
