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
