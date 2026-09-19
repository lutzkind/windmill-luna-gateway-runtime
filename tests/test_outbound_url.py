from __future__ import annotations

import asyncio
import logging

import httpx
import pytest

from app import outbound_url
from app.outbound_url import (
    OutboundTarget,
    OutboundURLRejected,
    fetch_outbound,
    is_public_address,
    redact_url,
    validate_outbound_url,
    verify_connected_peer,
)

PUBLIC_IP = "93.184.216.34"


def resolver_for(mapping):
    calls = []

    async def resolve(hostname, port):
        calls.append((hostname, port))
        if hostname in mapping:
            return list(mapping[hostname])
        raise OutboundURLRejected("dns_failure", "outbound hostname did not resolve")

    resolve.calls = calls
    return resolve


def validate(url, *, addresses=(PUBLIC_IP,), expected_host="images.example", **kwargs):
    resolver = resolver_for({expected_host: addresses})
    target = asyncio.run(validate_outbound_url(url, resolver=resolver, **kwargs))
    return target, resolver.calls


def mock_factory(handler):
    def factory(target):
        return httpx.MockTransport(handler)

    return factory


@pytest.mark.parametrize(
    "url",
    [
        "ftp://images.example/a.png",
        "file:///etc/passwd",
        "gopher://images.example/x",
        "images.example/a.png",
    ],
)
def test_non_http_schemes_are_rejected(url):
    with pytest.raises(OutboundURLRejected) as excinfo:
        validate(url)
    assert excinfo.value.code == "unsupported_scheme"


def test_credentials_in_url_are_rejected():
    with pytest.raises(OutboundURLRejected) as excinfo:
        validate("https://user:password@images.example/a.png")
    assert excinfo.value.code == "url_credentials"


def test_hostname_is_normalized_before_resolution():
    target, calls = validate(
        "https://IMAGES.Example.COM.:8443/a.png", expected_host="images.example.com"
    )
    assert target.hostname == "images.example.com"
    assert target.host_header == "images.example.com:8443"
    assert calls == [("images.example.com", 8443)]


def test_idna_hostname_is_normalized_before_resolution():
    target, calls = validate(
        "https://b\u00fccher.example/a.png", expected_host="xn--bcher-kva.example"
    )
    assert target.hostname == "xn--bcher-kva.example"
    assert calls == [("xn--bcher-kva.example", 443)]


@pytest.mark.parametrize(
    "url",
    [
        "https://a..b/a.png",
        "https://-bad.example/a.png",
        "https://bad-.example/a.png",
        "https://" + "x" * 64 + ".example/a.png",
        "https:///a.png",
    ],
)
def test_invalid_hostnames_are_rejected(url):
    with pytest.raises(OutboundURLRejected) as excinfo:
        validate(url)
    assert excinfo.value.code == "invalid_hostname"


def test_whitespace_and_invalid_ports_are_rejected():
    with pytest.raises(OutboundURLRejected) as excinfo:
        validate("https://exa mple.com/a.png")
    assert excinfo.value.code == "invalid_url"
    for url in ("https://images.example:0/a.png", "https://images.example:99999/a.png"):
        with pytest.raises(OutboundURLRejected) as excinfo:
            validate(url)
        assert excinfo.value.code == "invalid_url"


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/a.png",
        "http://api.localhost/a.png",
        "http://printer.local/a.png",
        "http://db.internal/a.png",
        "http://router.home.arpa/a.png",
    ],
)
def test_local_hostnames_are_rejected_without_resolution(url):
    resolver = resolver_for({"images.example": [PUBLIC_IP]})
    with pytest.raises(OutboundURLRejected) as excinfo:
        asyncio.run(validate_outbound_url(url, resolver=resolver))
    assert excinfo.value.code == "hostname_not_allowed"
    assert resolver.calls == []


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.5",
        "172.16.9.9",
        "192.168.1.10",
        "169.254.169.254",
        "100.64.0.1",
        "240.0.0.1",
        "0.0.0.0",
        "224.0.0.1",
        "203.0.113.9",
        "::1",
        "fd00::1",
        "fe80::1",
        "fec0::1",
        "::ffff:10.0.0.1",
    ],
)
def test_non_public_resolution_is_rejected(address):
    with pytest.raises(OutboundURLRejected) as excinfo:
        validate("https://images.example/a.png", addresses=(address,))
    assert excinfo.value.code == "address_not_public"


def test_mixed_public_and_private_resolution_fails_closed():
    with pytest.raises(OutboundURLRejected) as excinfo:
        validate("https://images.example/a.png", addresses=(PUBLIC_IP, "10.0.0.9"))
    assert excinfo.value.code == "address_not_public"


def test_ipv4_mapped_public_address_is_unwrapped_and_allowed():
    target, _ = validate(
        "https://images.example/a.png", addresses=("::ffff:93.184.216.34",)
    )
    assert target.addresses == ("::ffff:93.184.216.34",)
    assert is_public_address("::ffff:93.184.216.34")


def test_ip_literals_are_validated_without_dns():
    resolver = resolver_for({"images.example": [PUBLIC_IP]})
    with pytest.raises(OutboundURLRejected) as excinfo:
        asyncio.run(validate_outbound_url("http://127.0.0.1:8080/x", resolver=resolver))
    assert excinfo.value.code == "address_not_public"
    target = asyncio.run(
        validate_outbound_url("http://93.184.216.34/x", resolver=resolver)
    )
    assert target.addresses == (PUBLIC_IP,)
    assert resolver.calls == []


def test_dns_failure_is_rejected():
    async def failing(hostname, port):
        raise OSError("resolver unavailable")

    with pytest.raises(OutboundURLRejected) as excinfo:
        asyncio.run(validate_outbound_url("https://images.example/a.png", resolver=failing))
    assert excinfo.value.code == "dns_failure"


def test_system_resolver_is_used_when_none_is_injected(monkeypatch):
    async def fake_system_resolve(hostname, port):
        return [PUBLIC_IP]

    monkeypatch.setattr(outbound_url, "_system_resolve", fake_system_resolve)
    target = asyncio.run(validate_outbound_url("https://images.example/a.png"))
    assert target.addresses == (PUBLIC_IP,)


def test_redact_url_drops_credentials_query_and_fragment():
    assert (
        redact_url("https://user:pass@example.com:8443/a/b?q=1#f")
        == "https://example.com:8443/a/b"
    )
    assert (
        redact_url("http://[2001:db8::1]:8080/x?y=z")
        == "http://[2001:db8::1]:8080/x"
    )
    assert redact_url("") == "<empty-url>"


class _FakeStream:
    def __init__(self, peer):
        self._peer = peer

    def get_extra_info(self, name):
        return self._peer if name == "server_addr" else None


def _target() -> OutboundTarget:
    return OutboundTarget(
        url="https://images.example/a.png",
        scheme="https",
        hostname="images.example",
        port=443,
        addresses=(PUBLIC_IP,),
        host_header="images.example",
    )


def test_connected_peer_must_match_the_pinned_address_set():
    verify_connected_peer({"network_stream": _FakeStream((PUBLIC_IP, 443))}, _target())
    for peer in ("10.0.0.9", "::ffff:10.0.0.9", "127.0.0.1"):
        with pytest.raises(OutboundURLRejected) as excinfo:
            verify_connected_peer({"network_stream": _FakeStream((peer, 443))}, _target())
        assert excinfo.value.code == "peer_address_mismatch"
    verify_connected_peer({}, _target())


def test_fetch_pins_resolved_address_and_keeps_logical_host_header():
    requests = []
    sni = []

    def handler(request):
        requests.append((request.url.host, request.headers.get("host"), str(request.url)))
        sni.append(request.extensions.get("sni_hostname"))
        return httpx.Response(200, content=b"image", headers={"content-type": "image/png"})

    resolver = resolver_for({"images.example": [PUBLIC_IP]})
    response = asyncio.run(
        fetch_outbound(
            "https://images.example/a.png?signature=abc",
            max_bytes=64,
            timeout_seconds=5,
            resolver=resolver,
            transport_factory=mock_factory(handler),
        )
    )

    assert response.status_code == 200
    assert response.content == b"image"
    assert resolver.calls == [("images.example", 443)]
    assert requests == [
        (PUBLIC_IP, "images.example", "https://93.184.216.34/a.png?signature=abc")
    ]
    assert sni == ["images.example"]
    assert response.url == "https://images.example/a.png"


def test_redirects_are_revalidated_and_bounded():
    requests = []

    def handler(request):
        requests.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://images.example/next.png"})

    resolver = resolver_for({"images.example": [PUBLIC_IP]})
    with pytest.raises(OutboundURLRejected) as excinfo:
        asyncio.run(
            fetch_outbound(
                "https://images.example/a.png",
                max_bytes=64,
                max_redirects=2,
                resolver=resolver,
                transport_factory=mock_factory(handler),
            )
        )
    assert excinfo.value.code == "too_many_redirects"
    assert len(requests) == 3


def test_relative_redirect_is_resolved_against_the_validated_url():
    requests = []

    def handler(request):
        requests.append(str(request.url))
        if len(requests) == 1:
            return httpx.Response(302, headers={"location": "/second.png"})
        return httpx.Response(200, content=b"ok", headers={"content-type": "image/png"})

    resolver = resolver_for({"images.example": [PUBLIC_IP]})
    response = asyncio.run(
        fetch_outbound(
            "https://images.example/first.png",
            max_bytes=64,
            resolver=resolver,
            transport_factory=mock_factory(handler),
        )
    )
    assert response.redirects == 1
    assert requests == [
        "https://93.184.216.34/first.png",
        "https://93.184.216.34/second.png",
    ]
    assert response.url == "https://images.example/second.png"


def test_redirect_to_private_address_is_rejected():
    requests = []

    def handler(request):
        requests.append(str(request.url))
        return httpx.Response(
            302, headers={"location": "http://169.254.169.254/latest/meta-data"}
        )

    resolver = resolver_for({"images.example": [PUBLIC_IP]})
    with pytest.raises(OutboundURLRejected) as excinfo:
        asyncio.run(
            fetch_outbound(
                "https://images.example/a.png",
                max_bytes=64,
                resolver=resolver,
                transport_factory=mock_factory(handler),
            )
        )
    assert excinfo.value.code == "address_not_public"
    assert len(requests) == 1


def test_redirect_without_location_is_rejected():
    def handler(request):
        return httpx.Response(302)

    with pytest.raises(OutboundURLRejected) as excinfo:
        asyncio.run(
            fetch_outbound(
                "https://images.example/a.png",
                max_bytes=64,
                resolver=resolver_for({"images.example": [PUBLIC_IP]}),
                transport_factory=mock_factory(handler),
            )
        )
    assert excinfo.value.code == "redirect_without_location"


def test_response_size_limit_is_enforced():
    def handler(request):
        return httpx.Response(200, content=b"x" * 4096)

    with pytest.raises(OutboundURLRejected) as excinfo:
        asyncio.run(
            fetch_outbound(
                "https://images.example/big.png",
                max_bytes=1024,
                resolver=resolver_for({"images.example": [PUBLIC_IP]}),
                transport_factory=mock_factory(handler),
            )
        )
    assert excinfo.value.code == "response_too_large"
    assert excinfo.value.status_code == 413


def test_fetch_timeout_and_network_errors_are_reported():
    def timing_out(request):
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(OutboundURLRejected) as excinfo:
        asyncio.run(
            fetch_outbound(
                "https://images.example/a.png",
                max_bytes=64,
                resolver=resolver_for({"images.example": [PUBLIC_IP]}),
                transport_factory=mock_factory(timing_out),
            )
        )
    assert excinfo.value.code == "fetch_timeout"

    def refusing(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(OutboundURLRejected) as excinfo:
        asyncio.run(
            fetch_outbound(
                "https://images.example/a.png",
                max_bytes=64,
                resolver=resolver_for({"images.example": [PUBLIC_IP]}),
                transport_factory=mock_factory(refusing),
            )
        )
    assert excinfo.value.code == "fetch_failed"


def test_rejections_log_only_redacted_targets(caplog):
    with caplog.at_level(logging.WARNING, logger="app.outbound_url"):
        with pytest.raises(OutboundURLRejected):
            asyncio.run(
                fetch_outbound(
                    "https://user:pa55@images.example/a.png?token=supersecret",
                    max_bytes=64,
                    resolver=resolver_for({"images.example": [PUBLIC_IP]}),
                )
            )
    assert "supersecret" not in caplog.text
    assert "pa55" not in caplog.text
    assert "user@" not in caplog.text


def test_blocked_fetch_logs_redacted_target(caplog):
    with caplog.at_level(logging.WARNING, logger="app.outbound_url"):
        with pytest.raises(OutboundURLRejected):
            asyncio.run(
                fetch_outbound(
                    "https://images.example/a.png?token=supersecret",
                    max_bytes=64,
                    resolver=resolver_for({"images.example": ["10.0.0.9"]}),
                )
            )
    assert "supersecret" not in caplog.text
    assert "images.example/a.png" in caplog.text
