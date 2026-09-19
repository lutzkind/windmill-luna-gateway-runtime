"""Shared outbound URL validation for caller-supplied destinations.

Caller-supplied URLs must pass a scheme allowlist, reject embedded credentials,
normalize the hostname, resolve DNS, reject non-public addresses (loopback,
RFC1918, link-local, reserved, CGNAT, IPv6 unique-local/site-local, IPv4-mapped
IPv6), pin the connection to the validated address, revalidate every redirect
hop with a bounded count, enforce a timeout and a response size cap, and log
only redacted targets.
"""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Sequence
from urllib.parse import urljoin, urlsplit

import httpx

LOGGER = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_REDIRECTS = 5
MAX_REDIRECTS = 10
MAX_HOSTNAME_LENGTH = 253
MAX_HOST_LABEL_LENGTH = 63
DEFAULT_PORTS = {"http": 80, "https": 443}
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
BLOCKED_HOSTNAME_SUFFIXES = (".localhost", ".local", ".internal", ".home.arpa")
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
_HOST_LABEL = re.compile(r"[a-z0-9-]+")

Resolver = Callable[[str, int], "Sequence[str] | Awaitable[Sequence[str]]"]
TransportFactory = Callable[["OutboundTarget"], httpx.AsyncBaseTransport]


class OutboundURLRejected(Exception):
    """Raised when a caller-supplied outbound URL is not allowed."""

    def __init__(self, code: str, detail: str, *, status_code: int = 400) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.status_code = status_code
        self.redacted_target = "<unknown-url>"


@dataclass(frozen=True)
class OutboundTarget:
    """A validated outbound destination with its pinned address set."""

    url: str
    scheme: str
    hostname: str
    port: int
    addresses: tuple[str, ...]
    host_header: str


@dataclass(frozen=True)
class OutboundResponse:
    """A bounded outbound response; ``url`` is always redacted."""

    status_code: int
    headers: dict[str, str]
    content: bytes
    url: str
    redirects: int = 0


def redact_url(url: str) -> str:
    """Return a log-safe form of ``url`` without credentials, query, or fragment."""
    raw = _CONTROL_CHARACTERS.sub("", str(url or "")).strip()
    if not raw:
        return "<empty-url>"
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError:
        return "<invalid-url>"
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port is not None else host
    path = parsed.path if parsed.path.startswith("/") else f"/{parsed.path}"
    if len(path) > 80:
        path = path[:77] + "..."
    if parsed.scheme:
        return f"{parsed.scheme.lower()}://{netloc}{path}"
    return path or "<invalid-url>"


def _ip_literal(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(value)
    except ValueError:
        return None


def normalize_hostname(host: str) -> str:
    """Normalize a hostname (lowercase, IDNA, no trailing dot) or reject it."""
    candidate = _CONTROL_CHARACTERS.sub("", str(host or "")).strip().rstrip(".")
    if not candidate:
        raise OutboundURLRejected("invalid_hostname", "outbound URL must include a hostname")
    literal = _ip_literal(candidate)
    if literal is not None:
        return str(literal)
    try:
        ascii_host = candidate.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise OutboundURLRejected("invalid_hostname", "outbound hostname is not valid") from exc
    if len(ascii_host) > MAX_HOSTNAME_LENGTH:
        raise OutboundURLRejected("invalid_hostname", "outbound hostname is too long")
    for label in ascii_host.split("."):
        if (
            not label
            or len(label) > MAX_HOST_LABEL_LENGTH
            or label.startswith("-")
            or label.endswith("-")
            or _HOST_LABEL.fullmatch(label) is None
        ):
            raise OutboundURLRejected("invalid_hostname", "outbound hostname is not valid")
    return ascii_host


def address_rejection_reason(value: str) -> str | None:
    """Return why an address is not publicly routable, or ``None`` if it is."""
    try:
        address = ipaddress.ip_address(str(value or "").strip())
    except ValueError:
        return "not a valid IP address"
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address_rejection_reason(str(address.ipv4_mapped))
    if address.is_unspecified:
        return "unspecified address"
    if address.is_loopback:
        return "loopback address"
    if address.is_link_local:
        return "link-local address"
    if address.is_multicast:
        return "multicast address"
    if address.is_reserved:
        return "reserved address"
    if address.is_private:
        return "private address"
    if isinstance(address, ipaddress.IPv6Address) and address.is_site_local:
        return "site-local address"
    if isinstance(address, ipaddress.IPv4Address) and address in _CGNAT_NETWORK:
        return "carrier-grade NAT address"
    if not address.is_global:
        return "non-global address"
    return None


def is_public_address(value: str) -> bool:
    return address_rejection_reason(value) is None


def _canonical_ip(value: str) -> str:
    literal = _ip_literal(str(value or "").strip())
    return str(literal) if literal is not None else str(value or "").strip().lower()


async def _system_resolve(hostname: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(
            hostname, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
        )
    except (socket.gaierror, OSError) as exc:
        raise OutboundURLRejected("dns_failure", "outbound hostname did not resolve") from exc
    addresses: list[str] = []
    for info in infos:
        if len(info) > 4 and info[4]:
            address = str(info[4][0])
            if address not in addresses:
                addresses.append(address)
    if not addresses:
        raise OutboundURLRejected("dns_failure", "outbound hostname did not resolve")
    return addresses


async def _resolve_host(hostname: str, port: int, resolver: Resolver | None) -> list[str]:
    literal = _ip_literal(hostname)
    if literal is not None:
        return [str(literal)]
    if resolver is None:
        return await _system_resolve(hostname, port)
    try:
        result = resolver(hostname, port)
        if inspect.isawaitable(result):
            result = await result
        addresses = [str(item) for item in result]
    except OutboundURLRejected:
        raise
    except Exception as exc:
        raise OutboundURLRejected("dns_failure", "outbound hostname did not resolve") from exc
    if not addresses:
        raise OutboundURLRejected("dns_failure", "outbound hostname did not resolve")
    return addresses


async def validate_outbound_url(
    url: str,
    *,
    resolver: Resolver | None = None,
    allowed_schemes: Sequence[str] = ("http", "https"),
) -> OutboundTarget:
    """Validate one destination and return its pinned, normalized target."""
    raw = _CONTROL_CHARACTERS.sub("", str(url or "")).strip()
    if not raw:
        raise OutboundURLRejected("missing_url", "outbound URL is required")
    if any(character.isspace() for character in raw):
        raise OutboundURLRejected("invalid_url", "outbound URL cannot contain whitespace")
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise OutboundURLRejected("invalid_url", "outbound URL is not valid") from exc
    scheme = (parsed.scheme or "").lower()
    if scheme not in {candidate.lower() for candidate in allowed_schemes}:
        raise OutboundURLRejected("unsupported_scheme", "outbound URL scheme is not allowed")
    if parsed.username is not None or parsed.password is not None:
        raise OutboundURLRejected("url_credentials", "outbound URL cannot include credentials")
    hostname = normalize_hostname(parsed.hostname or "")
    if hostname == "localhost" or hostname.endswith(BLOCKED_HOSTNAME_SUFFIXES):
        raise OutboundURLRejected("hostname_not_allowed", "outbound hostname is not allowed")
    effective_port = port if port is not None else DEFAULT_PORTS[scheme]
    if effective_port < 1 or effective_port > 65535:
        raise OutboundURLRejected("invalid_url", "outbound URL has an invalid port")
    addresses = await _resolve_host(hostname, effective_port, resolver)
    for address in addresses:
        reason = address_rejection_reason(address)
        if reason is not None:
            raise OutboundURLRejected("address_not_public", "outbound URL resolves to a non-public address")
    host_header = f"[{hostname}]" if ":" in hostname else hostname
    if effective_port != DEFAULT_PORTS[scheme]:
        host_header = f"{host_header}:{effective_port}"
    return OutboundTarget(
        url=raw,
        scheme=scheme,
        hostname=hostname,
        port=effective_port,
        addresses=tuple(dict.fromkeys(addresses)),
        host_header=host_header,
    )


def pinned_url(target: OutboundTarget, address: str | None = None) -> str:
    """Build the request URL that connects to a validated address only."""
    selected = str(address or target.addresses[0])
    host = f"[{selected}]" if ":" in selected else selected
    if target.port != DEFAULT_PORTS[target.scheme]:
        host = f"{host}:{target.port}"
    parsed = urlsplit(target.url)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return f"{target.scheme}://{host}{path}"


def verify_connected_peer(extensions: Mapping[str, Any], target: OutboundTarget) -> None:
    """Fail when a connected transport reports a peer outside the pinned set."""
    stream = extensions.get("network_stream") if extensions else None
    if stream is None:
        return
    try:
        peer = stream.get_extra_info("server_addr")
    except Exception as exc:
        raise OutboundURLRejected("peer_address_mismatch", "outbound connection peer address is not allowed") from exc
    peer_host = ""
    if isinstance(peer, (tuple, list)) and peer:
        peer_host = str(peer[0])
    elif peer:
        peer_host = str(peer)
    allowed = {_canonical_ip(address) for address in target.addresses}
    if not peer_host or _canonical_ip(peer_host) not in allowed:
        raise OutboundURLRejected("peer_address_mismatch", "outbound connection peer address is not allowed")


def _pinned_transport(target: OutboundTarget) -> httpx.AsyncBaseTransport:
    del target
    return httpx.AsyncHTTPTransport(retries=0, trust_env=False)


async def fetch_outbound(
    source: str,
    *,
    max_bytes: int,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    resolver: Resolver | None = None,
    transport_factory: TransportFactory | None = None,
) -> OutboundResponse:
    """Fetch ``source`` through the shared outbound policy with bounded reads."""
    limit = int(max_bytes)
    if limit <= 0:
        raise OutboundURLRejected("invalid_limit", "outbound response limit must be positive")
    timeout = float(timeout_seconds)
    if timeout <= 0:
        raise OutboundURLRejected("invalid_timeout", "outbound timeout must be positive")
    redirect_budget = max(0, min(MAX_REDIRECTS, int(max_redirects)))
    logical_url = _CONTROL_CHARACTERS.sub("", str(source or "")).strip()
    redirects = 0
    try:
        for hop in range(redirect_budget + 1):
            target = await validate_outbound_url(logical_url, resolver=resolver)
            transport = (transport_factory or _pinned_transport)(target)
            headers = {"Host": target.host_header, "Accept": "*/*"}
            extensions = {"sni_hostname": target.hostname} if target.scheme == "https" else {}
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout),
                transport=transport,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                async with client.stream(
                    "GET",
                    pinned_url(target),
                    headers=headers,
                    extensions=extensions,
                ) as response:
                    verify_connected_peer(response.extensions, target)
                    if response.status_code in REDIRECT_STATUSES:
                        location = (response.headers.get("location") or "").strip()
                        if not location:
                            raise OutboundURLRejected(
                                "redirect_without_location",
                                "outbound redirect had no location",
                            )
                        if hop >= redirect_budget:
                            raise OutboundURLRejected(
                                "too_many_redirects",
                                "outbound redirect limit exceeded",
                            )
                        redirects += 1
                        logical_url = urljoin(target.url, location)
                        LOGGER.info(
                            "outbound_url_redirect hop=%d target=%s",
                            redirects,
                            redact_url(logical_url),
                        )
                        continue
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > limit:
                            raise OutboundURLRejected(
                                "response_too_large",
                                "outbound response exceeded the size limit",
                                status_code=413,
                            )
                        chunks.append(chunk)
                    return OutboundResponse(
                        status_code=response.status_code,
                        headers={key.lower(): value for key, value in response.headers.items()},
                        content=b"".join(chunks),
                        url=redact_url(target.url),
                        redirects=redirects,
                    )
        raise OutboundURLRejected("too_many_redirects", "outbound redirect limit exceeded")
    except OutboundURLRejected as exc:
        exc.redacted_target = redact_url(logical_url)
        LOGGER.warning(
            "outbound_url_rejected code=%s target=%s", exc.code, exc.redacted_target
        )
        raise
    except httpx.TimeoutException as exc:
        rejected = OutboundURLRejected("fetch_timeout", "outbound request timed out")
        rejected.redacted_target = redact_url(logical_url)
        LOGGER.warning(
            "outbound_url_rejected code=%s target=%s", rejected.code, rejected.redacted_target
        )
        raise rejected from exc
    except httpx.HTTPError as exc:
        rejected = OutboundURLRejected("fetch_failed", "outbound request failed")
        rejected.redacted_target = redact_url(logical_url)
        LOGGER.warning(
            "outbound_url_rejected code=%s target=%s", rejected.code, rejected.redacted_target
        )
        raise rejected from exc
