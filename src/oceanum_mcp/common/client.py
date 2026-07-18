"""Per-request API clients for Oceanum MCP servers.

Credential resolution order:
1. The authenticated request's access token (network transports) — the auth
   provider MUST store a connector-ready credential in the token claims.
2. The DATAMESH_TOKEN environment variable — but only for stdio, or for
   network transports explicitly configured with OCEANUM_MCP_AUTH=none. An
   unauthenticated network request must never silently escalate to the
   server's own credential.

Clients are cached per credential: Connector construction performs a gateway
round trip, so rebuilding it on every tool call would double request latency.
The cache is bounded and entries expire so revoked tokens do not keep a live
client forever.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable

from oceanum_mcp.common.config import auth_mode, datamesh_service, storage_service

if TYPE_CHECKING:
    from oceanum.datamesh import Connector
    from oceanum.storage import FileSystem

# Claim key under which auth verifiers store the string to hand to the
# oceanum clients (a raw Datamesh token, or "Bearer <jwt>" for Auth0).
CREDENTIAL_CLAIM = "datamesh_credential"

_CACHE_MAX = 64
_CACHE_TTL_S = 900.0


class _ClientCache:
    """Bounded TTL cache keyed by (credential, service), thread-safe.

    FastMCP runs sync tools in worker threads, so concurrent requests from
    different users hit this cache concurrently. Evicted clients are dropped
    without an explicit close(): a tenant's in-flight request may still be
    using one, and the underlying HTTP sessions are reclaimed by GC.
    """

    def __init__(
        self, max_entries: int = _CACHE_MAX, ttl_s: float = _CACHE_TTL_S
    ) -> None:
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, str], tuple[float, Any]] = OrderedDict()
        self._max = max_entries
        self._ttl = ttl_s

    def get_or_create(self, key: tuple[str, str], factory: Callable[[], Any]) -> Any:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and time.monotonic() - entry[0] < self._ttl:
                self._entries.move_to_end(key)
                return entry[1]
        # Build outside the lock: construction does network I/O and must not
        # serialize unrelated tenants. Two racing requests for the same new
        # credential may both build; last write wins, which is harmless.
        client = factory()
        with self._lock:
            # Timestamp taken after the build so a slow construction does not
            # eat into the entry's lifetime.
            self._entries[key] = (time.monotonic(), client)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)
        return client

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_datamesh_cache = _ClientCache()
_storage_cache = _ClientCache()


def _serving_network_request() -> bool:
    """Whether this call executes inside an HTTP request."""
    from fastmcp.server.dependencies import get_http_request

    try:
        return get_http_request() is not None
    except RuntimeError:  # no request context (stdio)
        return False


def resolve_credential() -> str:
    """Return the connector-ready credential for the current tool call."""
    from fastmcp.server.dependencies import get_access_token

    access = get_access_token()
    if access is not None:
        credential = access.claims.get(CREDENTIAL_CLAIM)
        if credential:
            return credential
        # Fail loudly instead of guessing: a provider that authenticated the
        # request but deposited no credential is misconfigured, and falling
        # back to the raw bearer would send wrong-format credentials to the
        # gateway (e.g. a bare JWT in X-DATAMESH-TOKEN).
        raise ValueError(
            "Authenticated request carries no Datamesh credential: the auth "
            f"provider must store one in claims[{CREDENTIAL_CLAIM!r}]."
        )
    if _serving_network_request() and auth_mode() != "none":
        # No verified identity on a network request: refuse the environment
        # fallback rather than execute as the server's own identity. This
        # protects launch paths that bypass the CLI's auth wiring.
        raise ValueError(
            "Unauthenticated network request: refusing to fall back to the "
            "server's DATAMESH_TOKEN. Attach an auth provider (see "
            "oceanum_mcp.app.create_http_app) or set OCEANUM_MCP_AUTH=none "
            "explicitly for trusted-network deployments."
        )
    token = os.environ.get("DATAMESH_TOKEN")
    if token:
        return token
    raise ValueError(
        "No Datamesh credential available: the request is not authenticated "
        "and DATAMESH_TOKEN is not set in the server environment. "
        "Get a token from https://oceanum.io"
    )


def _get_client(
    cache: _ClientCache, service: str, build: Callable[[str, str], Any]
) -> Any:
    credential = resolve_credential()
    return cache.get_or_create(
        (credential, service), lambda: build(credential, service)
    )


def get_datamesh_connector() -> Connector:
    """Return a Datamesh Connector for the current request's credential."""

    def build(credential: str, service: str) -> Connector:
        from oceanum.datamesh import Connector

        return Connector(token=credential, service=service)

    return _get_client(_datamesh_cache, datamesh_service(), build)


def get_storage_filesystem() -> FileSystem:
    """Return a Storage FileSystem for the current request's credential."""

    def build(credential: str, service: str) -> FileSystem:
        from oceanum.storage import FileSystem

        return FileSystem(token=credential, service=service)

    return _get_client(_storage_cache, storage_service(), build)
