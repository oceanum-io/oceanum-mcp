"""Per-request API clients for Oceanum MCP servers.

Credential resolution order:
1. The authenticated request's access token (network transports) — the
   verifier stores a connector-ready credential in the token claims.
2. The DATAMESH_TOKEN environment variable (stdio, or OCEANUM_MCP_AUTH=none).

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
from typing import Any, Callable

from oceanum_mcp.common.config import datamesh_service, storage_service

# Claim key under which auth verifiers store the string to hand to the
# oceanum clients (a raw Datamesh token, or "Bearer <jwt>" for Auth0).
CREDENTIAL_CLAIM = "datamesh_credential"

_CACHE_MAX = 64
_CACHE_TTL_S = 900.0


class _ClientCache:
    """Bounded TTL cache keyed by (credential, service), thread-safe.

    FastMCP runs sync tools in worker threads, so concurrent requests from
    different users hit this cache concurrently.
    """

    def __init__(self, max_entries: int = _CACHE_MAX, ttl_s: float = _CACHE_TTL_S):
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, str], tuple[float, Any]] = OrderedDict()
        self._max = max_entries
        self._ttl = ttl_s

    def get_or_create(self, key: tuple[str, str], factory: Callable[[], Any]) -> Any:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and now - entry[0] < self._ttl:
                self._entries.move_to_end(key)
                return entry[1]
        # Build outside the lock: construction does network I/O and must not
        # serialize unrelated tenants. Two racing requests for the same new
        # credential may both build; last write wins, which is harmless.
        client = factory()
        with self._lock:
            self._entries[key] = (now, client)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)
        return client

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_datamesh_cache = _ClientCache()
_storage_cache = _ClientCache()


def resolve_credential() -> str:
    """Return the connector-ready credential for the current tool call."""
    from fastmcp.server.dependencies import get_access_token

    access = get_access_token()
    if access is not None:
        credential = access.claims.get(CREDENTIAL_CLAIM) or access.token
        if credential:
            return credential
    token = os.environ.get("DATAMESH_TOKEN")
    if token:
        return token
    raise ValueError(
        "No Datamesh credential available: the request is not authenticated "
        "and DATAMESH_TOKEN is not set in the server environment."
    )


def get_datamesh_connector():
    """Return a Datamesh Connector for the current request's credential."""
    credential = resolve_credential()
    service = datamesh_service()

    def build():
        from oceanum.datamesh import Connector

        return Connector(token=credential, service=service)

    return _datamesh_cache.get_or_create((credential, service), build)


def get_storage_filesystem():
    """Return a Storage FileSystem for the current request's credential."""
    credential = resolve_credential()
    service = storage_service()

    def build():
        from oceanum.storage import FileSystem

        return FileSystem(token=credential, service=service)

    return _storage_cache.get_or_create((credential, service), build)
