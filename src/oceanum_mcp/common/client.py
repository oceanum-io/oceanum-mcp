"""Per-request API clients for Oceanum MCP servers.

Credential resolution order:
1. The authenticated request's access token (network transports) — the auth
   provider MUST store a connector-ready credential in the token claims.
2. The server's own token — but only for stdio, or for network transports
   explicitly configured with OCEANUM_MCP_AUTH=none. An unauthenticated
   network request must never silently escalate to the server's own
   credential. The token comes from DATAMESH_TOKEN, else the file named by
   DATAMESH_TOKEN_FILE, else (if DATAMESH_TOKEN_KEYRING is set) the OS
   credential store.

Clients are cached per credential: Connector construction performs a gateway
round trip, so rebuilding it on every tool call would double request latency.
The cache is bounded and entries expire so revoked tokens do not keep a live
client forever.
"""

from __future__ import annotations

import functools
import os
import stat
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from oceanum_mcp.common.config import (
    auth_mode,
    datamesh_service,
    env_flag,
    storage_service,
)

if TYPE_CHECKING:
    from oceanum.datamesh import Connector
    from oceanum.storage import FileSystem

# Claim key under which auth verifiers store the string to hand to the
# oceanum clients (a raw Datamesh token, or "Bearer <jwt>" for Auth0).
CREDENTIAL_CLAIM = "datamesh_credential"

# Shown when a credential is still a client config variable, e.g. VS Code's
# "${input:datamesh-token}" passed through verbatim (OCE-362).
UNRESOLVED_VARIABLE_MESSAGE = (
    "The token looks like an unresolved client variable (`${input:…}`): "
    "your MCP client did not substitute it."
)

# Where DATAMESH_TOKEN_KEYRING looks the token up in the OS credential store.
KEYRING_SERVICE = "oceanum-mcp"
KEYRING_USERNAME = "datamesh-token"

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
    token = _server_token()
    if token:
        return token
    raise ValueError(
        "No Datamesh credential available: the request is not authenticated "
        "and none of DATAMESH_TOKEN, DATAMESH_TOKEN_FILE or "
        "DATAMESH_TOKEN_KEYRING is set in the server environment. "
        "Get a token from https://oceanum.io"
    )


def is_unresolved_variable(value: str) -> bool:
    """Whether a credential is a client config variable left unsubstituted.

    Datamesh tokens and JWTs (base64url) never contain "$", while client
    variable syntaxes start with one: ${input:...}, ${env:...}, $VAR.
    """
    return value.startswith("$") or "${" in value


def _server_token() -> str | None:
    """The server's own token, from the first token source that is set."""
    token = os.environ.get("DATAMESH_TOKEN", "")
    if token.strip():
        return _checked_token(token, "DATAMESH_TOKEN")
    path = os.environ.get("DATAMESH_TOKEN_FILE")
    if path:
        return _read_token_file(path)
    use_keyring = env_flag("DATAMESH_TOKEN_KEYRING")
    if use_keyring is None:
        raise ValueError(
            "DATAMESH_TOKEN_KEYRING must be 1/true/yes or 0/false/no, got "
            f"{os.environ['DATAMESH_TOKEN_KEYRING']!r}"
        )
    if use_keyring:
        return _keyring_token()
    return None


def _read_token_file(raw: str) -> str:
    """Read the token from DATAMESH_TOKEN_FILE, which only its owner may access.

    Read on every call, so a rotated token takes effect without a restart.
    On POSIX a file with any group/other permission bits is refused, as ssh
    refuses a readable private key: the point of the file is that the token
    is not exposed in the client config or to other users.
    """
    if is_unresolved_variable(raw):
        raise ValueError(
            f"DATAMESH_TOKEN_FILE {raw!r} was not expanded: MCP clients pass "
            "env values literally. Use an absolute path, or one starting "
            "with ~."
        )
    path = Path(raw).expanduser()
    try:
        # utf-8-sig: Windows editors may prepend a byte-order mark.
        with path.open(encoding="utf-8-sig") as f:
            # fstat on the open file: no window to swap the file after a check.
            mode = os.fstat(f.fileno()).st_mode
            if os.name == "posix" and mode & 0o077:
                raise ValueError(
                    f"DATAMESH_TOKEN_FILE {path} is accessible by other users "
                    f"(mode {stat.S_IMODE(mode):04o}); restrict it with: "
                    f"chmod 600 {path}"
                )
            token = f.read()
    except OSError as exc:
        raise ValueError(
            f"DATAMESH_TOKEN_FILE {path} could not be read: "
            f"{exc.strerror or type(exc).__name__}"
        ) from None
    except UnicodeDecodeError:
        raise ValueError(f"DATAMESH_TOKEN_FILE {path} is not a text file") from None
    if not token.strip():
        raise ValueError(f"DATAMESH_TOKEN_FILE {path} is empty")
    return _checked_token(token, f"DATAMESH_TOKEN_FILE {path}")


def _checked_token(token: str, source: str) -> str:
    """A server token from any source, stripped, and refused if it is an
    unsubstituted variable such as ${input:...} or holds whitespace or
    control characters (e.g. a multi-line file)."""
    token = token.strip()
    if is_unresolved_variable(token):
        if source == "DATAMESH_TOKEN":
            raise ValueError(f"DATAMESH_TOKEN: {UNRESOLVED_VARIABLE_MESSAGE}")
        raise ValueError(
            f"The token in {source} looks like an unsubstituted variable "
            "(`${...}`), not a token: store the token itself."
        )
    if any(c.isspace() or not c.isprintable() for c in token):
        raise ValueError(
            f"The token in {source} contains whitespace or control "
            "characters: it must hold only the token, on one line."
        )
    return token


_KEYRING_LOCK = threading.Lock()


def _keyring_token() -> str:
    """The token stored in the OS credential store (macOS Keychain, Secret
    Service/libsecret, Windows Credential Manager) via the keyring package.

    Looked up once per process (a lookup can be a D-Bus or Keychain round
    trip, or an unlock prompt), so a changed token needs a server restart.
    Failures are not cached: storing a missing token takes effect at once.
    The lock makes concurrent first calls share one lookup (one prompt).
    """
    with _KEYRING_LOCK:
        return _keyring_lookup()


@functools.cache
def _keyring_lookup() -> str:
    try:
        import keyring
        from keyring.errors import KeyringError
    except ImportError:
        raise ValueError(
            "DATAMESH_TOKEN_KEYRING is set but the keyring package is not "
            "installed: install oceanum-mcp[keyring]."
        ) from None
    try:
        token = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    except KeyringError as exc:
        raise ValueError(
            f"DATAMESH_TOKEN_KEYRING is set but the OS credential store could "
            f"not be read: {exc}"
        ) from None
    except Exception as exc:  # noqa: BLE001 - backends raise their own types
        raise ValueError(
            "DATAMESH_TOKEN_KEYRING is set but the OS credential store "
            f"failed: {type(exc).__name__}: {exc}"
        ) from None
    if not token:
        raise ValueError(
            "DATAMESH_TOKEN_KEYRING is set but no token is stored: store it "
            f"with `keyring set {KEYRING_SERVICE} {KEYRING_USERNAME}`."
        )
    return _checked_token(token, "the OS credential store")


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
