"""Shared configuration for Oceanum MCP servers."""

import os
from pathlib import Path

# Default ceiling on bytes downloaded into the server process for inline
# results (query_data / load_datasource). Larger results must go through
# export_query or be narrowed with filters.
DEFAULT_MAX_INLINE_BYTES = 50_000_000

# Default number of rows/records previewed inline before a result is
# truncated. The byte ceiling above bounds total volume; this bounds how much
# of a within-budget tabular result is shown at once.
DEFAULT_MAX_INLINE_ROWS = 100

# Transport the current process was started with. Set by the CLI before the
# server modules are imported (they are imported lazily), so import-time
# decisions like disabling local-filesystem tools in http mode can key off it.
_transport = "stdio"


def set_transport(transport: str) -> None:
    """Record the transport the server is about to run with."""
    global _transport
    _transport = transport


def is_network_transport() -> bool:
    """Whether the server runs over a network transport (http/sse).

    Network transports imply a shared, multi-tenant server: credentials come
    from the request's auth context and tools must not touch the server-local
    filesystem.
    """
    return _transport in ("http", "sse")


def is_read_only() -> bool:
    """Whether write tools (update_metadata) should be disabled.

    Read at server start from OCEANUM_MCP_READ_ONLY; does not require a token.
    """
    return os.environ.get("OCEANUM_MCP_READ_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def max_inline_bytes() -> int:
    """Byte threshold above which results are not downloaded inline.

    Fails fast on an unparsable value — a silently ignored limit is worse
    than a loud misconfiguration.
    """
    raw = os.environ.get("OCEANUM_MCP_MAX_INLINE_BYTES")
    if not raw:
        return DEFAULT_MAX_INLINE_BYTES
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"OCEANUM_MCP_MAX_INLINE_BYTES must be an integer byte count, "
            f"got {raw!r}"
        ) from exc


def max_inline_rows() -> int:
    """Row/record count previewed inline before truncation.

    Fails fast on an unparsable value, like max_inline_bytes.
    """
    raw = os.environ.get("OCEANUM_MCP_MAX_INLINE_ROWS")
    if not raw:
        return DEFAULT_MAX_INLINE_ROWS
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"OCEANUM_MCP_MAX_INLINE_ROWS must be an integer row count, " f"got {raw!r}"
        ) from exc


def export_dir() -> Path | None:
    """Optional directory that export_query writes are confined to.

    Unset means exports may write to any path the process can reach.
    """
    raw = os.environ.get("OCEANUM_MCP_EXPORT_DIR")
    return Path(raw).expanduser().resolve() if raw else None


def datamesh_service() -> str:
    """Datamesh service URL from the environment (no token required)."""
    domain = os.environ.get("OCEANUM_DOMAIN", "oceanum.io")
    return os.environ.get("DATAMESH_SERVICE", f"https://datamesh.{domain}")


def storage_service() -> str:
    """Storage service URL from the environment (no token required)."""
    domain = os.environ.get("OCEANUM_DOMAIN", "oceanum.io")
    return os.environ.get("STORAGE_SERVICE", f"https://storage.{domain}")


def auth_mode() -> str:
    """Auth scheme for network transports, from OCEANUM_MCP_AUTH.

    - "auto" (default): accept either credential, detected per request — a
      JWT-shaped bearer is validated against the Auth0 tenant's JWKS, an
      opaque bearer against the Datamesh gateway.
    - "datamesh": the presented bearer must be a Datamesh token, validated
      against the gateway.
    - "auth0": the presented bearer must be an Auth0-issued JWT, validated
      against the tenant's JWKS and forwarded to the gateway as-is.
    - "none": no authentication — every request uses DATAMESH_TOKEN from the
      server's environment. Only for trusted-network deployments.
    """
    mode = os.environ.get("OCEANUM_MCP_AUTH", "auto").strip().lower()
    if mode not in ("auto", "datamesh", "auth0", "none"):
        raise ValueError(
            f"OCEANUM_MCP_AUTH must be one of auto, datamesh, auth0, none; "
            f"got {mode!r}"
        )
    return mode


def auth0_domain() -> str:
    return os.environ.get("OCEANUM_MCP_AUTH0_DOMAIN", "auth.oceanum.io")


def auth0_audience() -> str:
    return os.environ.get(
        "OCEANUM_MCP_AUTH0_AUDIENCE", "https://mcp.oceanum.io/datamesh"
    )


def public_url() -> str | None:
    """Externally visible base URL of this server, from OCEANUM_MCP_PUBLIC_URL.

    e.g. "https://mcp.oceanum.io". When set, network transports serve OAuth
    Protected Resource Metadata (RFC 9728) so OAuth clients — claude.ai
    custom connectors in particular — can discover the Auth0 authorization
    server. Unset means no discovery routes (header/bearer auth still works).
    """
    raw = os.environ.get("OCEANUM_MCP_PUBLIC_URL", "").strip().rstrip("/")
    return raw or None
