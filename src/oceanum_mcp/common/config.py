"""Shared configuration for Oceanum MCP servers."""

import os
from dataclasses import dataclass
from pathlib import Path

# Default ceiling on bytes downloaded into the server process for inline
# results (query_data / load_datasource). Larger results must go through
# export_query or be narrowed with filters.
DEFAULT_MAX_INLINE_BYTES = 50_000_000


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


def export_dir() -> Path | None:
    """Optional directory that export_query writes are confined to.

    Unset means exports may write to any path the process can reach.
    """
    raw = os.environ.get("OCEANUM_MCP_EXPORT_DIR")
    return Path(raw).expanduser().resolve() if raw else None


@dataclass
class OceanumConfig:
    token: str
    datamesh_service: str
    storage_service: str


def load_config() -> OceanumConfig:
    """Load Oceanum config from environment variables."""
    token = os.environ.get("DATAMESH_TOKEN")
    if not token:
        raise ValueError(
            "DATAMESH_TOKEN environment variable is required. "
            "Get a token from https://oceanum.io"
        )
    domain = os.environ.get("OCEANUM_DOMAIN", "oceanum.io")
    return OceanumConfig(
        token=token,
        datamesh_service=os.environ.get(
            "DATAMESH_SERVICE", f"https://datamesh.{domain}"
        ),
        storage_service=os.environ.get("STORAGE_SERVICE", f"https://storage.{domain}"),
    )
