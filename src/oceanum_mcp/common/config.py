"""Shared configuration for Oceanum MCP servers."""

import os
from dataclasses import dataclass

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
    """Byte threshold above which results are not downloaded inline."""
    try:
        return int(os.environ.get("OCEANUM_MCP_MAX_INLINE_BYTES", ""))
    except ValueError:
        return DEFAULT_MAX_INLINE_BYTES


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
