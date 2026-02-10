"""Shared configuration for Oceanum MCP servers."""

import os
from dataclasses import dataclass


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
        storage_service=os.environ.get(
            "STORAGE_SERVICE", f"https://storage.{domain}"
        ),
    )
