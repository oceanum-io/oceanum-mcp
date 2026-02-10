"""Shared API client singletons for Oceanum MCP servers."""

from functools import lru_cache

from oceanum_mcp.common.config import load_config


@lru_cache(maxsize=1)
def get_datamesh_connector():
    """Return a configured Datamesh Connector singleton."""
    config = load_config()
    from oceanum.datamesh import Connector

    return Connector(token=config.token, service=config.datamesh_service)


@lru_cache(maxsize=1)
def get_storage_filesystem():
    """Return a configured Storage FileSystem singleton."""
    config = load_config()
    from oceanum.storage import FileSystem

    return FileSystem(token=config.token, service=config.storage_service)
