"""Oceanum MCP server registry."""

from oceanum_mcp.cli import SERVER_REGISTRY


def get_server(name: str):
    """Import and return a server's FastMCP instance by name."""
    import importlib

    module = importlib.import_module(SERVER_REGISTRY[name])
    return module.mcp
