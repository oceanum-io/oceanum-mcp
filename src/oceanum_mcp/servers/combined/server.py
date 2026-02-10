"""Oceanum combined MCP server.

Mounts all domain servers under a single FastMCP instance.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "Oceanum",
    instructions=(
        "Access the Oceanum platform for ocean/environmental data and cloud storage. "
        "Datamesh tools search, query, and manage datasets. "
        "Storage tools manage files in Oceanum cloud storage."
    ),
)


def _mount_servers():
    """Lazy mount to avoid circular imports."""
    from oceanum_mcp.servers.datamesh.server import mcp as datamesh
    from oceanum_mcp.servers.storage.server import mcp as storage

    mcp.mount("datamesh", datamesh)
    mcp.mount("storage", storage)


_mount_servers()
