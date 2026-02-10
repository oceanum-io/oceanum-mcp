"""Oceanum combined MCP server.

Mounts all domain servers under a single FastMCP instance.
"""

from fastmcp import FastMCP

from oceanum_mcp.servers.datamesh.server import mcp as datamesh
from oceanum_mcp.servers.storage.server import mcp as storage

mcp = FastMCP(
    "Oceanum",
    instructions=(
        "Access the Oceanum platform for ocean/environmental data and cloud storage. "
        "Datamesh tools search, query, and manage datasets. "
        "Storage tools manage files in Oceanum cloud storage."
    ),
)

mcp.mount(datamesh, prefix="datamesh")
mcp.mount(storage, prefix="storage")
