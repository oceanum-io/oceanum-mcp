"""Oceanum Storage MCP server.

Exposes the Oceanum cloud storage filesystem as MCP tools for AI assistants.
"""

from __future__ import annotations

from fastmcp import FastMCP

from oceanum_mcp.common.client import get_storage_filesystem

mcp = FastMCP(
    "Oceanum Storage",
    instructions=(
        "Access the Oceanum cloud storage platform. "
        "Use list_files to browse, read_file/write_file for content, "
        "and delete_file to remove files."
    ),
)


# ---------------------------------------------------------------------------
# File Operations
# ---------------------------------------------------------------------------


@mcp.tool()
def list_files(
    path: str = "/",
    recursive: bool = False,
) -> str:
    """List files and directories in Oceanum cloud storage.

    Args:
        path: Directory path to list (default: root "/").
        recursive: Whether to list subdirectories recursively.

    Returns:
        List of files with name, size, and type.
    """
    fs = get_storage_filesystem()
    contents = fs.ls(path, detail=True, recursive=recursive)
    lines = []
    for item in contents:
        kind = "dir" if item["type"] == "directory" else "file"
        size = item.get("size", 0)
        lines.append(f"{kind}  {size:>10}  {item['name']}")
    return "\n".join(lines) if lines else f"Empty directory: {path}"


@mcp.tool()
def file_exists(path: str) -> str:
    """Check if a file or directory exists in Oceanum storage.

    Args:
        path: Path to check.

    Returns:
        Whether the path exists.
    """
    fs = get_storage_filesystem()
    result = fs.exists(path)
    return f"{'EXISTS' if result else 'NOT FOUND'}: {path}"


@mcp.tool()
def read_file(path: str) -> str:
    """Read the contents of a text file from Oceanum storage.

    Args:
        path: Path to the file.

    Returns:
        File contents as text. For large files, returns a size summary instead.
    """
    fs = get_storage_filesystem()
    info = fs.info(path)
    if info.get("size", 0) > 1_000_000:
        return (
            f"File too large to read inline ({info['size']} bytes). "
            "Use download instead."
        )
    with fs.open(path, "r") as f:
        return f.read()


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write text content to a file in Oceanum storage.

    Creates the file if it doesn't exist, overwrites if it does.

    Args:
        path: Destination path in Oceanum storage.
        content: Text content to write.

    Returns:
        Confirmation with bytes written.
    """
    fs = get_storage_filesystem()
    with fs.open(path, "w") as f:
        f.write(content)
    return f"Written {len(content)} bytes to {path}"


@mcp.tool()
def delete_file(path: str, recursive: bool = False) -> str:
    """Delete a file or directory from Oceanum storage.

    Args:
        path: Path to delete.
        recursive: For directories, delete contents recursively.

    Returns:
        Confirmation message.
    """
    fs = get_storage_filesystem()
    fs.rm(path, recursive=recursive)
    return f"Deleted: {path}"


@mcp.tool()
def file_info(path: str) -> str:
    """Get metadata about a file or directory in Oceanum storage.

    Args:
        path: Path to inspect.

    Returns:
        File type, size, and other metadata.
    """
    fs = get_storage_filesystem()
    info = fs.info(path)
    lines = [
        f"Path: {info['name']}",
        f"Type: {info.get('type', 'unknown')}",
        f"Size: {info.get('size', 0)} bytes",
    ]
    return "\n".join(lines)
