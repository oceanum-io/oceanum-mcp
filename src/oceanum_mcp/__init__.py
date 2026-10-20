"""Oceanum MCP — multi-server MCP package for the Oceanum platform."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("oceanum-mcp")
except PackageNotFoundError:  # an uninstalled source tree on sys.path
    __version__ = "0+unknown"
