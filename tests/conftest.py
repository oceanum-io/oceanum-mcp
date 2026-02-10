"""Test configuration — mock external dependencies that may not be installed."""

import sys
from unittest.mock import MagicMock

# Mock the `mcp` package so server modules can be imported without it installed
mock_mcp = MagicMock()


class _FakeFastMCP:
    """Minimal stand-in for mcp.server.fastmcp.FastMCP."""

    def __init__(self, name="test", **kwargs):
        self.name = name
        self._tools = {}

    def tool(self):
        """Decorator that registers a tool function and returns it unchanged."""

        def decorator(fn):
            self._tools[fn.__name__] = fn
            return fn

        return decorator

    def mount(self, prefix, other):
        pass

    def run(self, **kwargs):
        pass


mock_mcp.server.fastmcp.FastMCP = _FakeFastMCP
sys.modules.setdefault("mcp", mock_mcp)
sys.modules.setdefault("mcp.server", mock_mcp.server)
sys.modules.setdefault("mcp.server.fastmcp", mock_mcp.server.fastmcp)

# Mock the `oceanum` package
mock_oceanum = MagicMock()
sys.modules.setdefault("oceanum", mock_oceanum)
sys.modules.setdefault("oceanum.datamesh", mock_oceanum.datamesh)
sys.modules.setdefault("oceanum.datamesh.query", mock_oceanum.datamesh.query)
sys.modules.setdefault("oceanum.storage", mock_oceanum.storage)
