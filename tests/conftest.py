"""Test configuration — mock external dependencies that may not be installed."""

import sys
from unittest.mock import MagicMock

# Mock the `fastmcp` package so server modules can be imported without it installed
mock_fastmcp = MagicMock()


class _FakeFastMCP:
    """Minimal stand-in for fastmcp.FastMCP."""

    def __init__(self, name="test", **kwargs):
        self.name = name
        self._tools = {}

    def tool(self):
        """Decorator that registers a tool function and returns it unchanged."""

        def decorator(fn):
            self._tools[fn.__name__] = fn
            return fn

        return decorator

    def mount(self, other, prefix=None):
        pass

    def run(self, **kwargs):
        pass


mock_fastmcp.FastMCP = _FakeFastMCP
sys.modules.setdefault("fastmcp", mock_fastmcp)

# Mock the `oceanum` package
mock_oceanum = MagicMock()


# Real exception classes so they can be caught in except clauses
class _DatameshConnectError(Exception):
    pass


class _DatameshQueryError(Exception):
    pass


mock_exceptions = MagicMock()
mock_exceptions.DatameshConnectError = _DatameshConnectError
mock_exceptions.DatameshQueryError = _DatameshQueryError

sys.modules.setdefault("oceanum", mock_oceanum)
sys.modules.setdefault("oceanum.datamesh", mock_oceanum.datamesh)
sys.modules.setdefault("oceanum.datamesh.query", mock_oceanum.datamesh.query)
sys.modules.setdefault("oceanum.datamesh.exceptions", mock_exceptions)
sys.modules.setdefault("oceanum.storage", mock_oceanum.storage)
