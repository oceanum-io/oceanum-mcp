"""Shared exception types for Oceanum MCP servers."""


class OceanumMCPError(Exception):
    """Base exception for tool execution errors. Message is returned to the LLM."""

    pass


class AuthenticationError(OceanumMCPError):
    pass


class ResourceNotFoundError(OceanumMCPError):
    pass
