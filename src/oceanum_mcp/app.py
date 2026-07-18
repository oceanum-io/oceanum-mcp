"""ASGI app factory for serving oceanum-mcp under an external ASGI server.

This is the only supported way to run the hosted server outside the
oceanum-mcp CLI (uvicorn/gunicorn workers, serverless platforms): it applies
the same safety rails as ``oceanum-mcp --transport http`` — auth provider
attachment and the network-transport tool policy. Serving ``mcp.http_app()``
directly bypasses both.

Usage:
    uvicorn --factory oceanum_mcp.app:create_http_app
"""

from __future__ import annotations

import importlib
from typing import Any

from starlette.applications import Starlette

from oceanum_mcp.cli import SERVER_REGISTRY
from oceanum_mcp.common.auth import build_auth_provider
from oceanum_mcp.common.config import set_transport


def create_http_app(
    server: str = "combined",
    *,
    stateless: bool = True,
    path: str | None = None,
    **http_app_kwargs: Any,
) -> Starlette:
    """Build a fully wired ASGI app for the named server.

    Auth comes from OCEANUM_MCP_AUTH exactly as in the CLI's http mode.
    Stateless by default: external ASGI servers usually mean multiple
    workers or instances, where in-memory MCP sessions do not survive
    request routing. The endpoint path defaults to /<server> (matching the
    CLI), so several servers can share one domain behind an ingress.
    """
    if server not in SERVER_REGISTRY:
        raise ValueError(
            f"Unknown server {server!r}; choose from {sorted(SERVER_REGISTRY)}"
        )
    set_transport("http")
    module = importlib.import_module(SERVER_REGISTRY[server])
    mcp = module.mcp
    if server in ("datamesh", "combined"):
        # Idempotent re-application of the network-transport tool policy,
        # covering the case where the server module was already imported
        # (in stdio mode) before this factory ran.
        from oceanum_mcp.servers.datamesh.server import mcp as datamesh_mcp

        datamesh_mcp.disable(names={"export_query"})
    provider = build_auth_provider()
    if provider is not None:
        mcp.auth = provider
    return mcp.http_app(
        stateless_http=stateless, path=path or f"/{server}", **http_app_kwargs
    )
