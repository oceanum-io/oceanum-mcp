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
import re
from typing import Any

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware

from oceanum_mcp.cli import SERVER_REGISTRY
from oceanum_mcp.common.auth import DatameshHeaderMiddleware, build_auth_provider
from oceanum_mcp.common.config import cors_origins, set_transport

# What the MCP streamable-HTTP transport sends cross-origin: POST for
# messages, GET for the server event stream, DELETE to end a session.
CORS_ALLOW_METHODS = ["GET", "POST", "DELETE", "OPTIONS"]
CORS_ALLOW_HEADERS = [
    "Authorization",
    "Content-Type",
    "X-DATAMESH-TOKEN",
    "Mcp-Session-Id",
    "MCP-Protocol-Version",
    "Last-Event-ID",
]
# Mcp-Session-Id must be readable for stateful sessions; WWW-Authenticate
# lets a browser OAuth client find the resource metadata from a 401.
CORS_EXPOSE_HEADERS = ["Mcp-Session-Id", "WWW-Authenticate"]
CORS_MAX_AGE = 3600

# One DNS label: what a leading "*." in an allowlist entry may stand for.
# No dots, so "*" never spans several labels.
_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"


def cors_origin_regex(origins: list[str]) -> str | None:
    """Anchored regex matching exactly the given (validated) origins.

    Literal parts are escaped, so dots only match dots; "*." becomes a single
    label. Matched with fullmatch by Starlette, and \\Z-anchored as well so
    no suffix (``https://app.oceanum.io.evil.com``) can slip through.
    """
    if not origins:
        return None
    parts = []
    for origin in origins:
        scheme, _, host = origin.partition("://")
        if host.startswith("*."):
            parts.append(re.escape(f"{scheme}://") + _LABEL + re.escape(host[1:]))
        else:
            parts.append(re.escape(origin))
    return rf"(?:{'|'.join(parts)})\Z"


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
    # No per-tool policy to re-apply for network transports: export_query is
    # enabled on every transport (it brokers a download URL on hosted, writes a
    # local file on stdio — the branch lives in the tool itself).
    provider = build_auth_provider()
    if provider is not None:
        mcp.auth = provider
    app = mcp.http_app(
        stateless_http=stateless, path=path or f"/{server}", **http_app_kwargs
    )
    # add_middleware inserts OUTERMOST — required: fastmcp places middleware
    # passed to http_app() inside its auth middleware, where the header
    # promotion would run only after authentication already failed.
    app.add_middleware(DatameshHeaderMiddleware)
    # Added last so it is outermost of all: a browser preflight carries no
    # credential, so it must be answered before auth (or the header
    # promotion) ever sees it. Credentials travel in headers, never cookies,
    # hence allow_credentials=False.
    origin_regex = cors_origin_regex(cors_origins())
    if origin_regex is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origin_regex=origin_regex,
            allow_methods=CORS_ALLOW_METHODS,
            allow_headers=CORS_ALLOW_HEADERS,
            expose_headers=CORS_EXPOSE_HEADERS,
            allow_credentials=False,
            max_age=CORS_MAX_AGE,
        )
    return app
