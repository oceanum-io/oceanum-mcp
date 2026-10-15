"""CLI argument parsing and server dispatch for Oceanum MCP."""

import argparse
import sys
from typing import Any, TextIO

SERVER_REGISTRY = {
    "datamesh": "oceanum_mcp.servers.datamesh.server",
    "storage": "oceanum_mcp.servers.storage.server",
    "combined": "oceanum_mcp.servers.combined.server",
}


class _StdoutGuard:
    """sys.stdout stand-in while serving the stdio transport.

    The MCP stdio transport writes JSON-RPC to sys.stdout.buffer, so any text
    printed to stdout would corrupt the protocol stream. The oceanum library
    print()s to stdout (e.g. its "new version available" notice on first
    connect), so text writes go to stderr while .buffer stays the real stdout
    for the transport.
    """

    def __init__(self, stdout: TextIO, stderr: TextIO) -> None:
        self.buffer = stdout.buffer
        self._stderr = stderr

    def write(self, s: str) -> int:
        return self._stderr.write(s)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stderr, name)


def _run_stdio(mcp_server: Any) -> None:
    """Serve over stdio with nothing but JSON-RPC on stdout.

    No banner: stdio clients surface stderr output as warnings, and the
    banner also triggers a PyPI update check on every launch.
    """
    real_stdout = sys.stdout
    sys.stdout = _StdoutGuard(real_stdout, sys.stderr)
    try:
        mcp_server.run(transport="stdio", show_banner=False)
    finally:
        sys.stdout = real_stdout


def main():
    parser = argparse.ArgumentParser(
        prog="oceanum-mcp",
        description="Run an Oceanum MCP server.",
    )
    parser.add_argument(
        "server",
        choices=list(SERVER_REGISTRY.keys()),
        nargs="?",
        default="combined",
        help="Which MCP server to run (default: combined)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available servers and exit.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "sse"],
        default="stdio",
        help=(
            "MCP transport to use (default: stdio). http runs a hosted "
            "streamable-HTTP server; auth is configured via OCEANUM_MCP_AUTH "
            "(datamesh, auth0, or none)."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address for http/sse transports (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port for http/sse transports (default: 8000).",
    )
    parser.add_argument(
        "--path",
        default=None,
        help=(
            "URL path of the MCP endpoint for http/sse transports. Defaults "
            "to /<server> (e.g. /datamesh), so multiple servers can share one "
            "domain behind an ingress (https://mcp.oceanum.io/datamesh)."
        ),
    )
    parser.add_argument(
        "--stateless",
        action="store_true",
        help=(
            "Run the http transport without server-side sessions, so any "
            "instance can serve any request. Required behind load balancers "
            "and autoscaled platforms (e.g. Cloud Run) where consecutive "
            "requests may hit different instances."
        ),
    )

    args = parser.parse_args()

    if args.list:
        print("Available servers:")
        for name in SERVER_REGISTRY:
            print(f"  - {name}")
        sys.exit(0)

    # Record the transport BEFORE the lazy server import: server modules make
    # import-time decisions (e.g. disabling local-filesystem tools) off it.
    from oceanum_mcp.common.config import set_transport

    set_transport(args.transport)

    # Lazy import to only load the selected server's dependencies
    import importlib

    module = importlib.import_module(SERVER_REGISTRY[args.server])
    mcp_server = module.mcp

    if args.transport in ("http", "sse"):
        from oceanum_mcp.common.config import auth_mode

        if auth_mode() == "none":
            print(
                "WARNING: OCEANUM_MCP_AUTH=none — the server is UNAUTHENTICATED "
                "and every request uses the server's DATAMESH_TOKEN.",
                file=sys.stderr,
            )

    if args.transport == "http":
        # One app-construction path for hosted mode: the factory owns auth
        # attachment, tool policy, and the X-DATAMESH-TOKEN header promotion.
        import uvicorn

        from oceanum_mcp.app import create_http_app

        app = create_http_app(
            args.server,
            stateless=args.stateless,
            path=args.path or f"/{args.server}",
        )
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.transport == "sse":
        # Deprecated legacy transport: bearer auth only (no X-DATAMESH-TOKEN
        # header support).
        from oceanum_mcp.common.auth import build_auth_provider

        provider = build_auth_provider()
        if provider is not None:
            mcp_server.auth = provider
        mcp_server.run(
            transport="sse",
            host=args.host,
            port=args.port,
            path=args.path or f"/{args.server}",
        )
    else:
        _run_stdio(mcp_server)
