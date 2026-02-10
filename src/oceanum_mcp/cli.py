"""CLI argument parsing and server dispatch for Oceanum MCP."""

import argparse
import sys

SERVER_REGISTRY = {
    "datamesh": "oceanum_mcp.servers.datamesh.server",
    "storage": "oceanum_mcp.servers.storage.server",
    "combined": "oceanum_mcp.servers.combined.server",
}


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
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport to use (default: stdio).",
    )

    args = parser.parse_args()

    if args.list:
        print("Available servers:")
        for name in SERVER_REGISTRY:
            print(f"  - {name}")
        sys.exit(0)

    # Lazy import to only load the selected server's dependencies
    import importlib

    module = importlib.import_module(SERVER_REGISTRY[args.server])
    mcp_server = module.mcp

    mcp_server.run(transport=args.transport)
