"""Post-deploy smoke test for a hosted Datamesh MCP server (OCE-354).

tools/list and a 401 check pass even when the image's data stack is broken;
this opens real gridded datasources over zarr, so a zarr/xarray mismatch in
the deployed image (e.g. "open_group() got an unexpected keyword argument
'zarr_format'") fails it. There is no deploy pipeline: run it by hand after
every deploy_cloudrun*.sh:

    DATAMESH_TOKEN=... uv run python scripts/smoke_hosted.py [URL]

URL defaults to https://mcp.oceanum.io/datamesh; pass the service's run.app
URL + /datamesh for dev. The token is sent as X-DATAMESH-TOKEN and must be a
token for the gateway the server is deployed against. Datasources default to
the OCE-354 repro calls; see --help to override them.

Exit status:
    0  every check passed
    1  a check failed, or the server could not be reached or rejected the token
    2  usage error (e.g. DATAMESH_TOKEN unset)
    3  inconclusive: no failure, but a check could not exercise the zarr path
       (datasource not found or not shared, or the result was not lazy)
"""

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any

import httpx2
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from mcp.shared.exceptions import MCPError

DEFAULT_URL = "https://mcp.oceanum.io/datamesh"

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"

# The server's wording for a datasource that is missing or not shared with
# the caller: a bad smoke-test input, not a broken image.
_UNAVAILABLE_RE = re.compile(
    r"not found|not accessible|don't have access|not authorized", re.IGNORECASE
)

# Errors raised by the client for an unreachable server, a rejected token or
# a failed MCP handshake.
_CONNECT_ERRORS = (MCPError, RuntimeError, OSError, httpx2.HTTPError)


def build_checks(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    """The tool calls to make. Both must open their dataset lazily over zarr:
    query_data only does so when the staged result is above the server's
    inline limit (a full year of a regional grid is), otherwise it downloads
    the result whole and would pass even on a broken image."""
    return [
        ("load_datasource", {"datasource_id": args.load_datasource}),
        (
            "query_data",
            {
                "datasource_id": args.query_datasource,
                "variables": [args.query_variable],
                "time_start": "2020-01-01",
                "time_end": "2020-12-31",
            },
        ),
    ]


def check_result(is_error: bool, text: str) -> tuple[str, str]:
    """Classify a tool result as (PASS | FAIL | INCONCLUSIVE, detail)."""
    if is_error:
        return FAIL, text
    try:
        payload = json.loads(text)
    except ValueError:
        return FAIL, f"not JSON: {text[:200]}"
    if not isinstance(payload, dict):
        return FAIL, f"unexpected payload: {text[:200]}"
    if "error" in payload:
        error = str(payload["error"])
        if _UNAVAILABLE_RE.search(error):
            return INCONCLUSIVE, f"datasource unavailable: {error}"
        return FAIL, error
    if payload.get("container") != "dataset" or not payload.get("dims"):
        return FAIL, f"not a dataset summary: {text[:200]}"
    if payload.get("lazy") is not True:
        return INCONCLUSIVE, (
            "query did not reach the zarr path (result was downloaded, not "
            f"opened lazily; staged_size_bytes={payload.get('staged_size_bytes')})"
        )
    return PASS, str(payload["dims"])


async def run(url: str, token: str, checks: list[tuple[str, dict[str, Any]]]) -> int:
    transport = StreamableHttpTransport(url, headers={"X-DATAMESH-TOKEN": token})
    outcomes = []
    try:
        async with Client(transport, timeout=300) as client:
            for tool, args in checks:
                result = await client.call_tool(tool, args, raise_on_error=False)
                text = "".join(getattr(block, "text", "") for block in result.content)
                outcome, detail = check_result(result.is_error, text)
                outcomes.append(outcome)
                print(f"{outcome} {tool} {args['datasource_id']}: {detail}")
    except _CONNECT_ERRORS as exc:
        print(f"FAIL {url}: {type(exc).__name__}: {exc} (check the URL and token)")
        return 1
    if FAIL in outcomes:
        return 1
    if INCONCLUSIVE in outcomes:
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("url", nargs="?", default=DEFAULT_URL)
    parser.add_argument(
        "--load-datasource",
        default="oceanum_wave_nz_era5_v1_grid",
        help="gridded datasource for load_datasource (default: %(default)s)",
    )
    parser.add_argument(
        "--query-datasource",
        default="oceanum_wave_auckl_era5_grid",
        help="gridded datasource for query_data over 2020 (default: %(default)s)",
    )
    parser.add_argument(
        "--query-variable",
        default="hs",
        help="variable for query_data (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    token = os.environ.get("DATAMESH_TOKEN")
    if not token:
        print("Set DATAMESH_TOKEN to a Datamesh token.", file=sys.stderr)
        return 2
    return asyncio.run(run(args.url, token, build_checks(args)))


if __name__ == "__main__":
    sys.exit(main())
