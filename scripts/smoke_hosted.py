"""Post-deploy smoke test for a hosted Datamesh MCP server (OCE-354).

tools/list and a 401 check pass even when the image's data stack is broken;
this opens real gridded datasources, so a zarr/xarray mismatch in the deployed
image (e.g. "open_group() got an unexpected keyword argument 'zarr_format'")
fails it. Run it after every deploy:

    DATAMESH_TOKEN=... uv run python scripts/smoke_hosted.py [URL]

URL defaults to https://mcp.oceanum.io/datamesh; pass the service's run.app
URL + /datamesh for dev. The token is sent as X-DATAMESH-TOKEN and must be a
token for the gateway the server is deployed against. Exits non-zero if any
check fails.
"""

import asyncio
import json
import os
import sys
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

DEFAULT_URL = "https://mcp.oceanum.io/datamesh"

# The OCE-354 repro calls. Each must come back as a dataset summary. Both are
# opened lazily over zarr (no values downloaded); query_data must stay above
# the inline limit to take that path, as a small query is fetched whole and
# passes even on a broken image.
CHECKS: list[tuple[str, dict[str, Any]]] = [
    ("load_datasource", {"datasource_id": "oceanum_wave_nz_era5_v1_grid"}),
    (
        "query_data",
        {
            "datasource_id": "oceanum_wave_auckl_era5_grid",
            "variables": ["hs"],
            "time_start": "2020-01-01",
            "time_end": "2020-12-31",
        },
    ),
]


def check_result(is_error: bool, text: str) -> str | None:
    """Return a failure reason for a tool result, or None if it passed."""
    if is_error:
        return text
    try:
        payload = json.loads(text)
    except ValueError:
        return f"not JSON: {text[:200]}"
    if not isinstance(payload, dict):
        return f"unexpected payload: {text[:200]}"
    if "error" in payload:
        return str(payload["error"])
    if payload.get("container") != "dataset" or not payload.get("dims"):
        return f"not a dataset summary: {text[:200]}"
    return None


async def run(url: str, token: str) -> int:
    transport = StreamableHttpTransport(url, headers={"X-DATAMESH-TOKEN": token})
    failures = 0
    async with Client(transport, timeout=300) as client:
        for tool, args in CHECKS:
            result = await client.call_tool(tool, args, raise_on_error=False)
            text = "".join(getattr(block, "text", "") for block in result.content)
            reason = check_result(result.is_error, text)
            if reason is None:
                print(
                    f"PASS {tool} {args['datasource_id']}: {json.loads(text)['dims']}"
                )
            else:
                failures += 1
                print(f"FAIL {tool} {args['datasource_id']}: {reason}")
    return 1 if failures else 0


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    token = os.environ.get("DATAMESH_TOKEN")
    if not token:
        print("Set DATAMESH_TOKEN to a Datamesh token.", file=sys.stderr)
        return 2
    return asyncio.run(run(url, token))


if __name__ == "__main__":
    sys.exit(main())
