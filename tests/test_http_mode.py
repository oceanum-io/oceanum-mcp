"""HTTP-transport integration tests.

Covers the hosted-mode guarantees end to end at the ASGI layer:
- unauthenticated requests are rejected before any tool runs
- each request's tool call resolves that request's own credential
- export_query (server-local filesystem) is disabled in http mode
"""

import importlib
from contextlib import asynccontextmanager

import httpx
import pytest

from fastmcp import Client, FastMCP
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

import oceanum_mcp.servers.datamesh.server as datamesh_server
from oceanum_mcp.common.client import CREDENTIAL_CLAIM, resolve_credential
from oceanum_mcp.common.config import set_transport

INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}
CALL_WHOAMI = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {"name": "whoami", "arguments": {}},
}
HDRS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


@asynccontextmanager
async def http_client():
    """ASGI client for a minimal authed FastMCP app with a whoami tool.

    A context manager rather than an async fixture: the app lifespan holds an
    anyio cancel scope, which must enter and exit in the same task —
    pytest-asyncio runs async fixtures and tests in different tasks.
    """
    # StaticTokenVerifier exposes each token's config dict as the AccessToken
    # claims, so the credential claim is set the same way a real verifier does.
    verifier = StaticTokenVerifier(
        tokens={
            "tok-a": {"client_id": "a", CREDENTIAL_CLAIM: "tok-a"},
            "tok-b": {"client_id": "b", CREDENTIAL_CLAIM: "tok-b"},
        }
    )
    mcp = FastMCP("test-http", auth=verifier)

    @mcp.tool()
    def whoami() -> str:
        return resolve_credential()

    app = mcp.http_app(stateless_http=True)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield client


async def test_http_rejects_unauthenticated():
    async with http_client() as client:
        resp = await client.post("/mcp", json=INIT, headers=HDRS)
    assert resp.status_code == 401


async def test_http_rejects_invalid_token():
    async with http_client() as client:
        resp = await client.post(
            "/mcp", json=INIT, headers={**HDRS, "Authorization": "Bearer nope"}
        )
    assert resp.status_code == 401


async def test_http_accepts_valid_token():
    async with http_client() as client:
        resp = await client.post(
            "/mcp", json=INIT, headers={**HDRS, "Authorization": "Bearer tok-a"}
        )
    assert resp.status_code == 200


async def test_http_tool_call_uses_request_credential():
    """Two requests with different tokens must each see their own credential."""
    async with http_client() as client:
        for token in ("tok-a", "tok-b"):
            resp = await client.post(
                "/mcp",
                json=CALL_WHOAMI,
                headers={**HDRS, "Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200
            assert f'"result":"{token}"' in resp.text.replace(" ", "")


@pytest.fixture
def http_transport_datamesh():
    """Reload the datamesh server module under the http transport flag.

    importlib.reload re-executes the module in place, so references held by
    other test modules stay valid; the final reload restores stdio state.
    """
    set_transport("http")
    try:
        yield importlib.reload(datamesh_server)
    finally:
        set_transport("stdio")
        importlib.reload(datamesh_server)


async def test_export_query_disabled_in_http_mode(http_transport_datamesh):
    async with Client(http_transport_datamesh.mcp) as client:
        tools = {t.name for t in await client.list_tools()}
    assert "export_query" not in tools
    assert "query_data" in tools, "only the local-filesystem tool is disabled"


async def test_export_query_enabled_in_stdio_mode():
    async with Client(datamesh_server.mcp) as client:
        tools = {t.name for t in await client.list_tools()}
    assert "export_query" in tools


@pytest.fixture
def restore_datamesh_policy():
    """Undo create_http_app's mutations of the shared datamesh server."""
    yield
    set_transport("stdio")
    datamesh_server.mcp.enable(names={"export_query"})


async def test_create_http_app_applies_tool_policy(restore_datamesh_policy):
    """The ASGI factory disables export_query even when the server module was
    already imported (in stdio mode) before the factory ran, and mounts the
    endpoint at /<server> for multi-server ingress routing."""
    from oceanum_mcp.app import create_http_app

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OCEANUM_MCP_AUTH", "none")
        app = create_http_app("datamesh")
    assert "/datamesh" in [r.path for r in app.routes]
    async with Client(datamesh_server.mcp) as client:
        tools = {t.name for t in await client.list_tools()}
    assert "export_query" not in tools


def test_create_http_app_rejects_unknown_server():
    from oceanum_mcp.app import create_http_app

    with pytest.raises(ValueError, match="Unknown server"):
        create_http_app("nonexistent")
