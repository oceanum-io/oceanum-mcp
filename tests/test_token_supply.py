"""Token supply (OCE-362).

- an unresolved client variable such as "${input:datamesh-token}" is named
  as such, on the local (stdio) and the hosted (http) server
- a rejected X-DATAMESH-TOKEN gets a 401 about the header token, without
  the OAuth challenge; credential-less 401s keep it for OAuth discovery
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import httpx2
import pytest
from fastmcp import Client
from fastmcp.server.auth import AccessToken

import oceanum_mcp.servers.datamesh.server as datamesh_server
from oceanum_mcp.common.auth import (
    GATEWAY_UNAVAILABLE,
    HEADER_TOKEN_REJECTED,
    DatameshTokenVerifier,
)
from oceanum_mcp.common.client import (
    CREDENTIAL_CLAIM,
    UNRESOLVED_VARIABLE_MESSAGE,
)
from oceanum_mcp.common.config import set_transport

PLACEHOLDER = "${input:datamesh-token}"

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
HDRS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


# --- local (stdio) -----------------------------------------------------------


async def test_local_placeholder_token_named_in_tool_error(monkeypatch):
    """The literal VS Code variable in DATAMESH_TOKEN surfaces as an
    explicit message from the tool, not the gateway's "Invalid token"."""
    monkeypatch.setenv("DATAMESH_TOKEN", PLACEHOLDER)
    with patch("oceanum.datamesh.Connector") as connector:
        async with Client(datamesh_server.mcp) as client:
            result = await client.call_tool(
                "search_catalog", {"search": "wave"}, raise_on_error=False
            )
    connector.assert_not_called()  # never sent to the gateway
    assert result.is_error
    assert UNRESOLVED_VARIABLE_MESSAGE in result.content[0].text


# --- hosted (http) -------------------------------------------------------------


@asynccontextmanager
async def hosted_client(gateway: AsyncMock, mode: str = "auto"):
    """The real create_http_app("datamesh") (auth ``mode``, OAuth discovery
    on); the gateway verification call is replaced by ``gateway``."""
    from oceanum_mcp.app import create_http_app

    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("OCEANUM_MCP_CORS_ORIGINS", raising=False)
        mp.setenv("OCEANUM_MCP_AUTH", mode)
        mp.setenv("OCEANUM_MCP_PUBLIC_URL", "https://mcp.example.test")
        mp.setattr(datamesh_server.mcp, "auth", None)
        mp.setattr(DatameshTokenVerifier, "_verify_with_gateway", gateway)
        app = create_http_app("datamesh")
        try:
            async with app.router.lifespan_context(app):
                transport = httpx2.ASGITransport(app=app)
                async with httpx2.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    yield client
        finally:
            set_transport("stdio")


def _rejecting_gateway() -> AsyncMock:
    return AsyncMock(return_value=None)


@pytest.mark.parametrize(
    "headers",
    [{"X-DATAMESH-TOKEN": PLACEHOLDER}, {"Authorization": f"Bearer {PLACEHOLDER}"}],
    ids=["header", "bearer"],
)
async def test_hosted_placeholder_token_named_in_401(headers):
    gateway = _rejecting_gateway()
    async with hosted_client(gateway) as client:
        resp = await client.post("/datamesh", json=INIT, headers={**HDRS, **headers})
    assert resp.status_code == 401
    assert resp.json() == {
        "error": "invalid_token",
        "error_description": UNRESOLVED_VARIABLE_MESSAGE,
    }
    challenge = resp.headers["www-authenticate"]
    assert 'error="invalid_token"' in challenge
    assert "unresolved client variable" in challenge
    assert "resource_metadata" not in challenge
    gateway.assert_not_called()


async def test_hosted_rejected_header_token_401_names_the_header():
    """A header token the gateway rejects gets a 401 about that token, with
    no resource_metadata that would send VS Code into a futile OAuth flow."""
    async with hosted_client(_rejecting_gateway()) as client:
        resp = await client.post(
            "/datamesh",
            json=INIT,
            headers={
                **HDRS,
                "X-DATAMESH-TOKEN": "a" * 40,
                "Origin": "https://vscode.dev",
            },
        )
    assert resp.status_code == 401
    body = resp.json()
    assert body == {
        "error": "invalid_token",
        "error_description": HEADER_TOKEN_REJECTED,
    }
    assert "clear authentication tokens" not in resp.text
    challenge = resp.headers["www-authenticate"]
    assert "X-DATAMESH-TOKEN" in challenge
    assert "resource_metadata" not in challenge
    assert int(resp.headers["content-length"]) == len(resp.content)
    # Still readable by browser clients: CORS wraps the rewritten response.
    assert resp.headers["access-control-allow-origin"] == "https://vscode.dev"


def _gateway_5xx() -> httpx2.HTTPStatusError:
    request = httpx2.Request("GET", "https://datamesh.example.test/user/")
    return httpx2.HTTPStatusError(
        "Datamesh gateway returned 502",
        request=request,
        response=httpx2.Response(502, request=request),
    )


@pytest.mark.parametrize(
    "failure",
    [lambda: httpx2.ConnectError("unreachable"), _gateway_5xx],
    ids=["transport-error", "gateway-5xx"],
)
async def test_hosted_header_token_unverified_on_gateway_outage_is_503(failure):
    """A gateway outage fails verification closed; the valid-looking token
    must not be reported as invalid, but as unverified, to retry."""
    async with hosted_client(AsyncMock(side_effect=failure())) as client:
        resp = await client.post(
            "/datamesh", json=INIT, headers={**HDRS, "X-DATAMESH-TOKEN": "a" * 40}
        )
    assert resp.status_code == 503
    assert resp.json() == {
        "error": "temporarily_unavailable",
        "error_description": GATEWAY_UNAVAILABLE,
    }
    assert "retry-after" in resp.headers
    assert "www-authenticate" not in resp.headers


async def test_auth_none_ignores_placeholder_client_credentials():
    """With OCEANUM_MCP_AUTH=none client credentials are ignored, so a
    placeholder in them must not get the request refused."""
    gateway = _rejecting_gateway()
    async with hosted_client(gateway, mode="none") as client:
        for headers in (
            {"X-DATAMESH-TOKEN": PLACEHOLDER},
            {"Authorization": f"Bearer {PLACEHOLDER}"},
        ):
            resp = await client.post(
                "/datamesh", json=INIT, headers={**HDRS, **headers}
            )
            assert resp.status_code == 200, headers
    gateway.assert_not_called()


async def test_placeholder_credential_does_not_block_public_routes():
    """Only a 401 is rewritten: routes that need no auth still answer."""
    async with hosted_client(_rejecting_gateway()) as client:
        resp = await client.get(
            "/.well-known/oauth-protected-resource/datamesh",
            headers={"X-DATAMESH-TOKEN": PLACEHOLDER},
        )
    assert resp.status_code == 200


async def test_unexplained_header_token_401_passes_through():
    """A 401 the Datamesh verifier did not explain (here: auth0 mode, where
    only JWTs are accepted) is left as FastMCP sent it."""
    async with hosted_client(_rejecting_gateway(), mode="auth0") as client:
        resp = await client.post(
            "/datamesh", json=INIT, headers={**HDRS, "X-DATAMESH-TOKEN": "a" * 40}
        )
    assert resp.status_code == 401
    assert "resource_metadata=" in resp.headers["www-authenticate"]
    assert HEADER_TOKEN_REJECTED not in resp.text


async def test_hosted_credentialless_401_keeps_oauth_challenge():
    """OAuth discovery (claude.ai) depends on the challenge when no
    credential at all was sent: that 401 is left untouched."""
    async with hosted_client(_rejecting_gateway()) as client:
        resp = await client.post("/datamesh", json=INIT, headers=HDRS)
    assert resp.status_code == 401
    challenge = resp.headers["www-authenticate"]
    assert "resource_metadata=" in challenge
    assert "error=" not in challenge


async def test_hosted_invalid_bearer_401_unchanged():
    """An invalid Authorization bearer (an OAuth client's token) keeps
    FastMCP's response, challenge included: re-authorizing can fix it."""
    async with hosted_client(_rejecting_gateway()) as client:
        resp = await client.post(
            "/datamesh", json=INIT, headers={**HDRS, "Authorization": "Bearer nope"}
        )
    assert resp.status_code == 401
    assert "resource_metadata=" in resp.headers["www-authenticate"]
    assert resp.json()["error_description"] != HEADER_TOKEN_REJECTED


async def test_hosted_valid_header_token_still_accepted():
    gateway = AsyncMock(
        return_value=AccessToken(
            token="t", client_id="u", scopes=[], claims={CREDENTIAL_CLAIM: "t"}
        )
    )
    async with hosted_client(gateway) as client:
        resp = await client.post(
            "/datamesh", json=INIT, headers={**HDRS, "X-DATAMESH-TOKEN": "a" * 40}
        )
    assert resp.status_code == 200
