"""HTTP-transport integration tests.

Covers the hosted-mode guarantees end to end at the ASGI layer:
- unauthenticated requests are rejected before any tool runs
- each request's tool call resolves that request's own credential
- export_query (server-local filesystem) is disabled in http mode
- CORS preflights are answered ahead of auth, for allowlisted origins only
"""

import importlib
from contextlib import asynccontextmanager

import httpx
import pytest

from fastmcp import Client, FastMCP
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

from oceanum_mcp.common.auth import DatameshHeaderMiddleware

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

    # add_middleware (outermost) mirrors create_http_app: fastmcp's own
    # middleware kwarg would place the promotion inside auth, too late.
    app = mcp.http_app(stateless_http=True)
    app.add_middleware(DatameshHeaderMiddleware)
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


async def test_export_query_enabled_in_http_mode(http_transport_datamesh):
    # export_query stays enabled on hosted — it brokers a download URL there
    # instead of writing a local file.
    async with Client(http_transport_datamesh.mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
    assert "export_query" in tools
    assert "download" in (tools["export_query"].description or "").lower()


async def test_export_query_enabled_in_stdio_mode():
    async with Client(datamesh_server.mcp) as client:
        tools = {t.name for t in await client.list_tools()}
    assert "export_query" in tools


async def test_http_accepts_x_datamesh_token_header():
    """A Datamesh token in its conventional X-DATAMESH-TOKEN header
    authenticates without an Authorization header."""
    async with http_client() as client:
        resp = await client.post(
            "/mcp",
            json=CALL_WHOAMI,
            headers={**HDRS, "X-DATAMESH-TOKEN": "tok-a"},
        )
        assert resp.status_code == 200
        assert '"result":"tok-a"' in resp.text.replace(" ", "")


async def test_http_authorization_wins_over_datamesh_header():
    """When both headers are sent, the Authorization bearer is authoritative
    and the X-DATAMESH-TOKEN header is not promoted."""
    async with http_client() as client:
        resp = await client.post(
            "/mcp",
            json=CALL_WHOAMI,
            headers={
                **HDRS,
                "Authorization": "Bearer tok-a",
                "X-DATAMESH-TOKEN": "tok-b",
            },
        )
        assert resp.status_code == 200
        assert '"result":"tok-a"' in resp.text.replace(" ", "")


async def test_http_invalid_x_datamesh_token_rejected():
    async with http_client() as client:
        resp = await client.post(
            "/mcp", json=INIT, headers={**HDRS, "X-DATAMESH-TOKEN": "nope"}
        )
    assert resp.status_code == 401


@pytest.fixture
def restore_datamesh_policy():
    """Undo create_http_app's transport mutation of the shared server module."""
    yield
    set_transport("stdio")


async def test_create_http_app_enables_export_query(restore_datamesh_policy):
    """The ASGI factory — the real hosted serving path — keeps export_query
    ENABLED (it brokers a download URL there) and mounts the endpoint at
    /<server> for multi-server ingress routing."""
    from oceanum_mcp.app import create_http_app

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OCEANUM_MCP_AUTH", "none")
        app = create_http_app("datamesh")
    assert "/datamesh" in [r.path for r in app.routes]
    async with Client(datamesh_server.mcp) as client:
        tools = {t.name for t in await client.list_tools()}
    assert "export_query" in tools


def test_create_http_app_rejects_unknown_server():
    from oceanum_mcp.app import create_http_app

    with pytest.raises(ValueError, match="Unknown server"):
        create_http_app("nonexistent")


async def test_oauth_discovery_metadata_served(restore_datamesh_policy):
    """With a public URL configured, the app serves RFC 9728 Protected
    Resource Metadata naming the Auth0 tenant, and 401s carry a
    WWW-Authenticate header pointing OAuth clients at it."""
    from oceanum_mcp.app import create_http_app

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OCEANUM_MCP_AUTH", "auto")
        mp.setenv("OCEANUM_MCP_PUBLIC_URL", "https://mcp.example.test")
        app = create_http_app("datamesh")
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # The path-suffixed form is what fastmcp serves and what the 401
            # WWW-Authenticate challenge points at (claude.ai probes it
            # first per RFC 9728) — pin it so a route move fails the test.
            resp = await client.get("/.well-known/oauth-protected-resource/datamesh")
            assert resp.status_code == 200
            assert "auth.oceanum.io" in resp.text
            assert '"resource"' in resp.text
            unauth = await client.post("/datamesh", json=INIT, headers=HDRS)
            assert unauth.status_code == 401
            assert "www-authenticate" in unauth.headers


# --- CORS (OCE-312) ---------------------------------------------------------

PREFLIGHT_HEADERS = (
    "authorization, content-type, mcp-protocol-version, mcp-session-id, "
    "x-datamesh-token, last-event-id"
)


@asynccontextmanager
async def factory_client(**env: str):
    """ASGI client for the real create_http_app("datamesh") under ``env``.

    Resets the shared datamesh module's auth first: create_http_app only
    assigns mcp.auth when a provider is built, so an earlier test's provider
    would otherwise leak into an OCEANUM_MCP_AUTH=none app.
    """
    from oceanum_mcp.app import create_http_app

    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("OCEANUM_MCP_CORS_ORIGINS", raising=False)
        mp.delenv("OCEANUM_MCP_PUBLIC_URL", raising=False)
        mp.setenv("OCEANUM_MCP_AUTH", "auto")
        for key, value in env.items():
            mp.setenv(key, value)
        mp.setattr(datamesh_server.mcp, "auth", None)
        stateless = env.get("OCEANUM_MCP_AUTH") != "none"
        app = create_http_app("datamesh", stateless=stateless)
        try:
            async with app.router.lifespan_context(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    yield client
        finally:
            set_transport("stdio")


async def _preflight(client: httpx.AsyncClient, origin: str) -> httpx.Response:
    return await client.options(
        "/datamesh",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": PREFLIGHT_HEADERS,
        },
    )


@pytest.mark.parametrize(
    "origin",
    ["https://app.oceanum.io", "https://ui.oceanum.tech", "https://vscode.dev"],
)
async def test_cors_preflight_allowed_origin_skips_auth(origin):
    """A preflight from a default-allowlisted origin succeeds without any
    credential — CORS is outermost, so auth never sees it."""
    async with factory_client() as client:
        resp = await _preflight(client, origin)
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == origin
    methods = {
        m.strip() for m in resp.headers["access-control-allow-methods"].split(",")
    }
    assert {"GET", "POST", "DELETE", "OPTIONS"} <= methods
    allowed = {
        h.strip().lower()
        for h in resp.headers["access-control-allow-headers"].split(",")
    }
    assert {h.strip() for h in PREFLIGHT_HEADERS.split(",")} <= allowed
    assert int(resp.headers["access-control-max-age"]) > 0
    assert "access-control-allow-credentials" not in resp.headers


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.com",
        "https://evil-oceanum.io",  # lookalike apex
        "https://oceanum.io.evil.com",  # allowlisted name as a prefix
        "https://app.oceanum.io.evil.com",
        "https://appxoceanum.io",  # unescaped-dot trick
        "https://a.b.oceanum.io",  # '*' is ONE label, not several
        "https://oceanum.io",  # apex is not a subdomain
        "https://app.oceanum.io.",  # trailing dot
        "https://app.oceanum.io:8443",  # port not in the allowlist entry
        "http://app.oceanum.io",  # scheme downgrade
        "https://vscode.dev.evil.com",
        "https://xvscode.dev",
        "https://APP.oceanum.io",  # browsers serialize lowercase; exact match
        "null",
    ],
)
async def test_cors_preflight_disallowed_origin_gets_no_acao(origin):
    async with factory_client() as client:
        resp = await _preflight(client, origin)
    assert resp.status_code != 200
    assert "access-control-allow-origin" not in resp.headers


async def test_cors_actual_post_carries_acao_and_exposes_session_id():
    origin = "https://app.oceanum.io"
    async with factory_client(OCEANUM_MCP_AUTH="none") as client:
        resp = await client.post(
            "/datamesh", json=INIT, headers={**HDRS, "Origin": origin}
        )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == origin
    exposed = {
        h.strip().lower()
        for h in resp.headers["access-control-expose-headers"].split(",")
    }
    assert "mcp-session-id" in exposed
    assert "mcp-session-id" in resp.headers
    assert "access-control-allow-credentials" not in resp.headers


async def test_cors_does_not_bypass_auth_for_real_requests():
    """Only preflights skip auth: an allowed-origin POST without a credential
    is still a 401 (readable by the browser, hence the ACAO header)."""
    origin = "https://app.oceanum.io"
    async with factory_client() as client:
        resp = await client.post(
            "/datamesh", json=INIT, headers={**HDRS, "Origin": origin}
        )
    assert resp.status_code == 401
    assert resp.headers["access-control-allow-origin"] == origin


async def test_cors_empty_env_disables():
    async with factory_client(OCEANUM_MCP_CORS_ORIGINS="") as client:
        resp = await _preflight(client, "https://app.oceanum.io")
    assert resp.status_code != 200
    assert "access-control-allow-origin" not in resp.headers


async def test_cors_custom_env_list_replaces_defaults():
    env = {
        "OCEANUM_MCP_CORS_ORIGINS": " https://client.example.com ,https://*.corp.test"
    }
    async with factory_client(**env) as client:
        exact = await _preflight(client, "https://client.example.com")
        wild = await _preflight(client, "https://x.corp.test")
        default = await _preflight(client, "https://app.oceanum.io")
    assert exact.headers["access-control-allow-origin"] == "https://client.example.com"
    assert wild.headers["access-control-allow-origin"] == "https://x.corp.test"
    assert "access-control-allow-origin" not in default.headers


def test_cors_origins_env_parsing(monkeypatch):
    from oceanum_mcp.common.config import DEFAULT_CORS_ORIGINS, cors_origins

    monkeypatch.delenv("OCEANUM_MCP_CORS_ORIGINS", raising=False)
    assert cors_origins() == list(DEFAULT_CORS_ORIGINS)
    for disabled in ("", "  ", " , "):
        monkeypatch.setenv("OCEANUM_MCP_CORS_ORIGINS", disabled)
        assert cors_origins() == []
    monkeypatch.setenv(
        "OCEANUM_MCP_CORS_ORIGINS", "https://A.example.com, http://localhost:6274"
    )
    assert cors_origins() == ["https://a.example.com", "http://localhost:6274"]


@pytest.mark.parametrize(
    "bad",
    [
        "*",
        "https://*",
        "https://*.io",  # wildcard over a whole TLD
        "https://*.*.example.com",
        "https://foo.*.example.com",
        "https://example.com/",
        "https://example.com/path",
        "example.com",
        "ftp://example.com",
        "https://exa mple.com",
        "https://-x.example.com",  # label may not start with "-"
        "https://x-.example.com",  # ...or end with it
        "https://example.com:0",
        "https://example.com:99999",
    ],
)
def test_cors_origins_rejects_malformed(monkeypatch, bad):
    from oceanum_mcp.common.config import cors_origins

    monkeypatch.setenv("OCEANUM_MCP_CORS_ORIGINS", bad)
    with pytest.raises(ValueError, match="OCEANUM_MCP_CORS_ORIGINS"):
        cors_origins()


@pytest.mark.parametrize("method", ["GET", "DELETE"])
async def test_cors_preflight_covers_stream_and_session_methods(method):
    """GET (server event stream) and DELETE (end session) preflight too."""
    async with factory_client() as client:
        resp = await client.options(
            "/datamesh",
            headers={
                "Origin": "https://app.oceanum.io",
                "Access-Control-Request-Method": method,
                "Access-Control-Request-Headers": "mcp-session-id",
            },
        )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "https://app.oceanum.io"


async def test_cors_wildcard_with_port_and_default_port_entries():
    """A wildcard entry keeps its port; a scheme-default port is dropped so
    it matches the port-less Origin a browser actually sends."""
    env = {
        "OCEANUM_MCP_CORS_ORIGINS": "https://*.corp.test:8443,https://app.example.com:443"
    }
    async with factory_client(**env) as client:
        wild = await _preflight(client, "https://x.corp.test:8443")
        wild_no_port = await _preflight(client, "https://x.corp.test")
        default_port = await _preflight(client, "https://app.example.com")
    assert wild.headers["access-control-allow-origin"] == "https://x.corp.test:8443"
    assert "access-control-allow-origin" not in wild_no_port.headers
    assert (
        default_port.headers["access-control-allow-origin"] == "https://app.example.com"
    )


def test_cors_origins_drops_default_ports(monkeypatch):
    from oceanum_mcp.common.config import cors_origins

    monkeypatch.setenv(
        "OCEANUM_MCP_CORS_ORIGINS",
        "https://a.example.com:443,http://b.example.com:80,https://c.example.com:80",
    )
    assert cors_origins() == [
        "https://a.example.com",
        "http://b.example.com",
        "https://c.example.com:80",
    ]


def test_cors_origin_regex_covers_only_wildcards():
    """Exact origins go to Starlette's allow_origins membership check; the
    regex is built from wildcard entries alone (None when there are none)."""
    import re

    from oceanum_mcp.app import cors_origin_regex

    assert cors_origin_regex(["https://vscode.dev"]) is None
    pattern = re.compile(
        cors_origin_regex(["https://vscode.dev", "https://*.oceanum.io"])
    )
    assert pattern.fullmatch("https://app.oceanum.io")
    assert not pattern.fullmatch("https://vscode.dev")
    assert not pattern.fullmatch("https://a.b.oceanum.io")
    assert not pattern.fullmatch("https://appxoceanum.io")
