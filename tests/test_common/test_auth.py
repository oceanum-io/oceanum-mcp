"""Tests for the network-transport auth providers."""

import os
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from fastmcp.server.auth import AccessToken
from fastmcp.server.auth.providers.jwt import JWTVerifier

from fastmcp.server.auth import MultiAuth, RemoteAuthProvider

from oceanum_mcp.common.auth import (
    Auth0JWTVerifier,
    DatameshTokenVerifier,
    build_auth_provider,
)
from oceanum_mcp.common.client import CREDENTIAL_CLAIM


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient; records calls, returns a canned response."""

    calls = 0
    status = 200
    payload: Any = None
    error: Exception | None = None

    def __init__(self, **kwargs: Any) -> None:
        pass

    async def get(self, url: str, headers: dict | None = None) -> httpx.Response:
        cls = type(self)
        cls.calls += 1
        if cls.error is not None:
            raise cls.error
        return httpx.Response(
            cls.status, json=cls.payload, request=httpx.Request("GET", url)
        )


@pytest.fixture
def fake_gateway():
    _FakeAsyncClient.calls = 0
    _FakeAsyncClient.status = 200
    _FakeAsyncClient.payload = None
    _FakeAsyncClient.error = None
    with patch("oceanum_mcp.common.auth.httpx.AsyncClient", _FakeAsyncClient):
        yield _FakeAsyncClient


async def test_datamesh_verifier_valid_token(fake_gateway):
    fake_gateway.payload = [{"username": "alice", "is_active": True}]
    verifier = DatameshTokenVerifier(service="https://datamesh.test")
    result = await verifier.verify_token("good-token")
    assert result is not None
    assert result.claims[CREDENTIAL_CLAIM] == "good-token"
    assert result.subject == "alice"
    assert result.expires_at is not None, "revocation window must be bounded"


async def test_datamesh_verifier_invalid_token(fake_gateway):
    fake_gateway.status = 401
    fake_gateway.payload = {"detail": "Invalid token."}
    verifier = DatameshTokenVerifier(service="https://datamesh.test")
    assert await verifier.verify_token("bad-token") is None


async def test_datamesh_verifier_caches_results(fake_gateway):
    fake_gateway.payload = [{"username": "alice"}]
    verifier = DatameshTokenVerifier(service="https://datamesh.test")
    await verifier.verify_token("good-token")
    await verifier.verify_token("good-token")
    assert fake_gateway.calls == 1, "second verification must be served from cache"


async def test_datamesh_verifier_caches_confirmed_rejections(fake_gateway):
    fake_gateway.status = 401
    fake_gateway.payload = {"detail": "Invalid token."}
    verifier = DatameshTokenVerifier(service="https://datamesh.test")
    assert await verifier.verify_token("bad-token") is None
    assert await verifier.verify_token("bad-token") is None
    assert fake_gateway.calls == 1, "confirmed rejection must be served from cache"


async def test_datamesh_verifier_gateway_error_fails_closed_uncached(fake_gateway):
    fake_gateway.error = httpx.ConnectError("boom")
    verifier = DatameshTokenVerifier(service="https://datamesh.test")
    assert await verifier.verify_token("good-token") is None
    fake_gateway.error = None
    fake_gateway.payload = [{"username": "alice"}]
    result = await verifier.verify_token("good-token")
    assert result is not None, "gateway outages must not be cached as invalid"


@pytest.mark.parametrize("status", [429, 500, 502, 503, 301])
async def test_datamesh_verifier_non_auth_status_fails_closed_uncached(
    fake_gateway, status
):
    """Only 401/403 mean 'invalid token'; anything else is an outage and must
    not lock a valid token out past the blip itself."""
    fake_gateway.status = status
    verifier = DatameshTokenVerifier(service="https://datamesh.test")
    assert await verifier.verify_token("good-token") is None
    fake_gateway.status = 200
    fake_gateway.payload = [{"username": "alice"}]
    result = await verifier.verify_token("good-token")
    assert result is not None, f"status {status} must not be cached as invalid"


async def test_auth0_verifier_forwards_bearer_credential():
    jwt_result = AccessToken(token="jwt", client_id="c", scopes=[], claims={})
    with patch.object(
        JWTVerifier, "verify_token", new=AsyncMock(return_value=jwt_result)
    ):
        verifier = Auth0JWTVerifier(domain="auth.test", audience="https://api.test")
        result = await verifier.verify_token("the-jwt")
    assert result is not None
    assert result.claims[CREDENTIAL_CLAIM] == "Bearer the-jwt"


async def test_auth0_verifier_rejection_passthrough():
    with patch.object(JWTVerifier, "verify_token", new=AsyncMock(return_value=None)):
        verifier = Auth0JWTVerifier(domain="auth.test", audience="https://api.test")
        assert await verifier.verify_token("bad-jwt") is None


def test_auth0_verifier_default_tenant():
    verifier = Auth0JWTVerifier()
    assert verifier.jwks_uri == "https://auth.oceanum.io/.well-known/jwks.json"
    assert verifier.issuer == "https://auth.oceanum.io/"
    assert verifier.audience == "https://mcp.oceanum.io/datamesh"


def test_build_auth_provider_modes():
    with patch.dict(os.environ, {"OCEANUM_MCP_AUTH": "auto"}, clear=False):
        assert isinstance(build_auth_provider(), MultiAuth)
    with patch.dict(os.environ, {"OCEANUM_MCP_AUTH": "datamesh"}, clear=False):
        assert isinstance(build_auth_provider(), DatameshTokenVerifier)
    with patch.dict(os.environ, {"OCEANUM_MCP_AUTH": "auth0"}, clear=False):
        assert isinstance(build_auth_provider(), Auth0JWTVerifier)
    with patch.dict(os.environ, {"OCEANUM_MCP_AUTH": "none"}, clear=False):
        assert build_auth_provider() is None
    with patch.dict(os.environ, {"OCEANUM_MCP_AUTH": "bogus"}, clear=False):
        with pytest.raises(ValueError, match="OCEANUM_MCP_AUTH"):
            build_auth_provider()


def test_build_auth_provider_discovery_with_public_url():
    """A public URL upgrades the JWT side to a RemoteAuthProvider serving
    RFC 9728 discovery metadata (how claude.ai connectors find Auth0)."""
    env = {"OCEANUM_MCP_PUBLIC_URL": "https://mcp.oceanum.io"}
    with patch.dict(os.environ, {**env, "OCEANUM_MCP_AUTH": "auth0"}, clear=False):
        provider = build_auth_provider()
        assert isinstance(provider, RemoteAuthProvider)
        assert "auth.oceanum.io" in str(provider.authorization_servers[0])
    with patch.dict(os.environ, {**env, "OCEANUM_MCP_AUTH": "auto"}, clear=False):
        assert isinstance(build_auth_provider(), MultiAuth)


async def test_datamesh_verifier_declines_jwt_shaped_tokens(fake_gateway):
    """JWTs are never Datamesh tokens: no gateway round trip, no negative-cache
    entry for a credential that belongs to the other verifier."""
    verifier = DatameshTokenVerifier(service="https://datamesh.test")
    assert (
        await verifier.verify_token("eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ4In0.sig") is None
    )
    assert fake_gateway.calls == 0


async def test_auto_mode_accepts_either_credential(fake_gateway):
    """MultiAuth routes a JWT to the Auth0 verifier and an opaque token to the
    Datamesh verifier, each yielding its own credential claim."""
    fake_gateway.payload = [{"username": "alice"}]
    jwt = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ4In0.sig"
    jwt_result = AccessToken(token=jwt, client_id="c", scopes=[], claims={})

    async def fake_jwt_verify(self, token):
        return jwt_result if token == jwt else None

    with patch.dict(os.environ, {"OCEANUM_MCP_AUTH": "auto"}, clear=False):
        provider = build_auth_provider()
    with patch.object(JWTVerifier, "verify_token", new=fake_jwt_verify):
        via_jwt = await provider.verify_token(jwt)
        via_datamesh = await provider.verify_token("opaque-datamesh-token")
    assert via_jwt is not None
    assert via_jwt.claims[CREDENTIAL_CLAIM] == f"Bearer {jwt}"
    assert via_datamesh is not None
    assert via_datamesh.claims[CREDENTIAL_CLAIM] == "opaque-datamesh-token"
    assert fake_gateway.calls == 1, "only the opaque token may reach the gateway"
