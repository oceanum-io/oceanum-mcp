"""Tests for the network-transport auth providers."""

import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from fastmcp.server.auth import AccessToken
from fastmcp.server.auth.providers.jwt import JWTVerifier

from oceanum_mcp.common.auth import (
    Auth0JWTVerifier,
    DatameshTokenVerifier,
    build_auth_provider,
)
from oceanum_mcp.common.client import CREDENTIAL_CLAIM


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient; records calls, returns a canned response."""

    calls = 0
    response: httpx.Response | None = None
    error: Exception | None = None

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, headers=None):
        type(self).calls += 1
        if type(self).error is not None:
            raise type(self).error
        return type(self).response


@pytest.fixture
def fake_gateway():
    _FakeAsyncClient.calls = 0
    _FakeAsyncClient.response = None
    _FakeAsyncClient.error = None
    with patch("oceanum_mcp.common.auth.httpx.AsyncClient", _FakeAsyncClient):
        yield _FakeAsyncClient


async def test_datamesh_verifier_valid_token(fake_gateway):
    fake_gateway.response = httpx.Response(
        200, json=[{"username": "alice", "is_active": True}]
    )
    verifier = DatameshTokenVerifier(service="https://datamesh.test")
    result = await verifier.verify_token("good-token")
    assert result is not None
    assert result.claims[CREDENTIAL_CLAIM] == "good-token"
    assert result.subject == "alice"


async def test_datamesh_verifier_invalid_token(fake_gateway):
    fake_gateway.response = httpx.Response(401, json={"detail": "Invalid token."})
    verifier = DatameshTokenVerifier(service="https://datamesh.test")
    assert await verifier.verify_token("bad-token") is None


async def test_datamesh_verifier_caches_results(fake_gateway):
    fake_gateway.response = httpx.Response(200, json=[{"username": "alice"}])
    verifier = DatameshTokenVerifier(service="https://datamesh.test")
    await verifier.verify_token("good-token")
    await verifier.verify_token("good-token")
    assert fake_gateway.calls == 1, "second verification must be served from cache"


async def test_datamesh_verifier_gateway_error_fails_closed_uncached(fake_gateway):
    fake_gateway.error = httpx.ConnectError("boom")
    verifier = DatameshTokenVerifier(service="https://datamesh.test")
    assert await verifier.verify_token("good-token") is None
    fake_gateway.error = None
    fake_gateway.response = httpx.Response(200, json=[{"username": "alice"}])
    result = await verifier.verify_token("good-token")
    assert result is not None, "gateway outages must not be cached as invalid"


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
    assert verifier.audience == "https://api.oceanum.io"


def test_build_auth_provider_modes():
    with patch.dict(os.environ, {"OCEANUM_MCP_AUTH": "datamesh"}, clear=False):
        assert isinstance(build_auth_provider(), DatameshTokenVerifier)
    with patch.dict(os.environ, {"OCEANUM_MCP_AUTH": "auth0"}, clear=False):
        assert isinstance(build_auth_provider(), Auth0JWTVerifier)
    with patch.dict(os.environ, {"OCEANUM_MCP_AUTH": "none"}, clear=False):
        assert build_auth_provider() is None
    with patch.dict(os.environ, {"OCEANUM_MCP_AUTH": "bogus"}, clear=False):
        with pytest.raises(ValueError, match="OCEANUM_MCP_AUTH"):
            build_auth_provider()
