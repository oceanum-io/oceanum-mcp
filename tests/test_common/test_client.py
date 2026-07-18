"""Tests for per-request credential resolution and client caching."""

import os
from unittest.mock import MagicMock, patch

import pytest

from fastmcp.server.auth import AccessToken

from oceanum_mcp.common.client import (
    CREDENTIAL_CLAIM,
    _ClientCache,
    _datamesh_cache,
    _storage_cache,
    get_datamesh_connector,
    resolve_credential,
)


def make_access_token(credential: str) -> AccessToken:
    return AccessToken(
        token=credential,
        client_id="user",
        scopes=[],
        claims={CREDENTIAL_CLAIM: credential},
    )


@pytest.fixture(autouse=True)
def clean_caches():
    _datamesh_cache.clear()
    _storage_cache.clear()
    yield
    _datamesh_cache.clear()
    _storage_cache.clear()


def test_resolve_credential_prefers_auth_context():
    """An authenticated request's credential wins over the environment."""
    with patch(
        "fastmcp.server.dependencies.get_access_token",
        return_value=make_access_token("request-token"),
    ):
        with patch.dict(os.environ, {"DATAMESH_TOKEN": "env-token"}, clear=False):
            assert resolve_credential() == "request-token"


def test_resolve_credential_env_fallback():
    """Outside an auth context (stdio) the environment token is used."""
    with patch("fastmcp.server.dependencies.get_access_token", return_value=None):
        with patch.dict(os.environ, {"DATAMESH_TOKEN": "env-token"}, clear=False):
            assert resolve_credential() == "env-token"


def test_resolve_credential_missing_raises():
    with patch("fastmcp.server.dependencies.get_access_token", return_value=None):
        env = {k: v for k, v in os.environ.items() if k != "DATAMESH_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="credential"):
                resolve_credential()


def test_multi_tenant_connector_isolation():
    """Different request credentials must never share a Connector."""
    connector_cls = MagicMock(side_effect=lambda **kw: MagicMock(name=kw["token"]))
    with patch("oceanum.datamesh.Connector", connector_cls):
        with patch(
            "fastmcp.server.dependencies.get_access_token",
            return_value=make_access_token("token-a"),
        ):
            conn_a1 = get_datamesh_connector()
            conn_a2 = get_datamesh_connector()
        with patch(
            "fastmcp.server.dependencies.get_access_token",
            return_value=make_access_token("token-b"),
        ):
            conn_b = get_datamesh_connector()

    assert conn_a1 is conn_a2, "same credential must reuse the cached client"
    assert conn_a1 is not conn_b, "different credentials must get different clients"
    tokens = [c.kwargs["token"] for c in connector_cls.call_args_list]
    assert tokens == ["token-a", "token-b"]


def test_client_cache_ttl_expiry():
    cache = _ClientCache(max_entries=8, ttl_s=0.0)
    first = cache.get_or_create(("t", "s"), lambda: object())
    second = cache.get_or_create(("t", "s"), lambda: object())
    assert first is not second, "expired entries must be rebuilt"


def test_client_cache_bounded():
    cache = _ClientCache(max_entries=2, ttl_s=60.0)
    a = cache.get_or_create(("a", "s"), lambda: "A")
    cache.get_or_create(("b", "s"), lambda: "B")
    cache.get_or_create(("c", "s"), lambda: "C")
    rebuilt = cache.get_or_create(("a", "s"), lambda: "A2")
    assert a == "A" and rebuilt == "A2", "oldest entry must be evicted at the bound"
