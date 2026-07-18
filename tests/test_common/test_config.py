"""Tests for shared configuration."""

import os
from unittest.mock import patch

import pytest

from oceanum_mcp.common.config import (
    auth0_audience,
    auth0_domain,
    auth_mode,
    datamesh_service,
    is_network_transport,
    public_url,
    set_transport,
    storage_service,
)


def test_default_service_urls():
    with patch.dict(os.environ, {}, clear=True):
        assert datamesh_service() == "https://datamesh.oceanum.io"
        assert storage_service() == "https://storage.oceanum.io"


def test_custom_domain():
    with patch.dict(os.environ, {"OCEANUM_DOMAIN": "staging.oceanum.io"}, clear=True):
        assert datamesh_service() == "https://datamesh.staging.oceanum.io"
        assert storage_service() == "https://storage.staging.oceanum.io"


def test_custom_service_urls():
    env = {
        "DATAMESH_SERVICE": "https://custom-datamesh.example.com",
        "STORAGE_SERVICE": "https://custom-storage.example.com",
    }
    with patch.dict(os.environ, env, clear=True):
        assert datamesh_service() == "https://custom-datamesh.example.com"
        assert storage_service() == "https://custom-storage.example.com"


def test_auth_mode_default_and_validation():
    with patch.dict(os.environ, {}, clear=True):
        assert auth_mode() == "auto"
    with patch.dict(os.environ, {"OCEANUM_MCP_AUTH": "AUTH0"}, clear=True):
        assert auth_mode() == "auth0"
    with patch.dict(os.environ, {"OCEANUM_MCP_AUTH": "bogus"}, clear=True):
        with pytest.raises(ValueError, match="OCEANUM_MCP_AUTH"):
            auth_mode()


def test_auth0_defaults():
    with patch.dict(os.environ, {}, clear=True):
        assert auth0_domain() == "auth.oceanum.io"
        assert auth0_audience() == "https://api.oceanum.io"


def test_public_url():
    with patch.dict(os.environ, {}, clear=True):
        assert public_url() is None
    with patch.dict(
        os.environ, {"OCEANUM_MCP_PUBLIC_URL": "https://mcp.oceanum.io/"}, clear=True
    ):
        assert public_url() == "https://mcp.oceanum.io"


def test_transport_flag():
    assert not is_network_transport()
    try:
        set_transport("http")
        assert is_network_transport()
        set_transport("sse")
        assert is_network_transport()
    finally:
        set_transport("stdio")
    assert not is_network_transport()
