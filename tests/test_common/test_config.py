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
    is_read_only,
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


def test_max_inline_rows_default_and_validation():
    from oceanum_mcp.common.config import max_inline_rows

    with patch.dict(os.environ, {}, clear=True):
        assert max_inline_rows() == 100
    with patch.dict(os.environ, {"OCEANUM_MCP_MAX_INLINE_ROWS": "250"}, clear=True):
        assert max_inline_rows() == 250
    for bad in ("0", "-5"):
        with patch.dict(os.environ, {"OCEANUM_MCP_MAX_INLINE_ROWS": bad}, clear=True):
            with pytest.raises(ValueError, match="at least 1"):
                max_inline_rows()
    with patch.dict(os.environ, {"OCEANUM_MCP_MAX_INLINE_ROWS": "x"}, clear=True):
        with pytest.raises(ValueError, match="integer row count"):
            max_inline_rows()


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


@pytest.mark.parametrize("value", ["1", "TRUE", "yes", "on", " On "])
def test_read_only_true_values(monkeypatch, value):
    monkeypatch.setenv("OCEANUM_MCP_READ_ONLY", value)
    assert is_read_only() is True


@pytest.mark.parametrize("value", ["0", "false", "No", "", "off", "OFF "])
def test_read_only_false_values(monkeypatch, value):
    monkeypatch.setenv("OCEANUM_MCP_READ_ONLY", value)
    assert is_read_only() is False


@pytest.mark.parametrize("value", ["disabled", "ture", "readonly", "a" * 40])
def test_read_only_unrecognised_value_fails_closed(monkeypatch, value):
    """A misspelled flag must stop the server, not leave write tools on. The
    value is not echoed: it may be a token pasted into the wrong variable."""
    monkeypatch.setenv("OCEANUM_MCP_READ_ONLY", value)
    with pytest.raises(ValueError, match="OCEANUM_MCP_READ_ONLY must be") as exc:
        is_read_only()
    assert "on or 0/false/no/off" in str(exc.value)
    assert value not in str(exc.value)


def test_read_only_unrecognised_value_stops_server_start(monkeypatch):
    import importlib

    import oceanum_mcp.servers.storage.server as storage_server

    monkeypatch.setenv("OCEANUM_MCP_READ_ONLY", "disabled")
    try:
        with pytest.raises(ValueError, match="OCEANUM_MCP_READ_ONLY"):
            importlib.reload(storage_server)
    finally:
        monkeypatch.delenv("OCEANUM_MCP_READ_ONLY")
        importlib.reload(storage_server)
