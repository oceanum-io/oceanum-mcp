"""Tests for shared configuration."""

import os
from unittest.mock import patch

import pytest

from oceanum_mcp.common.config import load_config, OceanumConfig


def test_load_config_with_token():
    """Test config loads correctly with DATAMESH_TOKEN set."""
    with patch.dict(os.environ, {"DATAMESH_TOKEN": "test-token"}, clear=False):
        config = load_config()
        assert isinstance(config, OceanumConfig)
        assert config.token == "test-token"
        assert "datamesh" in config.datamesh_service
        assert "storage" in config.storage_service


def test_load_config_missing_token():
    """Test config raises ValueError when DATAMESH_TOKEN is missing."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="DATAMESH_TOKEN"):
            load_config()


def test_load_config_custom_domain():
    """Test config respects OCEANUM_DOMAIN override."""
    env = {"DATAMESH_TOKEN": "test-token", "OCEANUM_DOMAIN": "staging.oceanum.io"}
    with patch.dict(os.environ, env, clear=True):
        config = load_config()
        assert config.datamesh_service == "https://datamesh.staging.oceanum.io"
        assert config.storage_service == "https://storage.staging.oceanum.io"


def test_load_config_custom_service_urls():
    """Test config respects explicit service URL overrides."""
    env = {
        "DATAMESH_TOKEN": "test-token",
        "DATAMESH_SERVICE": "https://custom-datamesh.example.com",
        "STORAGE_SERVICE": "https://custom-storage.example.com",
    }
    with patch.dict(os.environ, env, clear=True):
        config = load_config()
        assert config.datamesh_service == "https://custom-datamesh.example.com"
        assert config.storage_service == "https://custom-storage.example.com"
