"""Tests for the CLI dispatcher."""

import subprocess
import sys


def test_cli_list():
    """Test that --list shows all available servers."""
    result = subprocess.run(
        [sys.executable, "-m", "oceanum_mcp", "--list"],
        capture_output=True,
        text=True,
        cwd="src",
    )
    assert result.returncode == 0
    assert "datamesh" in result.stdout
    assert "storage" in result.stdout
    assert "combined" in result.stdout


def test_cli_invalid_server():
    """Test that an invalid server name is rejected."""
    result = subprocess.run(
        [sys.executable, "-m", "oceanum_mcp", "nonexistent"],
        capture_output=True,
        text=True,
        cwd="src",
    )
    assert result.returncode != 0


def test_server_registry_keys():
    """Test that the server registry contains expected entries."""
    from oceanum_mcp.cli import SERVER_REGISTRY

    assert "datamesh" in SERVER_REGISTRY
    assert "storage" in SERVER_REGISTRY
    assert "combined" in SERVER_REGISTRY
