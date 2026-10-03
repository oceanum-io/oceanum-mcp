"""Tests for the CLI dispatcher."""

import json
import subprocess
import sys
from pathlib import Path


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


def test_cli_help_lists_transports():
    """The http transport and its bind options are exposed."""
    result = subprocess.run(
        [sys.executable, "-m", "oceanum_mcp", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent / "src",
    )
    assert result.returncode == 0
    assert "http" in result.stdout
    assert "--host" in result.stdout
    assert "--port" in result.stdout
    assert "--stateless" in result.stdout
    assert "--path" in result.stdout


# ---------------------------------------------------------------------------
# stdio hygiene (OCE-311): no banner, nothing but JSON-RPC on stdout
# ---------------------------------------------------------------------------

_STDIO_SERVER_SCRIPT = """
from fastmcp import FastMCP

from oceanum_mcp.cli import _run_stdio

mcp = FastMCP("noisy")


@mcp.tool
def noisy() -> str:
    # Stands in for the oceanum library, which print()s to stdout (e.g. its
    # "new version available" notice on first connect). flush=True mirrors an
    # unbuffered client launch (PYTHONUNBUFFERED / python -u).
    print("library noise", flush=True)
    return "ok"


_run_stdio(mcp)
"""

_JSONRPC_SESSION = [
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        },
    },
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "noisy", "arguments": {}},
    },
]


def test_stdio_runs_without_banner(monkeypatch):
    """The stdio branch of main() disables the FastMCP banner."""
    import importlib
    from unittest.mock import patch

    from oceanum_mcp import cli

    module = importlib.import_module(cli.SERVER_REGISTRY["datamesh"])
    monkeypatch.setattr(sys, "argv", ["oceanum-mcp", "datamesh"])
    with patch.object(module.mcp, "run") as run:
        cli.main()
    run.assert_called_once_with(transport="stdio", show_banner=False)


def test_stdio_stdout_guard_routes_text_to_stderr(capsys):
    """While serving stdio, print() lands on stderr; .buffer stays stdout's."""
    from unittest.mock import MagicMock

    from oceanum_mcp import cli

    real_stdout = sys.stdout
    seen = {}

    def fake_run(**kwargs):
        seen["buffer"] = sys.stdout.buffer
        print("stray")

    server = MagicMock()
    server.run.side_effect = fake_run
    cli._run_stdio(server)

    assert sys.stdout is real_stdout
    assert seen["buffer"] is real_stdout.buffer
    out, err = capsys.readouterr()
    assert "stray" not in out
    assert "stray" in err


def test_stdio_stdout_is_pure_jsonrpc():
    """End to end over a real stdio session: no banner, no stray stdout text."""
    import threading

    proc = subprocess.Popen(
        [sys.executable, "-c", _STDIO_SERVER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    watchdog = threading.Timer(60, proc.kill)
    watchdog.start()
    try:
        for message in _JSONRPC_SESSION:
            proc.stdin.write(json.dumps(message) + "\n")
        proc.stdin.flush()
        # Keep stdin open until the tool result arrives: EOF ends the session.
        messages = []
        while not any(m.get("id") == 2 for m in messages):
            line = proc.stdout.readline()
            assert line, "server closed stdout before answering the tool call"
            messages.append(json.loads(line))  # raises on any non-JSON line
        stdout_rest, stderr = proc.communicate()  # closes stdin -> clean exit
    finally:
        watchdog.cancel()
        proc.kill()

    assert not stdout_rest.strip()
    assert all(m.get("jsonrpc") == "2.0" for m in messages)
    call = next(m for m in messages if m.get("id") == 2)
    assert call["result"]["content"][0]["text"] == "ok"
    assert "library noise" in stderr
    assert "gofastmcp.com" not in stderr  # banner
