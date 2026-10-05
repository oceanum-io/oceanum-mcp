"""OCE-333: cancelling an in-flight sync tool must not take down the stdio server.

FastMCP runs sync tools in an anyio worker thread that a
notifications/cancelled cannot interrupt, so the tool finishes after the
cancel. With mcp 1.x that late answer failed the SDK's "Request already
responded to" assertion and the stdio server exited. These tests drive the
real `oceanum-mcp` stdio loop (`cli._run_stdio`) in a subprocess with the
Datamesh boundary mocked to be slow, cancel the call mid-flight, wait until
the tool has returned, then check that the same server still answers.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# How long a slow-ds call blocks in the mocked gateway: long enough to cancel
# it mid-flight, short enough to keep the test quick.
SLOW_SECONDS = 1.5
# Time allowed for the late answer to reach the SDK after the tool returns.
SETTLE_SECONDS = 0.5
TIMEOUT = 60.0

# Runs in the subprocess. Only the network boundary is mocked; calls on the
# "slow-ds" datasource block in the worker thread and drop marker files so
# the test knows when the tool entered and when it returned. The "returned"
# marker records whether the tool's own thread saw the request cancelled, which
# proves the server received and applied the notifications/cancelled.
_SERVER_SCRIPT = """
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import oceanum_mcp.servers.datamesh.server as server
from oceanum_mcp.cli import _run_stdio
from tests.conftest import make_stage
from tests.test_servers.test_datamesh import _mock_datasource, _small_dataset

markers = Path(os.environ["OCE333_MARKERS"])
slow_seconds = float(os.environ["OCE333_SLOW_SECONDS"])


def gateway(tool, datasource):
    if datasource == "slow-ds":
        (markers / f"entered-{tool}").touch()
        time.sleep(slow_seconds)
        seen = "cancelled" if server._cancel_requested() else "not cancelled"
        partial = markers / f".returned-{tool}"
        partial.write_text(seen)
        partial.rename(markers / f"returned-{tool}")  # atomic: never read empty


def fake_stage(conn, query):
    gateway("stage", query.datasource)
    return make_stage()


def get_datasource(datasource_id, *args, **kwargs):
    gateway("get_datasource", datasource_id)
    return _mock_datasource(id=datasource_id)


conn = MagicMock()
conn.get_datasource.side_effect = get_datasource
conn.query.return_value = _small_dataset()

with patch.object(server, "get_datamesh_connector", return_value=conn), patch.object(
    server, "_stage", side_effect=fake_stage
):
    _run_stdio(server.mcp)
"""

_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 0,
    "method": "initialize",
    "params": {
        # The handshake era real stdio clients (Claude Desktop/Code) speak.
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "oce-333-test", "version": "0"},
    },
}


class _StdioServer:
    """The datamesh server in a subprocess, driven with raw JSON-RPC lines.

    Raw lines rather than fastmcp's Client: they pin the 2025-06-18 handshake
    that stdio clients speak today, and the same file runs unchanged against
    fastmcp 3 / mcp 1.x, which is how it shows red on the pre-upgrade code.
    A reader thread keeps the harness synchronous and independent of either
    SDK's client.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.markers = tmp_path / "markers"
        self.markers.mkdir()
        self._stderr = (tmp_path / "stderr.txt").open("w+")
        env = {
            **os.environ,
            "OCE333_MARKERS": str(self.markers),
            "OCE333_SLOW_SECONDS": str(SLOW_SECONDS),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        self.proc = subprocess.Popen(
            [sys.executable, "-c", _SERVER_SCRIPT],
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
        )
        self.messages: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.received: list[dict[str, Any]] = []
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            try:
                self.messages.put(json.loads(line))
            except ValueError:
                self.messages.put({"non_json_stdout": line})
        self.messages.put(None)  # EOF: the server exited

    def send(self, message: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def response(self, request_id: int) -> dict[str, Any]:
        """Wait for the answer to request_id; fails if the server exits first."""
        deadline = time.monotonic() + TIMEOUT
        while True:
            for message in self.received:
                if message.get("id") == request_id:
                    return message
            try:
                message = self.messages.get(timeout=deadline - time.monotonic())
            except (queue.Empty, ValueError):
                pytest.fail(f"no answer to request {request_id}: {self.stderr()}")
            if message is None:
                pytest.fail(
                    f"stdio server exited (code {self.proc.wait()}) before "
                    f"answering request {request_id}: {self.stderr()}"
                )
            if "non_json_stdout" in message:
                pytest.fail(f"non-JSON on stdout: {message['non_json_stdout']!r}")
            self.received.append(message)

    def drain(self) -> None:
        """Collect whatever the server has written so far."""
        while True:
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                return
            if message is not None:
                self.received.append(message)

    def wait_for_marker(self, name: str) -> None:
        deadline = time.monotonic() + TIMEOUT
        while not (self.markers / name).exists():
            if self.proc.poll() is not None:
                pytest.fail(f"stdio server exited early: {self.stderr()}")
            if time.monotonic() > deadline:
                pytest.fail(f"marker {name} never appeared: {self.stderr()}")
            time.sleep(0.01)

    def cancel_seen(self, name: str) -> str:
        return (self.markers / name).read_text()

    def stderr(self) -> str:
        self._stderr.seek(0)
        return self._stderr.read()[-4000:]

    def call(self, request_id: int, tool: str, arguments: dict[str, Any]) -> None:
        self.send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": tool, "arguments": arguments},
            }
        )

    def close(self) -> int:
        """End the session with stdin EOF; returns the exit code."""
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        return self.proc.wait(timeout=TIMEOUT)

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self._reader.join(TIMEOUT)  # reads to EOF, then the pipes can close
        for pipe in (self.proc.stdin, self.proc.stdout):
            if pipe is not None:
                pipe.close()
        self._stderr.close()


@pytest.fixture
def stdio_server(tmp_path: Path) -> Any:
    server = _StdioServer(tmp_path)
    try:
        server.send(_INITIALIZE)
        assert "result" in server.response(0)
        server.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        yield server
    finally:
        server.kill()


# (tool, the mocked gateway call it blocks in)
SYNC_TOOLS = [
    ("query_data", "stage"),
    ("stage_query", "stage"),
    ("get_datasource_info", "get_datasource"),
]


@pytest.mark.parametrize(("tool", "blocks_in"), SYNC_TOOLS)
def test_cancelled_sync_tool_does_not_crash_stdio_server(
    stdio_server: _StdioServer, tool: str, blocks_in: str
) -> None:
    stdio_server.call(1, tool, {"datasource_id": "slow-ds"})
    stdio_server.wait_for_marker(f"entered-{blocks_in}")
    stdio_server.send(
        {
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": 1, "reason": "user cancelled"},
        }
    )
    # The worker thread runs to completion regardless of the cancel; let its
    # late answer reach the SDK before probing the server.
    stdio_server.wait_for_marker(f"returned-{blocks_in}")
    assert stdio_server.cancel_seen(f"returned-{blocks_in}") == "cancelled"
    time.sleep(SETTLE_SECONDS)

    # The same server still serves a later call.
    stdio_server.call(2, "get_datasource_info", {"datasource_id": "fast-ds"})
    later = stdio_server.response(2)
    assert "result" in later, later
    assert json.loads(later["result"]["content"][0]["text"])["id"] == "fast-ds"

    # A cancelled request is never answered: no result, no error.
    stdio_server.drain()
    assert [m for m in stdio_server.received if m.get("id") == 1] == []

    assert stdio_server.close() == 0


async def test_cancelled_sync_tools_on_modern_protocol_era(tmp_path: Path) -> None:
    """The same over the 2026-07-28 era, which fastmcp's Client negotiates.

    An mcp>=2 client cancels by abandoning the call: its dispatcher sends the
    notifications/cancelled. One server takes all three cancels in turn.
    """
    import anyio
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    markers = tmp_path / "markers"
    markers.mkdir()
    transport = StdioTransport(
        command=sys.executable,
        args=["-c", _SERVER_SCRIPT],
        env={
            **os.environ,
            "OCE333_MARKERS": str(markers),
            "OCE333_SLOW_SECONDS": str(SLOW_SECONDS),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        cwd=str(REPO_ROOT),
        keep_alive=False,
    )

    async def wait_for_marker(name: str) -> None:
        with anyio.fail_after(TIMEOUT):
            while not (markers / name).exists():
                await anyio.sleep(0.01)

    async with Client(transport, timeout=TIMEOUT) as client:
        # A 2026-era (sessionless) protocol, not the handshake era above.
        assert client.protocol_version >= "2026-07-28"
        for tool, blocks_in in SYNC_TOOLS:
            for marker in markers.iterdir():
                marker.unlink()
            answered: list[Any] = []

            async def call(tool: str = tool, answered: list[Any] = answered) -> None:
                answered.append(
                    await client.call_tool(tool, {"datasource_id": "slow-ds"})
                )

            async with anyio.create_task_group() as tg:
                tg.start_soon(call)
                await wait_for_marker(f"entered-{blocks_in}")
                tg.cancel_scope.cancel()
            await wait_for_marker(f"returned-{blocks_in}")
            # The client's abandon sent notifications/cancelled, and the
            # server applied it to the tool's request.
            assert (markers / f"returned-{blocks_in}").read_text() == "cancelled"
            await anyio.sleep(SETTLE_SECONDS)

            assert answered == [], tool
            later = await client.call_tool(
                "get_datasource_info", {"datasource_id": "fast-ds"}
            )
            assert json.loads(later.content[0].text)["id"] == "fast-ds", tool
