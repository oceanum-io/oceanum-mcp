"""OCE-323: an MCP cancellation stops an in-flight local export.

FastMCP runs the sync export_query in an anyio worker thread that a
notifications/cancelled cannot interrupt; the tool polls the request's cancel
scope between NetCDF chunks instead. These tests cancel real exports through
the MCP protocol (fastmcp's in-memory client) with no network: the Datamesh
fetch is mocked and the dataset's chunks are synthetic and deliberately slow.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import anyio
import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from fastmcp import Client
from mcp.shared.exceptions import McpError

from oceanum.datamesh.exceptions import DatameshConnectError
from oceanum.datamesh.query import Container

from oceanum_mcp.servers.datamesh import server
from tests.conftest import make_stage

# Synthetic export: N_CHUNKS chunks of CHUNK_SECONDS each on two dask workers,
# i.e. at least N_CHUNKS * CHUNK_SECONDS / 2 = 20 s to write if never cancelled.
N_CHUNKS = 400
CHUNK_SECONDS = 0.1
CHUNK_SHAPE = (1, 512, 512)  # 2 MB of float64 per chunk
CHUNK_BYTES = 8 * 512 * 512
# Acceptance: I/O stops "within a few seconds" of the cancel.
MAX_STOP_SECONDS = 3.0


class _SlowChunks:
    """A lazy dataset whose chunks each take CHUNK_SECONDS to "fetch"."""

    def __init__(self) -> None:
        self.calls = 0
        self.threads: set[threading.Thread] = set()
        self.last_chunk_end = 0.0
        self._lock = threading.Lock()

    def _block(self, block: np.ndarray) -> np.ndarray:
        if block.size == 0:  # dask's meta inference, not a chunk
            return block
        with self._lock:
            self.calls += 1
            self.threads.add(threading.current_thread())
        time.sleep(CHUNK_SECONDS)
        self.last_chunk_end = time.monotonic()
        return block + 1.0

    def dataset(self) -> xr.Dataset:
        base = da.zeros((N_CHUNKS, *CHUNK_SHAPE[1:]), chunks=CHUNK_SHAPE)
        return xr.Dataset(
            {"v": (("time", "y", "x"), base.map_blocks(self._block, dtype="f8"))}
        )


def _partial_files(parent: Path) -> list[Path]:
    return [p for d in parent.glob(".*.partial") for p in d.iterdir()]


def _disk_bytes(path: Path) -> int:
    # Allocated bytes: a NetCDF file's apparent size can jump ahead of the
    # data actually written to disk.
    try:
        return path.stat().st_blocks * 512
    except FileNotFoundError:
        return -1


def _open_fds_under(parent: Path) -> list[str]:
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.exists():  # not Linux: nothing to inspect
        return []
    out = []
    for fd in fd_dir.iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith(str(parent)):
            out.append(target)
    return out


async def _wait_until(predicate: Any, timeout: float) -> None:
    with anyio.fail_after(timeout):
        while not predicate():
            await anyio.sleep(0.01)


@pytest.fixture
def two_workers() -> Any:
    # Pin the pool size so the uncancelled write time does not depend on the
    # machine's core count. export_query reads dask's num_workers.
    with dask.config.set(num_workers=2):
        yield


async def _call_and_cancel(
    client: Client, args: dict[str, Any], started: Any, timeout: float = 10
) -> tuple[list[BaseException], float]:
    """Call export_query, wait for started(), cancel it; return (errors, t_cancel)."""
    errors: list[BaseException] = []

    async def call() -> None:
        try:
            await client.call_tool("export_query", args)
        except McpError as exc:
            errors.append(exc)

    async with anyio.create_task_group() as tg:
        tg.start_soon(call)
        await _wait_until(started, timeout)
        # call_tool exposes no request id: it is the session's last one.
        request_id = client.session._request_id - 1
        t_cancel = time.monotonic()
        await client.cancel(request_id, reason="test")
    return errors, t_cancel


class TestExportCancellation:
    async def test_cancel_stops_netcdf_write_and_removes_partial_file(
        self, mock_conn, mock_stage, tmp_path, two_workers
    ):
        chunks = _SlowChunks()
        mock_stage.return_value = make_stage(Container.Dataset, size=1000)
        mock_conn.query.return_value = chunks.dataset()
        dest = tmp_path / "out.nc"

        # Sample the partial file's allocated size from a separate thread
        # until it is removed: the last growth is when I/O stopped.
        samples: list[tuple[float, int]] = []
        stop_sampling = threading.Event()

        def sample() -> None:
            while not stop_sampling.is_set():
                files = _partial_files(tmp_path)
                size = _disk_bytes(files[0]) if files else -1
                samples.append((time.monotonic(), size))
                time.sleep(0.005)

        # Size seen by the cancel trigger itself. The sampler thread can be
        # starved on a slow runner and miss the pre-cancel window.
        on_disk_at_cancel: list[int] = []

        def started() -> bool:
            # Cancel once chunk data is landing on disk (past the netCDF
            # library's chunk cache).
            for f in _partial_files(tmp_path):
                size = _disk_bytes(f)
                if size >= 2 * CHUNK_BYTES:
                    on_disk_at_cancel.append(size)
                    return True
            return False

        sampler = threading.Thread(target=sample)
        sampler.start()
        try:
            async with Client(server.mcp) as client:
                errors, t_cancel = await _call_and_cancel(
                    client,
                    {"datasource_id": "test-ds", "path": str(dest)},
                    started=started,
                )
                # The tool call returns once the in-flight chunks drain and the
                # partial file is removed.
                await _wait_until(
                    lambda: not list(tmp_path.glob(".*.partial")), MAX_STOP_SECONDS
                )
                t_removed = time.monotonic()
                calls_at_removal = chunks.calls
                await anyio.sleep(0.5)

                # The server survived the cancel and still serves calls.
                assert "export_query" in {t.name for t in await client.list_tools()}
        finally:
            stop_sampling.set()
            sampler.join()

        assert [str(e) for e in errors] == ["Request cancelled"]
        assert not dest.exists()
        assert list(tmp_path.iterdir()) == []  # no partial file or temp dir
        assert _open_fds_under(tmp_path) == []

        # I/O stopped: no chunk started after the partial file was removed,
        # and far fewer than all chunks were ever computed.
        assert chunks.calls == calls_at_removal
        assert chunks.calls < N_CHUNKS // 4
        assert chunks.last_chunk_end - t_cancel < MAX_STOP_SECONDS
        assert on_disk_at_cancel and on_disk_at_cancel[-1] >= CHUNK_BYTES, (
            "no chunk data was on disk before the cancel"
        )
        grew = [t for (t, size), (_, prev) in zip(samples[1:], samples) if size > prev]
        last_growth = max(grew)
        assert last_growth - t_cancel < MAX_STOP_SECONDS
        assert t_removed - t_cancel < MAX_STOP_SECONDS

        # No leaked thread: the pool that ran the chunks was joined.
        assert chunks.threads
        assert not any(t.is_alive() for t in chunks.threads)

        print(
            f"\nOCE-323 cancel->last chunk end {chunks.last_chunk_end - t_cancel:.3f}s, "
            f"cancel->last disk growth {last_growth - t_cancel:.3f}s, "
            f"cancel->partial removed {t_removed - t_cancel:.3f}s, "
            f"chunks computed {chunks.calls}/{N_CHUNKS}"
        )

    async def test_cancel_before_write_writes_nothing(
        self, mock_conn, mock_stage, tmp_path
    ):
        # Cancelled while the (mocked) gateway download is still running.
        release = threading.Event()
        entered = threading.Event()
        frame = MagicMock()

        def slow_query(*args: Any, **kwargs: Any) -> Any:
            entered.set()
            release.wait(5)
            return frame

        mock_stage.return_value = make_stage(Container.DataFrame, size=100)
        mock_conn.query.side_effect = slow_query
        dest = tmp_path / "out.parquet"

        with patch.object(
            server, "_raise_if_cancelled", wraps=server._raise_if_cancelled
        ) as check:
            async with Client(server.mcp) as client:
                errors, _ = await _call_and_cancel(
                    client,
                    {"datasource_id": "test-ds", "path": str(dest)},
                    started=entered.is_set,
                )
                release.set()
                # The tool resumes and stops at its pre-write check.
                await _wait_until(lambda: check.call_count >= 1, 3)
                await client.list_tools()

        assert [str(e) for e in errors] == ["Request cancelled"]
        frame.to_parquet.assert_not_called()
        assert list(tmp_path.iterdir()) == []

    async def test_cancel_during_tabular_write_discards_file(
        self, mock_conn, mock_stage, tmp_path
    ):
        # Tabular writes are not interrupted mid-write, but a cancelled export
        # never lands at dest and its temp file is removed.
        release = threading.Event()
        writing = threading.Event()

        def blocking_write(path: Path) -> None:
            Path(path).write_bytes(b"partial")
            writing.set()
            release.wait(5)

        frame = MagicMock()
        frame.to_parquet.side_effect = blocking_write
        mock_stage.return_value = make_stage(Container.DataFrame, size=100)
        mock_conn.query.return_value = frame
        dest = tmp_path / "out.parquet"

        async with Client(server.mcp) as client:
            errors, _ = await _call_and_cancel(
                client,
                {"datasource_id": "test-ds", "path": str(dest)},
                started=writing.is_set,
            )
            release.set()
            # The temp file written before the cancel is removed.
            await _wait_until(lambda: not list(tmp_path.iterdir()), 3)
            await client.list_tools()

        assert [str(e) for e in errors] == ["Request cancelled"]
        assert not dest.exists()


class TestCancellableNetcdfWriter:
    """The chunked writer keeps the uncancelled and failure behaviour."""

    def test_multi_chunk_dataset_round_trips(self, mock_conn, mock_stage, tmp_path):
        values = np.arange(4 * 3 * 2, dtype="f8").reshape(4, 3, 2)
        ds = xr.Dataset(
            {"v": (("time", "y", "x"), da.from_array(values, chunks=(1, 3, 2)))},
            coords={"time": np.arange(4)},
        )
        mock_stage.return_value = make_stage(Container.Dataset, size=1000)
        mock_conn.query.return_value = ds
        dest = tmp_path / "out.nc"

        parsed = json.loads(
            server.export_query(datasource_id="test-ds", path=str(dest))
        )
        assert parsed["bytes_written"] == dest.stat().st_size
        with xr.open_dataset(dest) as out:
            np.testing.assert_array_equal(out["v"].values, values)
            # Chunked like the dask array, not contiguous: a contiguous
            # variable is pre-filled in full on its first write.
            assert out["v"].encoding["chunksizes"] == (1, 3, 2)
        assert list(tmp_path.iterdir()) == [dest]
        assert _open_fds_under(tmp_path) == []

    def test_chunk_failure_cleans_up_and_joins_pool(
        self, mock_conn, mock_stage, tmp_path, two_workers
    ):
        threads: set[threading.Thread] = set()

        def failing(block: np.ndarray, block_info: Any = None) -> np.ndarray:
            if block.size == 0:  # dask's meta inference, not a chunk
                return block
            threads.add(threading.current_thread())
            if block_info[0]["chunk-location"][0] == 5:
                raise DatameshConnectError("chunk fetch failed")
            time.sleep(0.05)
            return block

        base = da.zeros((20, 64, 64), chunks=(1, 64, 64))
        ds = xr.Dataset({"v": (("t", "y", "x"), base.map_blocks(failing, dtype="f8"))})
        mock_stage.return_value = make_stage(Container.Dataset, size=1000)
        mock_conn.query.return_value = ds
        dest = tmp_path / "out.nc"

        parsed = json.loads(
            server.export_query(datasource_id="test-ds", path=str(dest))
        )
        assert "chunk fetch failed" in parsed["error"]
        assert list(tmp_path.iterdir()) == []
        assert _open_fds_under(tmp_path) == []
        assert not any(t.is_alive() for t in threads)
