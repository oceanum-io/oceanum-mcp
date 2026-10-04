"""OCE-294 MCP mitigations and OCE-326.

OCE-294: the Datamesh engine lower-cases time_resolution, so "1MS" (month
start) becomes "1ms" (1 millisecond) and the stage hangs. The MCP side:
- no longer advertises calendar aliases in its docs,
- refuses resolutions the engine mangles, or finer than a minute,
- bounds staging/query requests with OCEANUM_MCP_STAGE_TIMEOUT.

OCE-326: the stdio stage_query recommendation accounts for a client-side
limit, and _limit_result limits dims that have no coordinate variable.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests
import xarray as xr
from fastmcp.exceptions import ToolError
from oceanum.datamesh import Connector
from oceanum.datamesh.query import Container
from oceanum.datamesh.utils import HTTPSession

from oceanum_mcp.common.config import (
    DEFAULT_STAGE_TIMEOUT_S,
    set_transport,
    stage_timeout,
)
from oceanum_mcp.servers.datamesh import server
from tests.conftest import make_stage

RANGE = {"time_start": "2023-01-01", "time_end": "2024-01-01"}


# ---------------------------------------------------------------------------
# Docs no longer advertise calendar aliases
# ---------------------------------------------------------------------------


class TestDocs:
    async def test_query_tool_docs_do_not_advertise_1ms(self):
        tools = {t.name: t for t in await server.mcp.list_tools()}
        for name in ("stage_query", "query_data", "export_query"):
            desc = tools[name].description
            # The Args text must land in the parameter schema on every Python
            # version (OCE-331: it stayed in the description on 3.13+).
            line = tools[name].parameters["properties"]["time_resolution"][
                "description"
            ]
            assert "1MS" not in desc and "1MS" not in line
            assert '"30D"' in line
            # Calendar aliases appear only as a caveat, not as an example.
            assert "not supported" in line


# ---------------------------------------------------------------------------
# time_resolution guard
# ---------------------------------------------------------------------------


class TestTimeResolutionGuard:
    @pytest.mark.parametrize(
        "resolution", ["1MS", "1ME", "1QS", "1YS", "2MS", "3MS", "1BMS"]
    )
    def test_rejects_engine_mangled_calendar_aliases(self, resolution):
        with pytest.raises(ToolError, match="30D") as exc:
            server._build_query("test-ds", time_resolution=resolution, **RANGE)
        assert "lower-cases" in str(exc.value)

    @pytest.mark.parametrize("resolution", ["1ms", "1s", "30s", "1us", "500ms"])
    def test_rejects_finer_than_a_minute(self, resolution):
        with pytest.raises(ToolError, match="finer than one minute"):
            server._build_query("test-ds", time_resolution=resolution, **RANGE)

    @pytest.mark.parametrize(
        "resolution",
        ["1h", "6h", "1D", "1d", "7D", "30D", "90D", "365D", "1min", "15min"]
        + ["1W", "W-MON", "1h30min", "1H", "native"],
    )
    def test_accepts_case_insensitive_resolutions(self, resolution):
        query = server._build_query("test-ds", time_resolution=resolution, **RANGE)
        assert query.timefilter.resolution == resolution

    # Legacy aliases pandas 3 no longer parses (so they used to be passed
    # through on pandas-3 hosts) are refused for the same reason on every
    # pandas version.
    @pytest.mark.parametrize(
        "resolution",
        ["1M", "1m", "1Y", "1y", "1A", "1AS", "1as", "1Q", "1q", "Q-DEC"]
        + ["A-JAN", "2BQ", "1BA", "1SM", "1MS", "1Ms", "1ME", "1QS", "1YS"],
    )
    def test_legacy_calendar_aliases_denied_on_any_pandas(self, resolution):
        with pytest.raises(ToolError, match="not supported yet") as exc:
            server._build_query("test-ds", time_resolution=resolution, **RANGE)
        assert '"30D"' in str(exc.value)

    @pytest.mark.parametrize(
        "resolution",
        ["1L", "1l", "1U", "1u", "1N", "1n", "1S", "5S", "1ms", "500ms", "1NS"],
    )
    def test_legacy_sub_minute_aliases_denied_on_any_pandas(self, resolution):
        with pytest.raises(ToolError, match="finer than one minute"):
            server._build_query("test-ds", time_resolution=resolution, **RANGE)

    def test_surrounding_whitespace_stripped(self):
        query = server._build_query("test-ds", time_resolution=" 1D ", **RANGE)
        assert query.timefilter.resolution == "1D"
        with pytest.raises(ToolError, match="not supported yet"):
            server._build_query("test-ds", time_resolution=" 1MS ", **RANGE)
        with pytest.raises(ToolError, match="finer than one minute"):
            server._build_query("test-ds", time_resolution="\t1L\n", **RANGE)

    @pytest.mark.parametrize("resolution", ["0D", "-1D", "0h"])
    def test_rejects_non_positive(self, resolution):
        with pytest.raises(ToolError, match="positive"):
            server._build_query("test-ds", time_resolution=resolution, **RANGE)

    def test_does_not_wait_behind_a_query_holding_the_warnings_lock(self):
        # _captured_warnings holds _WARNINGS_LOCK for a whole query; parameter
        # validation in another call must not block on it.
        result: list[Any] = []
        with server._WARNINGS_LOCK:
            worker = threading.Thread(
                target=lambda: result.append(
                    server._build_query("test-ds", time_resolution="1D", **RANGE)
                ),
                daemon=True,
            )
            worker.start()
            worker.join(5)
        assert result, "time_resolution validation blocked on _WARNINGS_LOCK"

    def test_unparsable_left_to_the_engine(self):
        # Not a pandas alias here: the engine validates it, as before.
        query = server._build_query("test-ds", time_resolution="bogus", **RANGE)
        assert query.timefilter.resolution == "bogus"

    def test_tool_rejects_before_staging(self, mock_conn, mock_stage):
        with pytest.raises(ToolError, match="30D"):
            server.query_data(datasource_id="test-ds", time_resolution="1MS", **RANGE)
        mock_stage.assert_not_called()
        mock_conn.query.assert_not_called()


# ---------------------------------------------------------------------------
# OCEANUM_MCP_STAGE_TIMEOUT
# ---------------------------------------------------------------------------


class TestStageTimeoutConfig:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("OCEANUM_MCP_STAGE_TIMEOUT", raising=False)
        assert stage_timeout() == DEFAULT_STAGE_TIMEOUT_S == 120.0

    def test_override(self, monkeypatch):
        monkeypatch.setenv("OCEANUM_MCP_STAGE_TIMEOUT", "2.5")
        assert stage_timeout() == 2.5

    @pytest.mark.parametrize("raw", ["abc", "0", "-5", "nan", "inf"])
    def test_invalid_fails_fast(self, monkeypatch, raw):
        monkeypatch.setenv("OCEANUM_MCP_STAGE_TIMEOUT", raw)
        with pytest.raises(ValueError, match="OCEANUM_MCP_STAGE_TIMEOUT"):
            stage_timeout()


class _RecordingSession:
    """Stands in for the SDK's HTTPSession; records each request's timeout."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc
        self.timeouts: list[Any] = []

    def request(self, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
        self.timeouts.append(kwargs.get("timeout"))
        if self.exc is not None:
            raise self.exc
        resp = MagicMock()
        resp.status_code = 204
        return resp


def _sdk_connector(http_session: Any, gateway: str = "http://gateway") -> Connector:
    """A real Connector (real _stage_request/_retried_request, so the SDK's
    retry loop runs) without the network round trip __init__ makes."""
    conn = object.__new__(Connector)
    conn._gateway = gateway
    conn._verify = True
    conn.http_session = http_session
    return conn


@pytest.fixture
def sdk_conn() -> Iterator[Any]:
    """Patch in a factory for real-SDK connectors; Session is mocked."""
    holder: dict[str, Any] = {}

    def use(http_session: Any, **kwargs: Any) -> Connector:
        holder["conn"] = _sdk_connector(http_session, **kwargs)
        return holder["conn"]

    with (
        patch.object(
            server, "get_datamesh_connector", side_effect=lambda: holder["conn"]
        ),
        patch.object(server, "Session") as session,
    ):
        session.acquire.return_value.header = {}
        yield use


class TestStageTimeout:
    def test_stage_timeout_returns_clear_error_without_retries(
        self, sdk_conn, monkeypatch
    ):
        monkeypatch.setenv("OCEANUM_MCP_STAGE_TIMEOUT", "0.5")
        http = _RecordingSession(exc=requests.ReadTimeout("read timed out"))
        sdk_conn(http)

        start = time.monotonic()
        out = json.loads(
            server.stage_query(datasource_id="test-ds", time_resolution="1D", **RANGE)
        )
        assert time.monotonic() - start < 1  # no SDK backoff/retries
        assert "staging timed out after 0.5s" in out["error"]
        assert "coarser time_resolution" in out["error"]
        assert "retrying once may succeed" in out["error"]
        assert "OCEANUM_MCP_STAGE_TIMEOUT" in out["error"]
        assert out["query"]["datasource"] == "test-ds"
        # One attempt, its 900 s SDK read timeout capped at the 0.5 s budget.
        assert len(http.timeouts) == 1
        assert 0 < http.timeouts[0][1] <= 0.5

    def test_body_read_timeout_also_converted(self, sdk_conn, monkeypatch):
        # requests raises a ConnectionError wrapping urllib3's
        # ReadTimeoutError when the body read times out.
        from urllib3.exceptions import ReadTimeoutError

        monkeypatch.setenv("OCEANUM_MCP_STAGE_TIMEOUT", "0.5")
        http = _RecordingSession(
            exc=requests.ConnectionError(ReadTimeoutError(None, "/", "timed out"))
        )
        sdk_conn(http)
        out = json.loads(server.stage_query(datasource_id="test-ds"))
        assert "timed out after 0.5s" in out["error"]
        assert len(http.timeouts) == 1

    def test_other_request_errors_keep_sdk_retries(self, sdk_conn, monkeypatch):
        monkeypatch.setenv("OCEANUM_MCP_STAGE_TIMEOUT", "0.5")
        http = _RecordingSession(exc=requests.ConnectionError("refused"))
        sdk_conn(http)
        with patch("oceanum.datamesh.utils.sleep"):
            out = json.loads(server.stage_query(datasource_id="test-ds"))
        assert "after 8 retries" in out["error"]
        assert len(http.timeouts) == 8

    def test_cap_only_applies_inside_a_tool_call(self, sdk_conn, monkeypatch):
        monkeypatch.setenv("OCEANUM_MCP_STAGE_TIMEOUT", "0.5")
        http = _RecordingSession()
        conn = sdk_conn(http)
        server.stage_query(datasource_id="test-ds")
        assert 0 < http.timeouts[-1][1] <= 0.5
        # The wrapped connector is shared: outside a call nothing is capped.
        conn._retried_request("http://gateway/x", timeout=(3.05, 900))
        assert http.timeouts[-1] == (3.05, 900)

    def test_tighter_sdk_timeout_left_alone(self, sdk_conn, monkeypatch):
        monkeypatch.setenv("OCEANUM_MCP_STAGE_TIMEOUT", "60")
        http = _RecordingSession()
        conn = sdk_conn(http)
        with server._gateway_timeout(conn, "staging"):
            conn._retried_request("http://gateway/x", timeout=(3.05, 10))
        assert http.timeouts == [(3.05, 10)]

    def test_bad_gateway_retries_stop_at_the_deadline(self, sdk_conn, monkeypatch):
        # The SDK sleeps 30 s after each 502 and retries, up to 8 times; no
        # read timeout ever fires. The shared deadline still ends the call.
        monkeypatch.setenv("OCEANUM_MCP_STAGE_TIMEOUT", "0.3")
        bad_gateway = MagicMock(status_code=502)

        class _BadGateway(_RecordingSession):
            def request(self, method: str, url: str, *a: Any, **kw: Any) -> Any:
                super().request(method, url, *a, **kw)
                return bad_gateway

        http = _BadGateway()
        sdk_conn(http)
        with patch(
            "oceanum.datamesh.utils.sleep", side_effect=lambda s: time.sleep(0.1)
        ):
            start = time.monotonic()
            out = json.loads(server.stage_query(datasource_id="test-ds"))
        assert "staging timed out after 0.3s" in out["error"]
        assert time.monotonic() - start < 2
        assert 1 <= len(http.timeouts) < 8

    def test_wrapped_session_survives_copy_and_pickle(self):
        import copy
        import pickle

        wrapped = server._CappedHTTPSession(HTTPSession(headers={"a": "b"}))
        for clone in (copy.copy(wrapped), pickle.loads(pickle.dumps(wrapped))):
            assert isinstance(clone, server._CappedHTTPSession)
            assert clone._headers == {"a": "b"}

    def test_connector_wrapped_once(self, sdk_conn):
        conn = sdk_conn(_RecordingSession())
        server.stage_query(datasource_id="test-ds")
        wrapped = conn.http_session
        server.stage_query(datasource_id="test-ds")
        assert conn.http_session is wrapped
        assert isinstance(wrapped, server._CappedHTTPSession)

    def test_hosted_export_download_stage_bounded(self, sdk_conn, monkeypatch):
        monkeypatch.setenv("OCEANUM_MCP_STAGE_TIMEOUT", "0.5")
        http = _RecordingSession(exc=requests.ReadTimeout("read timed out"))
        sdk_conn(http)
        try:
            set_transport("http")
            out = json.loads(server.export_query(datasource_id="test-ds"))
        finally:
            set_transport("stdio")
        assert "download staging timed out after 0.5s" in out["error"]
        assert len(http.timeouts) == 1


class _SilentGateway:
    """A TCP server that accepts a request and never answers, recording
    whether the client closed its end of the socket."""

    def __init__(self) -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen()
        self.connections = 0
        self.client_closed = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.sock.getsockname()[1]}"

    def _serve(self) -> None:
        self.sock.settimeout(10)
        try:
            client, _ = self.sock.accept()
        except OSError:
            return
        self.connections += 1
        client.settimeout(10)
        with client:
            try:
                while client.recv(65536):  # b"" once the client closes
                    pass
                self.client_closed.set()
            except OSError:
                pass

    def close(self) -> None:
        self.sock.close()


class TestStageTimeoutOnRealSocket:
    def test_silent_gateway_times_out_and_socket_is_closed(self, sdk_conn, monkeypatch):
        monkeypatch.setenv("OCEANUM_MCP_STAGE_TIMEOUT", "0.3")
        gateway = _SilentGateway()
        try:
            sdk_conn(HTTPSession(), gateway=gateway.url)
            start = time.monotonic()
            out = json.loads(server.stage_query(datasource_id="test-ds"))
            elapsed = time.monotonic() - start
            # Ends after ~0.3 s, not the SDK's 900 s x 8 attempts.
            assert "staging timed out after 0.3s" in out["error"]
            assert elapsed < 5
            assert gateway.client_closed.wait(5), "client socket left open"
            assert gateway.connections == 1
        finally:
            gateway.close()


class TestQueryAndLoadBounded:
    def _timeout_on_request(self, conn: MagicMock) -> None:
        conn.http_session = _RecordingSession(exc=requests.ReadTimeout("slow"))

        def query(*args: Any, **kwargs: Any) -> Any:
            # What the SDK does inside conn.query: a request on its session.
            return conn.http_session.request("POST", "u", timeout=(3.05, 900))

        conn.query.side_effect = query
        conn.load_datasource.side_effect = query

    def test_query_data_query_bounded(self, mock_conn, mock_stage, monkeypatch):
        monkeypatch.setenv("OCEANUM_MCP_STAGE_TIMEOUT", "0.5")
        mock_stage.return_value = make_stage(Container.DataFrame, size=100)
        self._timeout_on_request(mock_conn)
        out = json.loads(server.query_data(datasource_id="test-ds"))
        assert "query timed out after 0.5s" in out["error"]
        assert out["query"]["datasource"] == "test-ds"

    def test_load_datasource_bounded(self, mock_conn, mock_stage, monkeypatch):
        monkeypatch.setenv("OCEANUM_MCP_STAGE_TIMEOUT", "0.5")
        self._timeout_on_request(mock_conn)
        out = json.loads(server.load_datasource(datasource_id="test-ds"))
        assert "load timed out after 0.5s" in out["error"]
        assert "query_data" in out["error"]

    def _cap_during_query(self, conn: MagicMock, data: Any) -> list[Any]:
        seen: list[Any] = []

        def query(*args: Any, **kwargs: Any) -> Any:
            seen.append(server._TIMEOUT_BUDGET.get())
            return data

        conn.query.side_effect = query
        return seen

    def test_local_dataset_export_query_bounded(self, mock_conn, mock_stage, tmp_path):
        mock_stage.return_value = make_stage(Container.Dataset, size=100)
        ds = xr.Dataset({"hs": ("time", np.arange(3.0))})
        seen = self._cap_during_query(mock_conn, ds)
        server.export_query(datasource_id="test-ds", path=str(tmp_path / "o.nc"))
        assert seen and seen[0] is not None and "query timed out" in seen[0].message

    def test_local_frame_export_download_not_capped(
        self, mock_conn, mock_stage, tmp_path
    ):
        # A frame export downloads the whole result: SDK timeouts apply.
        mock_stage.return_value = make_stage(Container.DataFrame, size=100)
        seen = self._cap_during_query(mock_conn, pd.DataFrame({"x": [1, 2]}))
        server.export_query(datasource_id="test-ds", path=str(tmp_path / "o.parquet"))
        assert seen == [None]


# ---------------------------------------------------------------------------
# OCE-326
# ---------------------------------------------------------------------------


def _stage_with_coordkeys(container: Container, size: int, coordkeys: dict):
    return make_stage(container, size=size).model_copy(update={"coordkeys": coordkeys})


class TestStdioRecommendationHonoursLimit:
    def test_limited_dataset_above_local_cap_not_said_refused(
        self, mock_conn, mock_stage
    ):
        mock_stage.return_value = _stage_with_coordkeys(
            Container.Dataset, 73_000_000_000, {"t": "time"}
        )
        out = json.loads(
            server.stage_query(
                datasource_id="test-ds", time_resolution="1D", limit=2, **RANGE
            )
        )
        rec = out["recommendation"]
        assert "refuse" not in rec
        assert "last 2 time/ensemble steps" in rec
        assert "73.0 GB" in rec
        assert rec.index("Narrow") < rec.index("export_query")

    def test_limited_frame_above_local_cap_still_refused(self, mock_conn, mock_stage):
        # A frame downloads in full before the limit applies, so export_query
        # does refuse it on the unlimited size.
        mock_stage.return_value = make_stage(Container.DataFrame, size=3_000_000_000)
        rec = json.loads(
            server.stage_query(
                datasource_id="test-ds", aggregate_operations=["mean"], limit=2
            )
        )["recommendation"]
        assert "refuse" in rec

    def test_unlimited_dataset_above_local_cap_still_refused(
        self, mock_conn, mock_stage
    ):
        mock_stage.return_value = make_stage(Container.Dataset, size=73_000_000_000)
        rec = json.loads(
            server.stage_query(datasource_id="test-ds", time_resolution="1D", **RANGE)
        )["recommendation"]
        assert "refuse" in rec


class TestLimitDimWithoutCoordinate:
    def _ensemble(self) -> xr.Dataset:
        # "member" is a dimension with no coordinate variable.
        return xr.Dataset(
            {"hs": (("member", "time"), np.arange(12.0).reshape(4, 3))},
            coords={"time": pd.date_range("2024-01-01", periods=3, freq="D")},
        )

    def test_limit_result_limits_dim_named_in_coordkeys(self):
        data, note = server._limit_result(
            self._ensemble(), 2, {"t": "time", "e": "member"}
        )
        assert dict(data.sizes) == {"member": 2, "time": 2}
        assert "member" in note and "time" in note

    def test_query_data_limits_coordless_ensemble_dim(self, mock_conn, mock_stage):
        mock_stage.return_value = _stage_with_coordkeys(
            Container.Dataset, 100, {"t": "time", "e": "member"}
        )
        mock_conn.query.return_value = self._ensemble()
        out = json.loads(
            server.query_data(
                datasource_id="test-ds", time_resolution="1D", limit=2, **RANGE
            )
        )
        assert out["dims"] == {"member": 2, "time": 2}
