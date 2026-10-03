"""query_data `limit` with resampling/aggregation (OCE-298 MCP mitigation).

Datamesh applies `limit` to native records before time_resolution resampling
or aggregation, so query_data must not send it with those; it applies the
limit to the returned (resampled/aggregated) result instead.
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fastmcp.exceptions import ToolError
from oceanum.datamesh.query import Container

from oceanum_mcp.servers.datamesh import server
from tests.conftest import make_stage


def _hourly(n: int = 6) -> xr.Dataset:
    return xr.Dataset(
        {"hs": (("time",), np.arange(n, dtype=float))},
        coords={"time": pd.date_range("2024-01-01", periods=n, freq="D")},
    )


def _sent_query():
    return server.get_datamesh_connector().query.call_args.args[0]


def _dataset_stage(coordkeys: dict[str, str] | None = None):
    stage = make_stage(Container.Dataset, size=100)
    return stage.model_copy(update={"coordkeys": coordkeys or {}})


class TestLimitWithResampling:
    def test_limit_not_sent_with_time_resolution(self, mock_conn, mock_stage):
        mock_stage.return_value = _dataset_stage({"t": "time"})
        mock_conn.query.return_value = _hourly(6)

        server.query_data(
            datasource_id="test-ds",
            time_start="2024-01-01",
            time_end="2024-01-07",
            time_resolution="1D",
            limit=2,
        )
        assert _sent_query().limit is None
        # The size check must also see the un-limited (resampled) query.
        assert mock_stage.call_args.args[1].limit is None
        # Resampling itself is still requested from the backend.
        assert _sent_query().timefilter.resolution == "1D"

    def test_limit_applied_to_resampled_result(self, mock_conn, mock_stage):
        mock_stage.return_value = _dataset_stage({"t": "time"})
        mock_conn.query.return_value = _hourly(6)

        parsed = json.loads(
            server.query_data(
                datasource_id="test-ds",
                time_start="2024-01-01",
                time_end="2024-01-07",
                time_resolution="1D",
                limit=2,
            )
        )
        # Datamesh limit semantics: the last N steps along time.
        assert parsed["dims"] == {"time": 2}
        assert [r["hs"] for r in parsed["data"]] == [4.0, 5.0]
        assert "after" in parsed["limit_note"].lower()
        assert "resampl" in parsed["limit_note"]

    def test_datetime_dim_fallback_without_coordkeys(self, mock_conn, mock_stage):
        mock_stage.return_value = _dataset_stage()
        mock_conn.query.return_value = _hourly(6)

        parsed = json.loads(
            server.query_data(
                datasource_id="test-ds",
                time_start="2024-01-01",
                time_end="2024-01-07",
                time_resolution="1D",
                limit=3,
            )
        )
        assert parsed["dims"] == {"time": 3}

    def test_limit_applied_to_aggregated_frame(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(Container.DataFrame, size=100)
        mock_conn.query.return_value = pd.DataFrame({"x": range(10)})

        parsed = json.loads(
            server.query_data(
                datasource_id="test-ds", aggregate_operations=["mean"], limit=3
            )
        )
        assert _sent_query().limit is None
        assert _sent_query().aggregate is not None
        assert parsed["rows"] == 3
        assert [r["x"] for r in parsed["data"]] == [7, 8, 9]
        assert "limit_note" in parsed

    def test_limit_without_time_dim_reports_no_effect(self, mock_conn, mock_stage):
        # Temporal aggregation removes the time dimension entirely.
        mock_stage.return_value = _dataset_stage({"t": "time"})
        mock_conn.query.return_value = xr.Dataset(
            {"hs": (("station",), np.array([1.0, 2.0, 3.0, 4.0]))}
        )

        parsed = json.loads(
            server.query_data(
                datasource_id="test-ds", aggregate_operations=["mean"], limit=3
            )
        )
        assert _sent_query().limit is None
        assert parsed["dims"] == {"station": 4}
        assert "no effect" in parsed["limit_note"]

    def test_plain_limit_still_sent_to_backend(self, mock_conn, mock_stage):
        mock_stage.return_value = _dataset_stage({"t": "time"})
        mock_conn.query.return_value = _hourly(6)

        parsed = json.loads(server.query_data(datasource_id="test-ds", limit=5))
        assert _sent_query().limit == 5
        assert mock_stage.call_args.args[1].limit == 5
        # The backend applied it; the result is passed through untouched.
        assert parsed["dims"] == {"time": 6}
        assert "limit_note" not in parsed

    def test_no_limit_no_note(self, mock_conn, mock_stage):
        mock_stage.return_value = _dataset_stage({"t": "time"})
        mock_conn.query.return_value = _hourly(6)

        parsed = json.loads(
            server.query_data(
                datasource_id="test-ds",
                time_start="2024-01-01",
                time_end="2024-01-07",
                time_resolution="1D",
            )
        )
        assert parsed["dims"] == {"time": 6}
        assert "limit_note" not in parsed

    def test_nonpositive_limit_still_rejected(self, mock_conn, mock_stage):
        with pytest.raises(ToolError, match="at least 1"):
            server.query_data(
                datasource_id="test-ds", aggregate_operations=["mean"], limit=0
            )

    async def test_docstring_documents_client_side_limit(self):
        tools = {t.name: t for t in await server.mcp.list_tools()}
        desc = tools["query_data"].description.lower()
        assert "time_resolution" in desc and "aggregate_operations" in desc
        assert "after" in desc and "last" in desc
