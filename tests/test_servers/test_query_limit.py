"""query_data `limit` with resampling/aggregation (OCE-298 MCP mitigation).

Datamesh applies `limit` to native records before time_resolution resampling
or aggregation, so query_data must not send it with those; it applies the
limit to the returned (resampled/aggregated) result instead.
"""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fastmcp.exceptions import ToolError
from oceanum.datamesh.exceptions import DatameshConnectError
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
        # The backend applied it; the result is passed through untouched,
        # but flagged as a subset whose full size is unknown.
        assert parsed["dims"] == {"time": 6}
        assert parsed["preview"] is True
        assert parsed["total"] is None
        assert "Datamesh" in parsed["limit_note"]
        assert "NOT suitable for statistics" in parsed["limit_note"]

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

    def test_client_limited_result_flagged_as_preview(self, mock_conn, mock_stage):
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
        assert parsed["preview"] is True
        assert parsed["returned"] == 2
        assert parsed["total"] == 6  # the resampled result before limiting
        assert "NOT suitable for statistics" in parsed["limit_note"]

    def test_limit_not_cutting_anything_is_not_a_preview(self, mock_conn, mock_stage):
        mock_stage.return_value = _dataset_stage({"t": "time"})
        mock_conn.query.return_value = _hourly(3)

        parsed = json.loads(
            server.query_data(
                datasource_id="test-ds",
                time_start="2024-01-01",
                time_end="2024-01-04",
                time_resolution="1D",
                limit=10,
            )
        )
        assert parsed["preview"] is False
        assert parsed["total"] == 3

    def test_echoed_query_keeps_callers_limit(self, mock_conn, mock_stage):
        mock_stage.return_value = _dataset_stage({"t": "time"})
        mock_conn.query.side_effect = DatameshConnectError("server error: 500")

        parsed = json.loads(
            server.query_data(
                datasource_id="test-ds",
                time_start="2024-01-01",
                time_end="2024-01-07",
                time_resolution="1D",
                limit=2,
            )
        )
        assert "error" in parsed
        assert parsed["query"]["limit"] == 2

    def test_large_lazy_dataset_loaded_once_limited_small(self, mock_conn, mock_stage):
        stage = make_stage(Container.Dataset, size=10**9)
        mock_stage.return_value = stage.model_copy(update={"coordkeys": {"t": "time"}})
        mock_conn.query.return_value = _hourly(6).chunk({"time": 1})

        parsed = json.loads(
            server.query_data(
                datasource_id="test-ds",
                time_start="2024-01-01",
                time_end="2024-01-07",
                time_resolution="1D",
                limit=2,
            )
        )
        assert mock_conn.query.call_args.kwargs["use_dask"] is True
        assert parsed["lazy"] is False
        assert [r["hs"] for r in parsed["data"]] == [4.0, 5.0]

    def test_large_frame_refusal_explains_limit(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(Container.DataFrame, size=10**9)

        parsed = json.loads(
            server.query_data(
                datasource_id="test-ds", aggregate_operations=["mean"], limit=3
            )
        )
        assert parsed["refused"] is True
        assert "cannot shrink" in parsed["limit_note"]
        assert parsed["query"]["limit"] == 3
        mock_conn.query.assert_not_called()

    def test_native_time_resolution_keeps_backend_limit(self, mock_conn, mock_stage):
        mock_stage.return_value = _dataset_stage({"t": "time"})
        mock_conn.query.return_value = _hourly(6)

        server.query_data(
            datasource_id="test-ds",
            time_start="2024-01-01",
            time_end="2024-01-07",
            time_resolution="native",
            limit=2,
        )
        assert _sent_query().limit == 2


class TestStageQueryLimit:
    """OCE-319: stage_query sizes what query_data/export_query actually fetch."""

    def test_limit_not_staged_with_time_resolution(self, mock_conn, mock_stage):
        mock_stage.return_value = _dataset_stage({"t": "time"})

        parsed = json.loads(
            server.stage_query(
                datasource_id="test-ds",
                time_start="2024-01-01",
                time_end="2024-01-07",
                time_resolution="1D",
                limit=2,
            )
        )
        assert mock_stage.call_args.args[1].limit is None
        assert mock_stage.call_args.args[1].timefilter.resolution == "1D"
        assert "upper bound" in parsed["limit_note"]
        assert "after" in parsed["limit_note"].lower()
        # The echoed query keeps the caller's limit.
        assert parsed["query"]["limit"] == 2

    def test_limit_not_staged_with_aggregation(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(Container.DataFrame, size=100)

        parsed = json.loads(
            server.stage_query(
                datasource_id="test-ds", aggregate_operations=["mean"], limit=3
            )
        )
        assert mock_stage.call_args.args[1].limit is None
        assert "limit_note" in parsed

    def test_plain_limit_still_staged(self, mock_conn, mock_stage):
        mock_stage.return_value = _dataset_stage({"t": "time"})

        parsed = json.loads(server.stage_query(datasource_id="test-ds", limit=5))
        assert mock_stage.call_args.args[1].limit == 5
        assert "limit_note" not in parsed

    def test_hosted_note_says_export_refuses_combination(self, mock_conn, mock_stage):
        from oceanum_mcp.common.config import set_transport

        mock_stage.return_value = _dataset_stage({"t": "time"})
        try:
            set_transport("http")
            parsed = json.loads(
                server.stage_query(
                    datasource_id="test-ds",
                    time_start="2024-01-01",
                    time_end="2024-01-07",
                    time_resolution="1D",
                    limit=2,
                )
            )
        finally:
            set_transport("stdio")
        assert "export_query" in parsed["limit_note"]
        assert "not supported" in parsed["limit_note"]


class TestExportQueryLimit:
    """OCE-319: export_query applies limit after resampling, never before."""

    def test_local_export_does_not_send_limit(self, mock_conn, mock_stage, tmp_path):
        mock_stage.return_value = _dataset_stage({"t": "time"})
        mock_conn.query.return_value = _hourly(6)

        server.export_query(
            datasource_id="test-ds",
            path=str(tmp_path / "out.nc"),
            time_start="2024-01-01",
            time_end="2024-01-07",
            time_resolution="1D",
            limit=2,
        )
        assert _sent_query().limit is None
        assert _sent_query().timefilter.resolution == "1D"
        assert mock_stage.call_args.args[1].limit is None

    def test_local_export_writes_last_n_resampled_steps(
        self, mock_conn, mock_stage, tmp_path
    ):
        mock_stage.return_value = _dataset_stage({"t": "time"})
        mock_conn.query.return_value = _hourly(6)
        dest = tmp_path / "out.nc"

        parsed = json.loads(
            server.export_query(
                datasource_id="test-ds",
                path=str(dest),
                time_start="2024-01-01",
                time_end="2024-01-07",
                time_resolution="1D",
                limit=2,
            )
        )
        with xr.open_dataset(dest) as ds:
            assert ds["hs"].values.tolist() == [4.0, 5.0]
        assert parsed["summary"]["dims"] == {"time": 2}
        assert "after" in parsed["limit_note"].lower()

    def test_local_export_limits_aggregated_frame(
        self, mock_conn, mock_stage, tmp_path
    ):
        mock_stage.return_value = make_stage(Container.DataFrame, size=100)
        mock_conn.query.return_value = pd.DataFrame({"x": range(10)})
        dest = tmp_path / "out.csv"

        server.export_query(
            datasource_id="test-ds",
            path=str(dest),
            format="csv",
            aggregate_operations=["mean"],
            limit=3,
        )
        assert _sent_query().limit is None
        assert pd.read_csv(dest)["x"].tolist() == [7, 8, 9]

    def test_local_export_limits_before_flattening(
        self, mock_conn, mock_stage, tmp_path
    ):
        mock_stage.return_value = _dataset_stage({"t": "time"})
        mock_conn.query.return_value = _hourly(6)
        dest = tmp_path / "out.csv"

        server.export_query(
            datasource_id="test-ds",
            path=str(dest),
            format="csv",
            time_start="2024-01-01",
            time_end="2024-01-07",
            time_resolution="1D",
            limit=2,
        )
        assert pd.read_csv(dest)["hs"].tolist() == [4.0, 5.0]

    def test_plain_limit_still_sent_on_export(self, mock_conn, mock_stage, tmp_path):
        mock_stage.return_value = _dataset_stage({"t": "time"})
        mock_conn.query.return_value = _hourly(6)

        parsed = json.loads(
            server.export_query(
                datasource_id="test-ds", path=str(tmp_path / "out.nc"), limit=5
            )
        )
        assert _sent_query().limit == 5
        assert "limit_note" not in parsed

    def test_hosted_export_refuses_limit_with_resampling(self, mock_conn):
        from oceanum_mcp.common.config import set_transport

        try:
            set_transport("http")
            with patch.object(server, "_download_stage") as dl:
                with pytest.raises(ToolError, match="drop limit"):
                    server.export_query(
                        datasource_id="test-ds",
                        time_start="2024-01-01",
                        time_end="2024-01-07",
                        time_resolution="1D",
                        limit=2,
                    )
                with pytest.raises(ToolError, match="not supported for hosted"):
                    server.export_query(
                        datasource_id="test-ds",
                        aggregate_operations=["mean"],
                        limit=2,
                    )
        finally:
            set_transport("stdio")
        # Refused before any gateway download link is minted.
        dl.assert_not_called()

    def test_hosted_export_plain_limit_still_allowed(self, mock_conn):
        from oceanum_mcp.common.config import set_transport

        stage = {
            "container": "dataframe",
            "size": 10,
            "formats": ["parquet"],
            "url": "https://gw.test/x?sig=1",
        }
        try:
            set_transport("http")
            with patch.object(server, "_download_stage", return_value=stage) as dl:
                parsed = json.loads(
                    server.export_query(datasource_id="test-ds", limit=2)
                )
        finally:
            set_transport("stdio")
        assert "download_url" in parsed
        assert dl.call_args.args[1].limit == 2

    async def test_all_query_tools_document_same_limit_semantics(self):
        tools = {t.name: t for t in await server.mcp.list_tools()}
        # FastMCP keeps the Args section in the tool description.
        docs = [
            next(
                line.strip()
                for line in tools[name].description.splitlines()
                if line.strip().startswith("limit:")
            )
            for name in ("stage_query", "query_data", "export_query")
        ]
        assert docs[0] == docs[1] == docs[2]
        desc = docs[0].lower()
        assert "last" in desc and "after" in desc
        assert "time_resolution" in desc and "aggregate_operations" in desc
        assert "hosted" in desc
