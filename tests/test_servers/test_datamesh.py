"""Tests for the Datamesh MCP server."""

import importlib
import json
import warnings
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fastmcp.exceptions import ToolError

from oceanum.datamesh.exceptions import (
    DatameshConnectError,
    DatameshQueryError,
    DatameshSessionError,
)
from oceanum.datamesh.query import Container, CoordSelector, GeoFilter

import oceanum_mcp.servers.datamesh.server as server
from tests.conftest import make_stage


def _mock_datasource(**kwargs):
    """Create a mock Datasource object."""
    ds = MagicMock()
    ds.id = kwargs.get("id", "test-ds")
    ds.name = kwargs.get("name", "Test Dataset")
    ds.description = kwargs.get("description", "A test dataset")
    ds.geom = kwargs.get("geom", None)
    ds.bounds = kwargs.get("bounds", None)
    ds.tstart = kwargs.get("tstart", None)
    ds.tend = kwargs.get("tend", None)
    ds.tags = kwargs.get("tags", [])
    ds.labels = kwargs.get("labels", [])
    ds.info = kwargs.get("info", None)
    ds.coordinates = kwargs.get("coordinates", {})
    ds.variables = kwargs.get("variables", None)
    ds.attributes = kwargs.get("attributes", None)
    ds.dataschema = kwargs.get("dataschema", None)
    ds.driver = kwargs.get("driver", "zarr")
    ds.details = kwargs.get("details", None)
    ds.modified = kwargs.get("modified", None)
    ds.created = kwargs.get("created", None)
    return ds


def _mock_catalog(datasources):
    """Create a mock Catalog object."""
    catalog = MagicMock()
    catalog.__len__ = MagicMock(return_value=len(datasources))
    catalog.__iter__ = MagicMock(return_value=iter(datasources))
    return catalog


def _small_dataset() -> xr.Dataset:
    return xr.Dataset(
        {"hs": (("time",), np.array([1.0, 2.0, 3.0]))},
        coords={"time": pd.date_range("2024-01-01", periods=3, freq="h")},
    )


class TestSearchCatalog:
    def test_returns_results_with_count(self, mock_conn):
        ds = _mock_datasource(id="era5-waves", name="ERA5 Waves", tags=["wave"])
        mock_conn.get_catalog.return_value = _mock_catalog([ds])

        parsed = json.loads(server.search_catalog(search="wave"))
        assert parsed["count"] == 1
        assert parsed["results"][0]["id"] == "era5-waves"
        assert mock_conn.get_catalog.call_args.kwargs["limit"] == 20

    def test_no_sentinel_times(self, mock_conn):
        mock_conn.get_catalog.return_value = _mock_catalog([])

        server.search_catalog(time_start="2023-01-01")
        assert mock_conn.get_catalog.call_args.kwargs["timefilter"] == [
            "2023-01-01",
            None,
        ]

    def test_empty_catalog(self, mock_conn):
        mock_conn.get_catalog.return_value = _mock_catalog([])

        parsed = json.loads(server.search_catalog(search="nonexistent"))
        assert parsed["count"] == 0
        assert "No datasources found" in parsed["message"]

    def test_note_when_limit_reached(self, mock_conn):
        ds = _mock_datasource()
        mock_conn.get_catalog.return_value = _mock_catalog([ds])

        parsed = json.loads(server.search_catalog(search="wave", limit=1))
        assert "more matches may exist" in parsed["note"]

    def test_with_bbox(self, mock_conn):
        mock_conn.get_catalog.return_value = _mock_catalog([])

        server.search_catalog(bbox=[120, -50, 180, 10])
        geofilter = mock_conn.get_catalog.call_args.kwargs["geofilter"]
        assert isinstance(geofilter, GeoFilter)

    def test_rejects_nonpositive_limit(self, mock_conn):
        with pytest.raises(ToolError, match="at least 1"):
            server.search_catalog(search="wave", limit=0)
        mock_conn.get_catalog.assert_not_called()


class TestGetDatasourceInfo:
    def test_returns_metadata(self, mock_conn):
        ds = _mock_datasource(id="my-ds", name="My Dataset", driver="onzarr")
        mock_conn.get_datasource.return_value = ds

        parsed = json.loads(server.get_datasource_info("my-ds"))
        assert parsed["id"] == "my-ds"
        assert parsed["name"] == "My Dataset"
        assert parsed["driver"] == "onzarr"


class TestBuildQuery:
    def test_range_times_without_sentinels(self):
        q = server._build_query("test-ds", time_start="2024-01-01")
        assert q.timefilter.times[1] is None

    def test_series_times(self):
        q = server._build_query("test-ds", times=["2024-01-01", "2024-02-01"])
        assert q.timefilter.type.value == "series"

    def test_times_and_range_conflict(self):
        with pytest.raises(ToolError, match="not both"):
            server._build_query(
                "test-ds", time_start="2024-01-01", times=["2024-01-02"]
            )

    def test_time_resolution_and_resample(self):
        q = server._build_query(
            "test-ds",
            time_start="2024-01-01",
            time_end="2024-12-31",
            time_resolution="1D",
            time_resample="mean",
        )
        assert q.timefilter.resolution == "1D"
        assert q.timefilter.resample.value == "mean"

    def test_time_resolution_requires_range(self):
        with pytest.raises(ToolError, match="range"):
            server._build_query("test-ds", time_resolution="1D")

    def test_time_resolution_rejected_with_series_times(self):
        with pytest.raises(ToolError, match="range"):
            server._build_query("test-ds", times=["2024-01-01"], time_resolution="1D")

    def test_bbox_and_feature_conflict(self):
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [174.0, -41.0]},
            "properties": {},
        }
        with pytest.raises(ToolError, match="not both"):
            server._build_query("test-ds", bbox=[0, 0, 1, 1], geofilter_feature=feature)

    def test_feature_geofilter_with_interp(self):
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [174.0, -41.0]},
            "properties": {},
        }
        q = server._build_query(
            "test-ds", geofilter_feature=feature, geofilter_interp="nearest"
        )
        assert q.geofilter.type.value == "feature"
        assert q.geofilter.interp.value == "nearest"

    def test_level_series(self):
        q = server._build_query("test-ds", levels=[0.0, 10.0], level_interp="nearest")
        assert q.levelfilter.type.value == "series"
        assert q.levelfilter.interp.value == "nearest"

    def test_levels_and_range_conflict(self):
        with pytest.raises(ToolError, match="not both"):
            server._build_query("test-ds", levels=[0.0], level_min=0.0)

    def test_coord_filters_typed(self):
        q = server._build_query(
            "test-ds",
            coord_filters=[CoordSelector(coord="station", values=["A1", "B2"])],
        )
        assert q.coordfilter[0].coord == "station"

    def test_crs_and_aggregate(self):
        q = server._build_query(
            "test-ds",
            crs=4326,
            aggregate_operations=["mean", "max"],
            aggregate_temporal=False,
        )
        assert q.crs == 4326
        assert q.aggregate.temporal is False

    def test_invalid_bbox_raises_tool_error(self):
        with pytest.raises(ToolError, match="Invalid query parameters"):
            server._build_query("test-ds", bbox=[0, 0, 1])

    def test_nonpositive_limit_raises_tool_error(self):
        with pytest.raises(ToolError, match="at least 1"):
            server._build_query("test-ds", limit=0)


class TestStageQuery:
    def test_small_dataset_recommends_inline(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(Container.Dataset, size=1000, dlen=10)

        parsed = json.loads(server.stage_query(datasource_id="test-ds"))
        assert parsed["staged"] is True
        assert parsed["container"] == "dataset"
        assert parsed["size_bytes"] == 1000
        assert "query_data" in parsed["recommendation"]
        assert parsed["query"]["datasource"] == "test-ds"

    def test_large_dataset_recommends_export(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(Container.Dataset, size=10**9)

        parsed = json.loads(server.stage_query(datasource_id="test-ds"))
        assert "export_query" in parsed["recommendation"]

    def test_row_cap_warning(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(
            Container.DataFrame, size=10**9, dlen=3_000_000
        )

        parsed = json.loads(server.stage_query(datasource_id="test-ds"))
        assert any("caps tabular" in w for w in parsed["warnings"])

    def test_no_data(self, mock_conn, mock_stage):
        mock_stage.return_value = None

        parsed = json.loads(server.stage_query(datasource_id="test-ds"))
        assert parsed["staged"] is False

    def test_stage_error_echoes_query(self, mock_conn, mock_stage):
        mock_stage.side_effect = DatameshQueryError("bad datasource")

        parsed = json.loads(server.stage_query(datasource_id="test-ds"))
        assert "bad datasource" in parsed["error"]
        assert parsed["query"]["datasource"] == "test-ds"


class TestQueryData:
    def test_small_dataframe_inline(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(Container.DataFrame, size=100)
        mock_conn.query.return_value = pd.DataFrame({"temp": [15.0, 16.0]})

        parsed = json.loads(server.query_data(datasource_id="test-ds"))
        assert parsed["container"] == "dataframe"
        assert parsed["rows"] == 2
        assert parsed["truncated"] is False
        assert parsed["data"][0]["temp"] == 15.0
        assert parsed["staged_size_bytes"] == 100
        assert mock_conn.query.call_args.kwargs["use_dask"] is False

    def test_truncation_flagged(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(Container.DataFrame, size=100)
        mock_conn.query.return_value = pd.DataFrame({"x": range(150)})

        parsed = json.loads(server.query_data(datasource_id="test-ds"))
        assert parsed["truncated"] is True
        assert len(parsed["data"]) == 100  # DEFAULT_MAX_INLINE_ROWS
        assert "note" in parsed

    def test_large_frame_refused_without_download(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(Container.DataFrame, size=10**9)

        parsed = json.loads(server.query_data(datasource_id="test-ds"))
        assert parsed["refused"] is True
        assert "export_query" in parsed["message"]
        mock_conn.query.assert_not_called()

    def test_large_dataset_goes_lazy(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(Container.Dataset, size=10**9)
        mock_conn.query.return_value = _small_dataset().chunk({"time": 1})

        parsed = json.loads(server.query_data(datasource_id="test-ds"))
        assert mock_conn.query.call_args.kwargs["use_dask"] is True
        assert parsed["lazy"] is True
        assert "data" not in parsed

    def test_small_dataset_values_have_coordinates(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(Container.Dataset, size=100)
        mock_conn.query.return_value = _small_dataset()

        parsed = json.loads(server.query_data(datasource_id="test-ds"))
        assert parsed["container"] == "dataset"
        assert parsed["data"][0]["hs"] == 1.0
        assert "time" in parsed["data"][0]

    def test_library_warnings_surfaced(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(Container.DataFrame, size=100)

        def _query_with_warning(*args, **kwargs):
            warnings.warn("Query limited to 2000000 rows")
            return pd.DataFrame({"x": [1]})

        mock_conn.query.side_effect = _query_with_warning

        parsed = json.loads(server.query_data(datasource_id="test-ds"))
        assert any("2000000 rows" in w for w in parsed["warnings"])

    def test_query_error_echoes_query(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(Container.DataFrame, size=100)
        mock_conn.query.side_effect = DatameshConnectError("server error: 500")

        parsed = json.loads(
            server.query_data(datasource_id="test-ds", variables=["Hs"])
        )
        assert "error" in parsed
        assert parsed["query"]["datasource"] == "test-ds"
        assert parsed["query"]["variables"] == ["Hs"]

    def test_session_error_returns_structured_error(self, mock_conn, mock_stage):
        mock_stage.side_effect = DatameshSessionError("bad token")

        parsed = json.loads(server.query_data(datasource_id="test-ds"))
        assert "bad token" in parsed["error"]
        assert parsed["query"]["datasource"] == "test-ds"

    def test_midsize_eager_dataset_shows_values(self, mock_conn, mock_stage):
        # Eager datasets between 1 MB and the inline limit must include
        # values, not a false "larger than inline limit" note.
        mock_stage.return_value = make_stage(Container.Dataset, size=2_000_000)
        big = xr.Dataset({"v": (("x",), np.zeros(300_000))})
        mock_conn.query.return_value = big

        parsed = json.loads(server.query_data(datasource_id="test-ds"))
        assert parsed["lazy"] is False
        assert len(parsed["data"]) == 100  # DEFAULT_MAX_INLINE_ROWS
        assert parsed["truncated"] is True


class TestExportQuery:
    def test_frame_to_parquet(self, mock_conn, mock_stage, tmp_path):
        mock_stage.return_value = make_stage(Container.DataFrame, size=100)
        mock_conn.query.return_value = pd.DataFrame({"temp": [15.0, 16.0]})
        dest = tmp_path / "out.parquet"

        parsed = json.loads(
            server.export_query(datasource_id="test-ds", path=str(dest))
        )
        assert dest.exists()
        assert parsed["format"] == "parquet"
        assert parsed["bytes_written"] == dest.stat().st_size
        assert "data" not in parsed["summary"]
        # A successful full export must not claim truncation.
        assert "truncated" not in parsed["summary"]
        assert "note" not in parsed["summary"]

    def test_frame_to_csv(self, mock_conn, mock_stage, tmp_path):
        mock_stage.return_value = make_stage(Container.DataFrame, size=100)
        mock_conn.query.return_value = pd.DataFrame({"temp": [15.0]})
        dest = tmp_path / "out.csv"

        parsed = json.loads(
            server.export_query(datasource_id="test-ds", path=str(dest), format="csv")
        )
        assert parsed["format"] == "csv"
        assert "temp" in dest.read_text()

    def test_dataset_to_netcdf_streams_lazily(self, mock_conn, mock_stage, tmp_path):
        mock_stage.return_value = make_stage(Container.Dataset, size=100)
        mock_conn.query.return_value = _small_dataset()
        dest = tmp_path / "out.nc"

        parsed = json.loads(
            server.export_query(datasource_id="test-ds", path=str(dest))
        )
        assert dest.exists()
        assert parsed["format"] == "netcdf"
        assert mock_conn.query.call_args.kwargs["use_dask"] is True

    def test_dataset_rejects_csv(self, mock_conn, mock_stage, tmp_path):
        mock_stage.return_value = make_stage(Container.Dataset, size=100)

        with pytest.raises(ToolError, match="netcdf"):
            server.export_query(
                datasource_id="test-ds",
                path=str(tmp_path / "out.csv"),
                format="csv",
            )

    def test_refuses_overwrite(self, mock_conn, mock_stage, tmp_path):
        dest = tmp_path / "exists.nc"
        dest.write_text("data")

        with pytest.raises(ToolError, match="overwrite"):
            server.export_query(datasource_id="test-ds", path=str(dest))

    def test_oversized_frame_refused(self, mock_conn, mock_stage, tmp_path):
        mock_stage.return_value = make_stage(Container.DataFrame, size=3 * 10**9)

        parsed = json.loads(
            server.export_query(
                datasource_id="test-ds", path=str(tmp_path / "out.parquet")
            )
        )
        assert parsed["refused"] is True
        mock_conn.query.assert_not_called()

    def test_none_result_writes_nothing(self, mock_conn, mock_stage, tmp_path):
        mock_stage.return_value = make_stage(Container.Dataset, size=100)
        mock_conn.query.return_value = None
        dest = tmp_path / "out.nc"

        parsed = json.loads(
            server.export_query(datasource_id="test-ds", path=str(dest))
        )
        assert parsed["status"] == "no_data"
        assert not dest.exists()

    def test_write_failure_cleans_partial_file(self, mock_conn, mock_stage, tmp_path):
        mock_stage.return_value = make_stage(Container.Dataset, size=100)
        broken = MagicMock()
        broken.to_netcdf.side_effect = DatameshConnectError("chunk fetch failed")
        mock_conn.query.return_value = broken
        dest = tmp_path / "out.nc"

        parsed = json.loads(
            server.export_query(datasource_id="test-ds", path=str(dest))
        )
        assert "chunk fetch failed" in parsed["error"]
        assert not dest.exists()

    def test_export_dir_confinement(self, mock_conn, mock_stage, tmp_path, monkeypatch):
        monkeypatch.setenv("OCEANUM_MCP_EXPORT_DIR", str(tmp_path / "allowed"))
        with pytest.raises(ToolError, match="OCEANUM_MCP_EXPORT_DIR"):
            server.export_query(
                datasource_id="test-ds", path=str(tmp_path / "escape.nc")
            )


class TestLoadDatasource:
    def test_dataset_loaded(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(Container.Dataset, size=10**12)
        mock_conn.load_datasource.return_value = _small_dataset().chunk({"time": 1})

        parsed = json.loads(server.load_datasource(datasource_id="test-ds"))
        assert parsed["container"] == "dataset"
        assert parsed["lazy"] is True

    def test_large_frame_refused(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(Container.DataFrame, size=10**9)

        parsed = json.loads(server.load_datasource(datasource_id="test-ds"))
        assert parsed["refused"] is True
        mock_conn.load_datasource.assert_not_called()

    def test_load_error_returns_datasource_id(self, mock_conn, mock_stage):
        mock_stage.side_effect = DatameshConnectError("server error: 500")

        parsed = json.loads(server.load_datasource(datasource_id="bad-ds"))
        assert "error" in parsed
        assert parsed["datasource_id"] == "bad-ds"

    def test_invalid_id_raises_tool_error(self, mock_conn, mock_stage):
        # Query validates datasource ids (min_length=3); the failure must be
        # a ToolError, not a raw pydantic ValidationError.
        with pytest.raises(ToolError, match="Invalid datasource_id"):
            server.load_datasource(datasource_id="ds")


class TestUpdateMetadata:
    def test_updates_fields_with_typed_info(self, mock_conn):
        ds = _mock_datasource(id="my-ds", name="Updated Name", tags=["new-tag"])
        mock_conn.update_metadata.return_value = ds

        result = server.update_metadata(
            datasource_id="my-ds",
            name="Updated Name",
            tags=["new-tag"],
            info={"source": "buoy"},
        )
        mock_conn.update_metadata.assert_called_once_with(
            "my-ds", name="Updated Name", tags=["new-tag"], info={"source": "buoy"}
        )
        parsed = json.loads(result)
        assert parsed["id"] == "my-ds"


class TestRegistration:
    async def test_tools_registered_with_annotations(self):
        tools = {t.name: t for t in await server.mcp.list_tools()}
        assert set(tools) >= {
            "search_catalog",
            "get_datasource_info",
            "stage_query",
            "query_data",
            "export_query",
            "load_datasource",
            "update_metadata",
        }
        assert tools["search_catalog"].annotations.readOnlyHint is True
        assert tools["query_data"].annotations.readOnlyHint is True
        assert tools["update_metadata"].annotations.destructiveHint is True
        assert tools["update_metadata"].annotations.readOnlyHint is False

    async def test_docstrings_present(self):
        for tool in await server.mcp.list_tools():
            assert tool.description, f"tool {tool.name} has no description"

    async def test_read_only_mode_hides_update_metadata(self, monkeypatch):
        monkeypatch.setenv("OCEANUM_MCP_READ_ONLY", "1")
        try:
            reloaded = importlib.reload(server)
            tools = {t.name for t in await reloaded.mcp.list_tools()}
            assert "update_metadata" not in tools
            assert "query_data" in tools
        finally:
            monkeypatch.delenv("OCEANUM_MCP_READ_ONLY")
            importlib.reload(server)
