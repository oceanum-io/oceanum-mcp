"""Tests for the Datamesh MCP server."""

import json
from unittest.mock import patch, MagicMock

import pytest


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


class TestSearchCatalog:
    def test_returns_results(self):
        mock_conn = MagicMock()
        ds = _mock_datasource(id="era5-waves", name="ERA5 Waves", tags=["wave"])
        mock_conn.get_catalog.return_value = _mock_catalog([ds])

        with patch(
            "oceanum_mcp.servers.datamesh.server.get_datamesh_connector",
            return_value=mock_conn,
        ):
            from oceanum_mcp.servers.datamesh.server import search_catalog

            result = search_catalog(search="wave")
            assert "era5-waves" in result
            assert "ERA5 Waves" in result
            parsed = json.loads(result)
            assert len(parsed) == 1
            assert parsed[0]["id"] == "era5-waves"

    def test_empty_catalog(self):
        mock_conn = MagicMock()
        mock_conn.get_catalog.return_value = _mock_catalog([])

        with patch(
            "oceanum_mcp.servers.datamesh.server.get_datamesh_connector",
            return_value=mock_conn,
        ):
            from oceanum_mcp.servers.datamesh.server import search_catalog

            result = search_catalog(search="nonexistent")
            assert "No datasources found" in result

    def test_with_bbox(self):
        mock_conn = MagicMock()
        mock_conn.get_catalog.return_value = _mock_catalog([])

        with patch(
            "oceanum_mcp.servers.datamesh.server.get_datamesh_connector",
            return_value=mock_conn,
        ):
            from oceanum_mcp.servers.datamesh.server import search_catalog

            search_catalog(bbox=[120, -50, 180, 10])
            call_kwargs = mock_conn.get_catalog.call_args
            assert call_kwargs.kwargs.get("geofilter") is not None


class TestGetDatasourceInfo:
    def test_returns_metadata(self):
        mock_conn = MagicMock()
        ds = _mock_datasource(id="my-ds", name="My Dataset", driver="onzarr")
        mock_conn.get_datasource.return_value = ds

        with patch(
            "oceanum_mcp.servers.datamesh.server.get_datamesh_connector",
            return_value=mock_conn,
        ):
            from oceanum_mcp.servers.datamesh.server import get_datasource_info

            result = get_datasource_info("my-ds")
            parsed = json.loads(result)
            assert parsed["id"] == "my-ds"
            assert parsed["name"] == "My Dataset"
            assert parsed["driver"] == "onzarr"


class TestQueryData:
    def test_basic_query(self):
        import pandas as pd

        mock_conn = MagicMock()
        mock_conn.query.return_value = pd.DataFrame({"temp": [15.0, 16.0]})

        with patch(
            "oceanum_mcp.servers.datamesh.server.get_datamesh_connector",
            return_value=mock_conn,
        ):
            from oceanum_mcp.servers.datamesh.server import query_data

            result = query_data(datasource_id="test-ds")
            assert "DataFrame" in result
            assert "2 rows" in result

    def test_query_builds_filters(self):
        mock_conn = MagicMock()
        mock_conn.query.return_value = None

        with patch(
            "oceanum_mcp.servers.datamesh.server.get_datamesh_connector",
            return_value=mock_conn,
        ):
            from oceanum_mcp.servers.datamesh.server import query_data

            query_data(
                datasource_id="test-ds",
                variables=["Hs", "Tp"],
                time_start="2024-01-01",
                time_end="2024-01-31",
                bbox=[120, -50, 180, 10],
                limit=100,
            )
            query_dict = mock_conn.query.call_args[0][0]
            assert query_dict["datasource"] == "test-ds"
            assert query_dict["variables"] == ["Hs", "Tp"]
            assert "timefilter" in query_dict
            assert "geofilter" in query_dict
            assert query_dict["limit"] == 100


class TestUpdateMetadata:
    def test_updates_fields(self):
        mock_conn = MagicMock()
        ds = _mock_datasource(id="my-ds", name="Updated Name", tags=["new-tag"])
        mock_conn.update_metadata.return_value = ds

        with patch(
            "oceanum_mcp.servers.datamesh.server.get_datamesh_connector",
            return_value=mock_conn,
        ):
            from oceanum_mcp.servers.datamesh.server import update_metadata

            result = update_metadata(
                datasource_id="my-ds",
                name="Updated Name",
                tags=["new-tag"],
            )
            mock_conn.update_metadata.assert_called_once_with(
                "my-ds", name="Updated Name", tags=["new-tag"]
            )
            parsed = json.loads(result)
            assert parsed["id"] == "my-ds"
