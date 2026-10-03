"""OCE-303: "did you mean" suggestions for unknown datasources and variables.

The SDK/gateway error strings mocked here are the real ones:
- metadata server (get_datasource): 404 -> the SDK's "Datasource <id> not
  found"; 403 (exists, not shared) -> the server's "You do not have
  permission to access this datasource".
- query engine (stage): 404 "Datasource <id> not found or not authorized"
  (missing and unentitled look the same) and 400 "Invalid variable selection
  - variable not found: '<name>'", which the SDK's _stage_request wraps as
  'Datamesh server error: {"detail": ...}'.
"""

import json
import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from oceanum.datamesh.exceptions import DatameshConnectError, DatameshQueryError

import oceanum_mcp.servers.datamesh.server as server

FORBIDDEN = "You do not have permission to access this datasource"


def _not_found(ds_id: str) -> DatameshConnectError:
    return DatameshConnectError(f"Datasource {ds_id} not found")


def _stage_not_found(ds_id: str) -> DatameshConnectError:
    return DatameshConnectError(
        'Datamesh server error: {"detail":"Datasource '
        f'{ds_id} not found or not authorized"}}'
    )


def _stage_bad_variable(name: str) -> DatameshConnectError:
    return DatameshConnectError(
        'Datamesh server error: {"detail":"Invalid variable selection - '
        f"variable not found: '{name}'\"}}"
    )


def _summary(ds_id: str, name: str) -> MagicMock:
    ds = MagicMock()
    ds.id = ds_id
    ds.name = name
    return ds


def _catalog(*datasources: MagicMock) -> MagicMock:
    catalog = MagicMock()
    catalog.__iter__ = MagicMock(return_value=iter(datasources))
    return catalog


def _wave_datasource() -> MagicMock:
    ds = MagicMock()
    ds.id = "era5_wave_global"
    ds.variables = {
        "hs": {
            "attrs": {
                "standard_name": "sea_surface_wave_significant_height",
                "long_name": "Significant wave height",
                "units": "m",
            }
        },
        "tp": {
            "attrs": {
                "standard_name": "sea_surface_wave_period_at_variance_spectral_density_maximum",
                "long_name": "Peak wave period",
            }
        },
        "dpm": {"attrs": {"long_name": "Mean direction at peak"}},
    }
    ds.dataschema.coords = {"time": {}, "latitude": {}, "longitude": {}}
    return ds


NEAR_CATALOG = (
    _summary("era5_wind_global", "ERA5 global winds"),
    _summary("era5_wave_global", "ERA5 global waves"),
    _summary("gebco_bathymetry", "GEBCO bathymetry"),
)


class TestGetDatasourceInfoNotFound:
    def test_unknown_id_suggests_near_matches(self, mock_conn):
        mock_conn.get_datasource.side_effect = _not_found("era5_wave_glob")
        mock_conn.get_catalog.return_value = _catalog(*NEAR_CATALOG)

        parsed = json.loads(server.get_datasource_info("era5_wave_glob"))

        assert "not found" in parsed["error"]
        ids = [s["id"] for s in parsed["suggestions"]]
        assert ids[0] == "era5_wave_global"
        assert "gebco_bathymetry" not in ids
        assert parsed["suggestions"][0]["name"] == "ERA5 global waves"
        assert parsed["datasource_id"] == "era5_wave_glob"

    def test_catalog_lookup_is_bounded_and_uses_id_tokens(self, mock_conn):
        mock_conn.get_datasource.side_effect = _not_found("era5_wave_glob")
        mock_conn.get_catalog.return_value = _catalog(*NEAR_CATALOG)

        server.get_datasource_info("era5_wave_glob")

        mock_conn.get_catalog.assert_called_once()
        kwargs = mock_conn.get_catalog.call_args.kwargs
        assert kwargs["limit"] == server.SUGGEST_CATALOG_LIMIT
        for token in ("era5", "wave", "glob"):
            assert token in kwargs["search"]

    def test_suggestions_capped(self, mock_conn):
        mock_conn.get_datasource.side_effect = _not_found("wave")
        mock_conn.get_catalog.return_value = _catalog(
            *(_summary(f"wave_{i}", f"Wave {i}") for i in range(12))
        )

        parsed = json.loads(server.get_datasource_info("wave"))

        assert len(parsed["suggestions"]) == server.MAX_SUGGESTIONS

    def test_unknown_id_without_near_matches(self, mock_conn):
        mock_conn.get_datasource.side_effect = _not_found("zzqx")
        mock_conn.get_catalog.return_value = _catalog(
            _summary("gebco_bathymetry", "GEBCO bathymetry")
        )

        parsed = json.loads(server.get_datasource_info("zzqx"))

        assert parsed["suggestions"] == []
        assert "search_catalog" in parsed["error"]

    def test_forbidden_is_not_a_did_you_mean(self, mock_conn):
        mock_conn.get_datasource.side_effect = DatameshConnectError(FORBIDDEN)

        parsed = json.loads(server.get_datasource_info("private_ds"))

        assert "suggestions" not in parsed
        assert "access" in parsed["error"]
        assert "private_ds" in parsed["error"]
        mock_conn.get_catalog.assert_not_called()

    def test_suggestion_lookup_failure_returns_original_error(self, mock_conn):
        mock_conn.get_datasource.side_effect = _not_found("era5_wave_glob")
        mock_conn.get_catalog.side_effect = DatameshConnectError("catalog down")

        parsed = json.loads(server.get_datasource_info("era5_wave_glob"))

        assert parsed["error"] == "Datasource era5_wave_glob not found"
        assert "suggestions" not in parsed

    def test_other_errors_are_reported_unchanged(self, mock_conn):
        mock_conn.get_datasource.side_effect = DatameshConnectError(
            "Datamesh server error: 502 Bad Gateway"
        )

        parsed = json.loads(server.get_datasource_info("era5_wave_global"))

        assert parsed["error"] == "Datamesh server error: 502 Bad Gateway"
        assert "suggestions" not in parsed
        mock_conn.get_catalog.assert_not_called()


class TestQueryToolsUnknownDatasource:
    def test_404_suggests_near_matches(self, mock_conn, mock_stage):
        mock_stage.side_effect = _stage_not_found("era5_wave_glob")
        mock_conn.get_datasource.side_effect = _not_found("era5_wave_glob")
        mock_conn.get_catalog.return_value = _catalog(*NEAR_CATALOG)

        parsed = json.loads(server.stage_query(datasource_id="era5_wave_glob"))

        assert parsed["suggestions"][0]["id"] == "era5_wave_global"
        assert parsed["query"]["datasource"] == "era5_wave_glob"
        mock_conn.get_datasource.assert_called_once_with("era5_wave_glob")
        mock_conn.get_catalog.assert_called_once()

    def test_403_reports_no_access(self, mock_conn, mock_stage):
        # The query engine says "not found or not authorized" either way; the
        # metadata server's 403 tells the two apart.
        mock_stage.side_effect = _stage_not_found("private_ds")
        mock_conn.get_datasource.side_effect = DatameshConnectError(FORBIDDEN)

        parsed = json.loads(server.query_data(datasource_id="private_ds"))

        assert "access" in parsed["error"]
        assert "suggestions" not in parsed
        assert parsed["query"]["datasource"] == "private_ds"
        mock_conn.get_catalog.assert_not_called()

    def test_visible_datasource_keeps_original_error(self, mock_conn, mock_stage):
        exc = _stage_not_found("era5_wave_global")
        mock_stage.side_effect = exc
        mock_conn.get_datasource.return_value = _wave_datasource()

        parsed = json.loads(server.stage_query(datasource_id="era5_wave_global"))

        assert parsed["error"] == str(exc)
        assert "suggestions" not in parsed
        mock_conn.get_catalog.assert_not_called()

    def test_lookup_failure_returns_original_error(self, mock_conn, mock_stage):
        exc = _stage_not_found("era5_wave_glob")
        mock_stage.side_effect = exc
        mock_conn.get_datasource.side_effect = _not_found("era5_wave_glob")
        mock_conn.get_catalog.side_effect = RuntimeError("boom")

        parsed = json.loads(server.query_data(datasource_id="era5_wave_glob"))

        assert parsed["error"] == str(exc)
        assert "suggestions" not in parsed

    def test_local_export(self, mock_conn, mock_stage, tmp_path):
        mock_stage.side_effect = _stage_not_found("era5_wave_glob")
        mock_conn.get_datasource.side_effect = _not_found("era5_wave_glob")
        mock_conn.get_catalog.return_value = _catalog(*NEAR_CATALOG)

        parsed = json.loads(
            server.export_query(
                datasource_id="era5_wave_glob", path=str(tmp_path / "out.nc")
            )
        )

        assert parsed["suggestions"][0]["id"] == "era5_wave_global"

    def test_hosted_export(self, mock_conn):
        from oceanum_mcp.common.config import set_transport

        mock_conn.get_datasource.side_effect = _not_found("era5_wave_glob")
        mock_conn.get_catalog.return_value = _catalog(*NEAR_CATALOG)
        # _download_stage raises the gateway's detail as a DatameshQueryError.
        exc = DatameshQueryError(
            "Datasource era5_wave_glob not found or not authorized"
        )
        set_transport("http")
        try:
            with patch.object(server, "_download_stage", side_effect=exc):
                parsed = json.loads(server.export_query(datasource_id="era5_wave_glob"))
        finally:
            set_transport("stdio")

        assert parsed["suggestions"][0]["id"] == "era5_wave_global"
        assert parsed["query"]["datasource"] == "era5_wave_glob"

    def test_load_datasource(self, mock_conn, mock_stage):
        mock_stage.side_effect = _stage_not_found("era5_wave_glob")
        mock_conn.get_datasource.side_effect = _not_found("era5_wave_glob")
        mock_conn.get_catalog.return_value = _catalog(*NEAR_CATALOG)

        parsed = json.loads(server.load_datasource("era5_wave_glob"))

        assert parsed["suggestions"][0]["id"] == "era5_wave_global"
        assert parsed["datasource_id"] == "era5_wave_glob"


class TestQueryToolsUnknownVariable:
    def test_near_name(self, mock_conn, mock_stage):
        mock_stage.side_effect = _stage_bad_variable("hss")
        mock_conn.get_datasource.return_value = _wave_datasource()

        parsed = json.loads(
            server.query_data(datasource_id="era5_wave_global", variables=["hss", "tp"])
        )

        assert parsed["suggestions"] == [{"variable": "hss", "did_you_mean": ["hs"]}]
        assert "hss" in parsed["error"]
        assert parsed["query"]["variables"] == ["hss", "tp"]
        mock_conn.get_datasource.assert_called_once_with("era5_wave_global")
        mock_conn.get_catalog.assert_not_called()

    def test_matches_standard_name(self, mock_conn, mock_stage):
        mock_stage.side_effect = _stage_bad_variable("significant_wave_height")
        ds = _wave_datasource()
        del ds.variables["hs"]["attrs"]["long_name"]  # standard_name only
        mock_conn.get_datasource.return_value = ds

        parsed = json.loads(
            server.stage_query(
                datasource_id="era5_wave_global",
                variables=["significant_wave_height"],
            )
        )

        assert parsed["suggestions"][0]["variable"] == "significant_wave_height"
        assert parsed["suggestions"][0]["did_you_mean"][0] == "hs"
        assert "tp" not in parsed["suggestions"][0]["did_you_mean"]

    def test_matches_long_name(self, mock_conn, mock_stage):
        mock_stage.side_effect = _stage_bad_variable("peak_period")
        mock_conn.get_datasource.return_value = _wave_datasource()

        parsed = json.loads(
            server.stage_query(
                datasource_id="era5_wave_global", variables=["peak_period"]
            )
        )

        assert parsed["suggestions"][0]["did_you_mean"][0] == "tp"

    def test_coordinates_are_not_unknown(self, mock_conn, mock_stage):
        mock_stage.side_effect = _stage_bad_variable("hss")
        mock_conn.get_datasource.return_value = _wave_datasource()

        parsed = json.loads(
            server.stage_query(
                datasource_id="era5_wave_global", variables=["time", "hss"]
            )
        )

        assert [s["variable"] for s in parsed["suggestions"]] == ["hss"]

    def test_no_close_variable(self, mock_conn, mock_stage):
        mock_stage.side_effect = _stage_bad_variable("chlorophyll")
        mock_conn.get_datasource.return_value = _wave_datasource()

        parsed = json.loads(
            server.stage_query(
                datasource_id="era5_wave_global", variables=["chlorophyll"]
            )
        )

        assert parsed["suggestions"] == [
            {"variable": "chlorophyll", "did_you_mean": []}
        ]
        assert "get_datasource_info" in parsed["error"]

    def test_lookup_failure_returns_original_error(self, mock_conn, mock_stage):
        exc = _stage_bad_variable("hss")
        mock_stage.side_effect = exc
        mock_conn.get_datasource.side_effect = DatameshConnectError("metadata down")

        parsed = json.loads(
            server.query_data(datasource_id="era5_wave_global", variables=["hss"])
        )

        assert parsed["error"] == str(exc)
        assert "suggestions" not in parsed

    def test_all_requested_variables_exist_keeps_original_error(
        self, mock_conn, mock_stage
    ):
        exc = _stage_bad_variable("hs")
        mock_stage.side_effect = exc
        mock_conn.get_datasource.return_value = _wave_datasource()

        parsed = json.loads(
            server.query_data(datasource_id="era5_wave_global", variables=["hs"])
        )

        assert parsed["error"] == str(exc)
        assert "suggestions" not in parsed

    def test_different_quantity_not_suggested(self, mock_conn, mock_stage):
        # Sharing some words is not enough: temperature is not salinity.
        mock_stage.side_effect = _stage_bad_variable("sea_surface_temperature")
        ds = _wave_datasource()
        ds.variables = {"sss": {"attrs": {"standard_name": "sea_surface_salinity"}}}
        mock_conn.get_datasource.return_value = ds

        parsed = json.loads(
            server.stage_query(
                datasource_id="era5_wave_global", variables=["sea_surface_temperature"]
            )
        )

        assert parsed["suggestions"][0]["did_you_mean"] == []


class TestNoExtraCallsForOtherErrors:
    @pytest.mark.parametrize(
        "exc",
        [
            DatameshConnectError("Datamesh server error: 500"),
            server.GatewayTimeout("Datamesh staging timed out after 60s."),
            DatameshQueryError("No valid cycle found for datasource era5_wave_global"),
        ],
    )
    def test_unrelated_error_unchanged(self, mock_conn, mock_stage, exc):
        mock_stage.side_effect = exc

        parsed = json.loads(
            server.query_data(datasource_id="era5_wave_global", variables=["hss"])
        )

        assert parsed["error"] == str(exc)
        assert "suggestions" not in parsed
        mock_conn.get_datasource.assert_not_called()
        mock_conn.get_catalog.assert_not_called()


class TestSuggestionRobustness:
    @pytest.mark.parametrize(
        "message",
        [
            "Datamesh server error: 500 permission check failed upstream",
            "Datamesh server error: index not found",
            "Datasource era5_wave_glob not Authorized",
        ],
    )
    def test_metadata_errors_matched_exactly(self, mock_conn, message):
        mock_conn.get_datasource.side_effect = DatameshConnectError(message)

        parsed = json.loads(server.get_datasource_info("era5_wave_glob"))

        assert parsed == {"error": message, "datasource_id": "era5_wave_glob"}
        mock_conn.get_catalog.assert_not_called()

    def test_search_operator_words_dropped(self, mock_conn):
        mock_conn.get_datasource.side_effect = _not_found("wave_or_wind_and_not")
        mock_conn.get_catalog.return_value = _catalog()

        server.get_datasource_info("wave_or_wind_and_not")

        assert mock_conn.get_catalog.call_args.kwargs["search"] == "wave or wind"

    def test_slow_lookup_returns_original_error(self, mock_conn, mock_stage):
        exc = _stage_not_found("era5_wave_glob")
        mock_stage.side_effect = exc
        mock_conn.get_datasource.side_effect = _not_found("era5_wave_glob")
        mock_conn.get_catalog.side_effect = lambda **_: time.sleep(1)

        start = time.monotonic()
        with patch.object(server, "SUGGEST_TIMEOUT", 0.05):
            parsed = json.loads(server.query_data(datasource_id="era5_wave_glob"))

        assert time.monotonic() - start < 0.5
        assert parsed["error"] == str(exc)
        assert "suggestions" not in parsed

    def test_lookup_failure_is_logged(self, mock_conn, caplog):
        mock_conn.get_datasource.side_effect = _not_found("era5_wave_glob")
        mock_conn.get_catalog.side_effect = RuntimeError("boom")

        with caplog.at_level(logging.WARNING, logger=server.__name__):
            server.get_datasource_info("era5_wave_glob")

        assert any("Suggestions" in r.getMessage() for r in caplog.records)
