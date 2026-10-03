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
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from oceanum.datamesh.exceptions import DatameshConnectError, DatameshQueryError

import oceanum_mcp.servers.datamesh.server as server
from tests.conftest import make_stage

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


ALIAS = "sea_surface_wave_significant_height"


def _hs_dataset() -> xr.Dataset:
    return xr.Dataset(
        {"hs": (("time",), np.array([1.0, 2.0, 3.0]))},
        coords={"time": pd.date_range("2024-01-01", periods=3, freq="h")},
    )


class TestStandardNameResolution:
    """OCE-303 criterion 2: a variable named by the exact CF standard_name of
    exactly one variable is resolved to it, and the response says so."""

    def test_stage_query_resolves(self, mock_conn, mock_stage):
        mock_stage.side_effect = [_stage_bad_variable(ALIAS), make_stage()]
        mock_conn.get_datasource.return_value = _wave_datasource()

        parsed = json.loads(
            server.stage_query(datasource_id="era5_wave_global", variables=[ALIAS])
        )

        assert parsed["staged"] is True
        assert parsed["resolved_variables"] == {ALIAS: "hs"}
        assert parsed["query"]["variables"] == ["hs"]
        assert mock_stage.call_count == 2
        assert mock_stage.call_args_list[1].args[1].variables == ["hs"]
        mock_conn.get_datasource.assert_called_once_with("era5_wave_global")

    def test_query_data_resolves_and_queries_resolved_name(self, mock_conn, mock_stage):
        mock_stage.side_effect = [_stage_bad_variable(ALIAS), make_stage()]
        mock_conn.get_datasource.return_value = _wave_datasource()
        mock_conn.query.return_value = _hs_dataset()

        parsed = json.loads(
            server.query_data(datasource_id="era5_wave_global", variables=[ALIAS])
        )

        assert "error" not in parsed
        assert parsed["resolved_variables"] == {ALIAS: "hs"}
        assert mock_conn.query.call_args.args[0].variables == ["hs"]

    def test_mixed_real_and_alias(self, mock_conn, mock_stage):
        mock_stage.side_effect = [_stage_bad_variable(ALIAS), make_stage()]
        mock_conn.get_datasource.return_value = _wave_datasource()

        parsed = json.loads(
            server.stage_query(
                datasource_id="era5_wave_global", variables=["tp", ALIAS, "hs"]
            )
        )

        assert parsed["resolved_variables"] == {ALIAS: "hs"}
        # The alias and the real name it resolves to are not sent twice.
        assert parsed["query"]["variables"] == ["tp", "hs"]

    def test_ambiguous_alias_is_not_resolved(self, mock_conn, mock_stage):
        exc = _stage_bad_variable(ALIAS)
        mock_stage.side_effect = exc
        ds = _wave_datasource()
        ds.variables["hs_total"] = {"attrs": {"standard_name": ALIAS}}
        mock_conn.get_datasource.return_value = ds

        parsed = json.loads(
            server.stage_query(datasource_id="era5_wave_global", variables=[ALIAS])
        )

        assert "resolved_variables" not in parsed
        assert set(parsed["suggestions"][0]["did_you_mean"][:2]) == {"hs", "hs_total"}
        assert mock_stage.call_count == 1

    @pytest.mark.parametrize(
        "variables",
        [
            ["significant_wave_height"],  # close, but not an exact standard_name
            ["SEA_SURFACE_WAVE_SIGNIFICANT_HEIGHT"],  # exact match only
            [ALIAS, "chlorophyll"],  # one resolvable, one not: no partial retry
        ],
    )
    def test_no_exact_match_keeps_error_and_suggestions(
        self, mock_conn, mock_stage, variables
    ):
        mock_stage.side_effect = _stage_bad_variable(variables[-1])
        mock_conn.get_datasource.return_value = _wave_datasource()

        parsed = json.loads(
            server.stage_query(datasource_id="era5_wave_global", variables=variables)
        )

        assert "Variable(s) not in datasource" in parsed["error"]
        assert "suggestions" in parsed
        assert "resolved_variables" not in parsed
        assert mock_stage.call_count == 1

    def test_failed_retry_returns_original_error(self, mock_conn, mock_stage):
        # The retry reports a missing variable again (e.g. the engine still
        # rejects the resolved name): the original error stands.
        mock_stage.side_effect = [
            _stage_bad_variable(ALIAS),
            _stage_bad_variable("hs (retry)"),
        ]
        mock_conn.get_datasource.return_value = _wave_datasource()

        parsed = json.loads(
            server.stage_query(datasource_id="era5_wave_global", variables=[ALIAS])
        )

        assert "retry" not in parsed["error"]
        assert ALIAS in parsed["error"]
        assert parsed["suggestions"][0]["did_you_mean"][0] == "hs"
        assert parsed["query"]["variables"] == [ALIAS]
        assert "resolved_variables" not in parsed
        # The schema fetched for the resolution is reused for the suggestions.
        mock_conn.get_datasource.assert_called_once()

    def test_retry_timeout_is_not_masked(self, mock_conn, mock_stage):
        # The alias resolved; a timeout on the retry is the real problem and
        # must not be reported as a missing variable.
        timeout = server.GatewayTimeout("Datamesh staging timed out after 60s.")
        mock_stage.side_effect = [_stage_bad_variable(ALIAS), timeout]
        mock_conn.get_datasource.return_value = _wave_datasource()

        parsed = json.loads(
            server.stage_query(datasource_id="era5_wave_global", variables=[ALIAS])
        )

        assert parsed["error"] == str(timeout)
        assert "suggestions" not in parsed

    def test_local_export_resolves(self, mock_conn, mock_stage, tmp_path):
        mock_stage.side_effect = [_stage_bad_variable(ALIAS), make_stage()]
        mock_conn.get_datasource.return_value = _wave_datasource()
        mock_conn.query.return_value = _hs_dataset()

        parsed = json.loads(
            server.export_query(
                datasource_id="era5_wave_global",
                variables=[ALIAS],
                path=str(tmp_path / "out.nc"),
            )
        )

        assert parsed["resolved_variables"] == {ALIAS: "hs"}
        assert (tmp_path / "out.nc").exists()
        assert mock_conn.query.call_args.args[0].variables == ["hs"]

    def test_hosted_export_resolves(self, mock_conn):
        from oceanum_mcp.common.config import set_transport

        mock_conn.get_datasource.return_value = _wave_datasource()
        stage = {
            "container": "dataset",
            "size": 1234,
            "formats": ["nc"],
            "url": "https://datamesh.oceanum.io/oceanql/abc?sig=xyz",
        }
        bad = DatameshQueryError(
            f"Invalid variable selection - variable not found: '{ALIAS}'"
        )
        set_transport("http")
        try:
            with patch.object(
                server, "_download_stage", side_effect=[bad, stage]
            ) as download:
                parsed = json.loads(
                    server.export_query(
                        datasource_id="era5_wave_global", variables=[ALIAS]
                    )
                )
        finally:
            set_transport("stdio")

        assert parsed["download_url"].endswith("&f=nc")
        assert parsed["resolved_variables"] == {ALIAS: "hs"}
        assert parsed["query"]["variables"] == ["hs"]
        assert download.call_args.args[1].variables == ["hs"]


class TestLookupThreads:
    def test_timed_out_lookup_runs_in_a_daemon_thread(self, mock_conn):
        release = threading.Event()
        mock_conn.get_datasource.side_effect = _not_found("era5_wave_glob")
        mock_conn.get_catalog.side_effect = lambda **_: release.wait(5)

        try:
            with patch.object(server, "SUGGEST_TIMEOUT", 0.05):
                parsed = json.loads(server.get_datasource_info("era5_wave_glob"))
            lookups = [t for t in threading.enumerate() if t.name == "datamesh-suggest"]
            assert lookups
            # A daemon thread never delays interpreter exit.
            assert all(t.daemon for t in lookups)
        finally:
            release.set()
        assert parsed["error"] == "Datasource era5_wave_glob not found"

    def test_lookups_skipped_when_all_slots_busy(self, mock_conn):
        mock_conn.get_datasource.side_effect = _not_found("era5_wave_glob")

        with patch.object(server, "_SUGGEST_SLOTS", threading.BoundedSemaphore(1)):
            server._SUGGEST_SLOTS.acquire()
            parsed = json.loads(server.get_datasource_info("era5_wave_glob"))

        assert parsed["error"] == "Datasource era5_wave_glob not found"
        mock_conn.get_catalog.assert_not_called()

    def test_thread_start_failure_releases_the_slot(self, mock_conn):
        mock_conn.get_datasource.side_effect = _not_found("era5_wave_glob")
        slots = threading.BoundedSemaphore(1)

        with (
            patch.object(server, "_SUGGEST_SLOTS", slots),
            patch.object(
                threading.Thread, "start", side_effect=RuntimeError("no threads")
            ),
        ):
            parsed = json.loads(server.get_datasource_info("era5_wave_glob"))

        assert parsed["error"] == "Datasource era5_wave_glob not found"
        assert slots.acquire(blocking=False)  # the slot was given back
