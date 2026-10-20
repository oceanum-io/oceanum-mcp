"""Catalog output size and detail levels (OCE-309, OCE-360).

search_catalog returns compact summaries by default and get_datasource_info a
summary view; detail="full" on either returns the complete record. Fixtures
are real oceanum Datasource objects shaped like live Datamesh records: long
descriptions, many tags, GRIB-style per-variable attribute blobs,
model-parameter global attributes, and 100+ variables; plus one recorded live
record (tests/fixtures/oceanum_wave_nzpar_era5_grid.json, storage args
removed).
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oceanum.datamesh.datasource import Datasource

import oceanum_mcp.servers.datamesh.server as server
from oceanum_mcp.common import formatting
from oceanum_mcp.common.formatting import format_datasource, to_compact_json, to_json
from tests.test_servers.test_datamesh import _mock_catalog

N_VARIABLES = 120

_DESCRIPTION = (
    "## Global wave hindcast\n\n"
    + "Spectral wave model hindcast forced by ERA5 winds and sea ice, with "
    "partitioned swell and wind-sea parameters on a regular grid. " * 10
)


def _grib_pv() -> list[float]:
    # ERA5 model-level variables carry the 276 hybrid-level coefficients.
    return [i * 1.234567 for i in range(276)]


def _variable(i: int) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "units": "m",
        "long_name": f"significant height of partition {i}",
        "standard_name": "sea_surface_wave_significant_height",
        "valid_min": 0.0,
        "valid_max": 50.0,
        "comment": "Derived from the 2D spectrum by integration. " * 8,
    }
    for k in range(10):
        attrs[f"GRIB_param{k}"] = k * i
    if i % 3 == 0:
        attrs["GRIB_pv"] = _grib_pv()
    return {
        "dims": ["time", "latitude", "longitude"],
        "attrs": attrs,
        "dtype": "float32",
        "shape": [292897, 281, 301],
    }


def _schema(n_variables: int) -> dict[str, Any]:
    coord_attrs = {"units": "degrees_north", "long_name": "latitude", "axis": "Y"}
    return {
        "dims": {"time": 292897, "latitude": 281, "longitude": 301},
        "coords": {
            "time": {
                "dims": ["time"],
                "attrs": {},
                "dtype": "datetime64[ns]",
                "shape": [292897],
            },
            "latitude": {
                "dims": ["latitude"],
                "attrs": coord_attrs,
                "dtype": "float32",
                "shape": [281],
            },
            "longitude": {
                "dims": ["longitude"],
                "attrs": {**coord_attrs, "units": "degrees_east"},
                "dtype": "float32",
                "shape": [301],
            },
        },
        "data_vars": {f"hs_part{i}": _variable(i) for i in range(n_variables)},
        "attrs": {
            **{f"param_{k}": k * 0.1 for k in range(150)},
            "history": "regridded; " * 300,
            "Conventions": "CF-1.6",
        },
    }


def make_datasource(
    i: int = 0, *, detail: bool = True, n_variables: int = N_VARIABLES
) -> Datasource:
    """A Datasource as Datamesh returns it; detail=False mimics a catalog hit."""
    ds = Datasource(
        id=f"oceanum_wave_region{i}_era5_grid",
        name=f"Oceanum wave hindcast region {i}",
        description=_DESCRIPTION,
        geom={
            "type": "Polygon",
            "coordinates": [
                [[165, -48], [179, -48], [179, -34], [165, -34], [165, -48]]
            ],
        },
        tstart=datetime.datetime(1993, 1, 1, tzinfo=datetime.timezone.utc),
        tend=datetime.datetime(2026, 5, 1, tzinfo=datetime.timezone.utc),
        tags=[f"tag-number-{k}" for k in range(30)],
        labels=["oceanum", "hindcast"],
        info={
            "citation": "Oceanum Ltd (2024). Wave hindcast. " * 20,
            "provenance": {"steps": [f"step {k}" for k in range(50)]},
            "geospatial_lat_resolution": "0.05 degree",
        },
        schema=_schema(n_variables),
        coordinates={"t": "time", "x": "longitude", "y": "latitude"},
        driver="onzarr",
    )
    ds._detail = detail
    return ds


NZPAR_FIXTURE = (
    Path(__file__).parent.parent / "fixtures" / "oceanum_wave_nzpar_era5_grid.json"
)


def nzpar_datasource() -> Datasource:
    """The recorded live oceanum_wave_nzpar_era5_grid record, as
    get_datasource returns it."""
    ds = Datasource(**json.loads(NZPAR_FIXTURE.read_text()))
    ds._detail = True
    return ds


def _float32(x: float) -> float:
    # Live catalog bounds carry float32 noise, e.g. -38.599998474121094.
    return float(np.float32(x + 0.01))


def nz_wave_hit(i: int) -> Datasource:
    """A catalog hit shaped like the live NZ wave bbox search's (OCE-360):
    ~60-character name, long description, float32 bounds, a dozen tags, no
    schema."""
    lon, lat = 165.1 + i * 0.37, -47.3 + i * 0.29
    ds = Datasource(
        id=f"oceanum_wave_region{i}_era5_spec_nowcast",
        name=f"Oceanum region {i} 1km ECMWF wave nowcast spectra (hourly)",
        description=_DESCRIPTION,
        geom={
            "type": "Polygon",
            "coordinates": [
                [
                    [_float32(lon), _float32(lat)],
                    [_float32(lon + 1.3), _float32(lat)],
                    [_float32(lon + 1.3), _float32(lat + 1.4)],
                    [_float32(lon), _float32(lat + 1.4)],
                    [_float32(lon), _float32(lat)],
                ]
            ],
        },
        tstart=datetime.datetime(1993, 1, 1, tzinfo=datetime.timezone.utc),
        tend=datetime.datetime(2026, 5, 31, 21, tzinfo=datetime.timezone.utc),
        tags=["era5", "forecast", "hindcast", "model", "new_zealand", "nowcast"]
        + ["oceania", "oceanum", "spectra", "swan", "wave", "waves"],
        driver="onzarr",
    )
    ds._detail = False
    return ds


def _results_chars(out: dict[str, Any]) -> int:
    """Characters the "results" array occupies in the serialized response."""
    return len(to_compact_json(out)) - len(to_compact_json({**out, "results": []}))


def _search(mock_conn, datasources, **kwargs) -> dict[str, Any]:
    mock_conn.get_catalog.return_value = _mock_catalog(datasources)
    return json.loads(server.search_catalog(**kwargs))


class TestSearchCatalogSummary:
    def test_default_is_compact_summary(self, mock_conn):
        out = _search(mock_conn, [make_datasource(detail=False)], search="wave")
        hit = out["results"][0]
        assert set(hit) <= {
            "id",
            "name",
            "description",
            "bounds",
            "tstart",
            "tend",
            "variables",
            "variables_total",
            "tags",
            "tags_total",
        }
        for dumped in ("schema", "attributes", "info", "coordinates", "labels"):
            assert dumped not in hit
        assert hit["id"] == "oceanum_wave_region0_era5_grid"
        assert hit["bounds"] == [165.0, -48.0, 179.0, -34.0]
        assert hit["tstart"].startswith("1993-01-01")
        assert hit["tend"].startswith("2026-05-01")
        assert "get_datasource_info" in out["hint"]

    def test_description_truncated_with_ellipsis(self, mock_conn):
        hit = _search(mock_conn, [make_datasource(detail=False)])["results"][0]
        limit = formatting.SUMMARY_DESCRIPTION_CHARS
        assert len(hit["description"]) <= limit + 3
        assert hit["description"].endswith("...")
        assert "\n" not in hit["description"]

    def test_short_description_untouched(self, mock_conn):
        ds = make_datasource(detail=False)
        ds.description = "Short."
        hit = _search(mock_conn, [ds])["results"][0]
        assert hit["description"] == "Short."

    def test_variable_names_only_capped_with_total(self, mock_conn):
        hit = _search(mock_conn, [make_datasource(detail=False)])["results"][0]
        cap = formatting.SUMMARY_MAX_VARIABLES
        assert hit["variables"] == [f"hs_part{i}" for i in range(cap)]
        assert hit["variables_total"] == N_VARIABLES

    def test_variable_names_not_capped_when_few(self, mock_conn):
        ds = make_datasource(detail=False, n_variables=3)
        hit = _search(mock_conn, [ds])["results"][0]
        assert hit["variables"] == ["hs_part0", "hs_part1", "hs_part2"]
        assert "variables_total" not in hit

    def test_no_variables_key_without_schema(self, mock_conn):
        # Live catalog hits carry no schema; there are no names to show.
        ds = make_datasource(detail=False)
        ds.dataschema = None
        hit = _search(mock_conn, [ds])["results"][0]
        assert "variables" not in hit

    def test_tags_capped_with_total(self, mock_conn):
        hit = _search(mock_conn, [make_datasource(detail=False)])["results"][0]
        assert hit["tags"] == [f"tag-number-{k}" for k in range(5)]
        assert hit["tags_total"] == 30

    def test_few_tags_not_capped(self, mock_conn):
        ds = make_datasource(detail=False)
        ds.tags = ["wave", "era5"]
        hit = _search(mock_conn, [ds])["results"][0]
        assert hit["tags"] == ["wave", "era5"]
        assert "tags_total" not in hit

    def test_bounds_rounded_outward(self, mock_conn):
        ds = nz_wave_hit(0)
        hit = _search(mock_conn, [ds])["results"][0]
        xmin, ymin, xmax, ymax = ds.bounds
        # Never shrinks the coverage, never by more than the rounding step.
        assert xmin - 1e-4 < hit["bounds"][0] <= xmin
        assert ymin - 1e-4 < hit["bounds"][1] <= ymin
        assert xmax <= hit["bounds"][2] < xmax + 1e-4
        assert ymax <= hit["bounds"][3] < ymax + 1e-4
        for b in hit["bounds"]:
            assert round(b, 4) == b

    def test_exact_bounds_unchanged(self, mock_conn):
        hit = _search(mock_conn, [make_datasource(detail=False)])["results"][0]
        assert hit["bounds"] == [165.0, -48.0, 179.0, -34.0]

    def test_response_is_compact_json(self, mock_conn):
        mock_conn.get_catalog.return_value = _mock_catalog([nz_wave_hit(0)])
        out = server.search_catalog()
        assert "\n" not in out
        assert out == to_compact_json(json.loads(out))

    def test_full_detail_is_todays_output(self, mock_conn):
        ds = make_datasource(detail=False)
        out = _search(mock_conn, [ds], detail="full")
        assert out["results"] == [json.loads(to_json(format_datasource(ds)))]
        assert "hint" not in out


class TestSearchCatalogTotalBound:
    def test_summary_output_bounded_with_omitted_note(self, mock_conn):
        hits = [make_datasource(i, detail=False) for i in range(200)]
        out = _search(mock_conn, hits, limit=200)
        assert 0 < out["count"] < 200
        assert out["omitted"] == 200 - out["count"]
        assert f"{out['omitted']} more results" in out["note"]
        assert "refine the search" in out["note"]
        results_chars = _results_chars(out)
        assert results_chars <= server.SEARCH_BUDGET_CHARS["summary"]

    def test_full_output_bounded(self, mock_conn):
        # Live catalog hits carry no schema: ~4 kB full records here.
        hits = [make_datasource(i, detail=False) for i in range(50)]
        for hit in hits:
            hit.dataschema = None
        out = _search(mock_conn, hits, limit=50, detail="full")
        assert 0 < out["count"] < 50
        assert out["omitted"] == 50 - out["count"]
        results_chars = _results_chars(out)
        assert results_chars <= server.SEARCH_BUDGET_CHARS["full"]

    def test_full_record_larger_than_budget_still_shown_alone(self, mock_conn):
        # A record carrying a schema can exceed the whole budget by itself.
        hits = [make_datasource(i, detail=False) for i in range(3)]
        out = _search(mock_conn, hits, detail="full")
        assert out["count"] == 1
        assert out["omitted"] == 2

    def test_first_result_always_shown(self, mock_conn, monkeypatch):
        monkeypatch.setitem(server.SEARCH_BUDGET_CHARS, "summary", 10)
        hits = [make_datasource(i, detail=False) for i in range(3)]
        out = _search(mock_conn, hits)
        assert out["count"] == 1
        assert out["omitted"] == 2

    def test_no_omitted_key_when_everything_fits(self, mock_conn):
        hits = [make_datasource(i, detail=False) for i in range(5)]
        out = _search(mock_conn, hits)
        assert out["count"] == 5
        assert "omitted" not in out
        assert "note" not in out

    def test_omitted_note_keeps_limit_caveat(self, mock_conn, monkeypatch):
        monkeypatch.setitem(server.SEARCH_BUDGET_CHARS, "summary", 10)
        hits = [make_datasource(i, detail=False) for i in range(3)]
        out = _search(mock_conn, hits, limit=3)
        assert "further matches may exist" in out["note"]
        out = _search(mock_conn, hits, limit=10)
        assert "further matches may exist" not in out["note"]

    def test_omitted_note_replaces_raise_limit_advice(self, mock_conn, monkeypatch):
        # Raising limit cannot help when results are already being dropped.
        monkeypatch.setitem(server.SEARCH_BUDGET_CHARS, "summary", 10)
        hits = [make_datasource(i, detail=False) for i in range(3)]
        out = _search(mock_conn, hits, limit=3)
        assert "Raise limit" not in out["note"]
        assert "refine the search" in out["note"]

    def test_twenty_summaries_far_smaller_than_full(self, mock_conn):
        hits = [make_datasource(i, detail=False) for i in range(20)]
        mock_conn.get_catalog.return_value = _mock_catalog(hits)
        summary = server.search_catalog(limit=20)
        full_unbounded = to_json(
            {"count": 20, "results": [format_datasource(d) for d in hits]}
        )
        assert json.loads(summary)["count"] == 20
        assert len(summary) * 20 < len(full_unbounded)


class TestOutputSizeBudget:
    """OCE-360: responses short enough for clients to use inline."""

    # Before -> after: 17,070 -> 10,816 characters on these fixtures.
    NZ_WAVE_SEARCH_CHARS = 11_500
    # Before -> after: 20,911 -> 13,086 characters, fixture and live alike.
    NZPAR_SUMMARY_CHARS = 14_000

    def test_nz_wave_search(self, mock_conn):
        mock_conn.get_catalog.return_value = _mock_catalog(
            [nz_wave_hit(i) for i in range(20)]
        )
        out = server.search_catalog(search="wave", bbox=[165, -48, 179, -34])
        parsed = json.loads(out)
        assert parsed["count"] == 20
        assert parsed["text_matches"] == 20
        assert len(out) <= self.NZ_WAVE_SEARCH_CHARS

    def test_nzpar_summary(self, mock_conn):
        ds = nzpar_datasource()
        mock_conn.get_datasource.return_value = ds
        out = server.get_datasource_info(ds.id)
        assert len(out) <= self.NZPAR_SUMMARY_CHARS
        parsed = json.loads(out)
        assert len(parsed["variables"]) == 38
        assert parsed["variables"]["hs"] == {
            "dims": ["time", "latitude", "longitude"],
            "attrs": {
                "units": "m",
                "long_name": "significant height of wind and swell waves",
                "standard_name": "sea_surface_wave_significant_height",
            },
            "attrs_omitted": 2,
            "dtype": "float32",
            "shape": [292897, 281, 301],
        }
        assert "variables_omitted" not in parsed
        assert parsed["bounds"] == [165.0, -48.0, 180.0, -34.0]
        assert parsed["tstart"].startswith("1993-01-01")
        assert parsed["schema"]["dims"] == {
            "time": 292897,
            "latitude": 281,
            "longitude": 301,
        }

    def test_nzpar_full_keeps_every_attribute(self, mock_conn):
        ds = nzpar_datasource()
        mock_conn.get_datasource.return_value = ds
        parsed = json.loads(server.get_datasource_info(ds.id, detail="full"))
        hs = parsed["variables"]["hs"]
        assert hs["attrs"]["standard_name"] == "sea_surface_wave_significant_height"
        assert hs["attrs"]["valid_max"] == 50.0
        assert hs["shape"] == [292897, 281, 301]


class TestGetDatasourceInfoDetail:
    def _info(self, mock_conn, n_variables=10, **kwargs) -> dict[str, Any]:
        ds = make_datasource(n_variables=n_variables)
        mock_conn.get_datasource.return_value = ds
        return json.loads(server.get_datasource_info(ds.id, **kwargs))

    def test_full_is_todays_output(self, mock_conn):
        ds = make_datasource()
        mock_conn.get_datasource.return_value = ds
        out = json.loads(server.get_datasource_info(ds.id, detail="full"))
        assert out == json.loads(to_json(format_datasource(ds)))

    def test_default_keeps_core_metadata_per_variable(self, mock_conn):
        out = self._info(mock_conn)
        var = out["variables"]["hs_part0"]
        assert var["dims"] == ["time", "latitude", "longitude"]
        assert var["shape"] == [292897, 281, 301]
        assert var["dtype"] == "float32"
        assert var["attrs"] == {
            "units": "m",
            "long_name": "significant height of partition 0",
            "standard_name": "sea_surface_wave_significant_height",
        }
        assert var["attrs_omitted"] == 14

    def test_default_keeps_record_fields(self, mock_conn):
        out = self._info(mock_conn)
        for key in (
            "id",
            "name",
            "bounds",
            "tstart",
            "tend",
            "coordinates",
            "tags",
            "labels",
            "driver",
        ):
            assert key in out
        assert out["coordinates"] == {"t": "time", "x": "longitude", "y": "latitude"}
        assert out["schema"]["dims"] == {
            "time": 292897,
            "latitude": 281,
            "longitude": 301,
        }
        assert out["schema"]["coords"]["latitude"] == {
            "dims": ["latitude"],
            "attrs": {"units": "degrees_north", "long_name": "latitude"},
            "attrs_omitted": 1,
            "dtype": "float32",
            "shape": [281],
        }
        assert 'detail="full"' in out["note"]

    def test_default_caps_attribute_blobs(self, mock_conn):
        out = self._info(mock_conn)
        assert len(out["attributes"]) == formatting.INFO_MAX_GLOBAL_ATTRS
        assert out["attributes_omitted"] == 152 - formatting.INFO_MAX_GLOBAL_ATTRS
        assert len(out["info"]["citation"]) <= formatting.INFO_VALUE_CHARS + 3
        assert out["info"]["citation"].endswith("...")
        assert isinstance(out["info"]["provenance"], str)

    def test_default_drops_duplicate_schema_sections(self, mock_conn):
        out = self._info(mock_conn)
        assert set(out["schema"]) == {"dims", "coords"}

    def test_normal_record_not_truncated(self, mock_conn):
        out = self._info(mock_conn)
        assert len(out["variables"]) == 10
        assert "variables_omitted" not in out
        assert "too large" not in out["note"]

    def test_large_record_truncated_within_budget(self, mock_conn):
        n = 400
        ds = make_datasource(n_variables=n)
        mock_conn.get_datasource.return_value = ds
        raw = server.get_datasource_info(ds.id)
        out = json.loads(raw)
        assert len(raw) <= formatting.INFO_BUDGET_CHARS
        kept = list(out["variables"])
        # The leading variables are kept whole, in order; the rest counted.
        assert kept == [f"hs_part{i}" for i in range(len(kept))]
        assert out["variables"]["hs_part0"]["attrs"]["units"] == "m"
        assert out["variables_omitted"] == n - len(kept) > 0
        assert set(out["schema"]["coords"]) == {"time", "latitude", "longitude"}
        assert 'detail="full"' in out["note"]
        assert "variables_omitted" in out["note"]

    def test_description_clipped(self, mock_conn):
        out = self._info(mock_conn)
        assert len(out["description"]) <= formatting.INFO_DESCRIPTION_CHARS + 3
        assert out["description"].endswith("...")
        assert out["description"].startswith("## Global wave hindcast\n\n")

    def test_undetailed_record_keeps_schema_variables(self, mock_conn):
        # data_vars/attrs are only duplicates when variables/attributes exist.
        ds = make_datasource(n_variables=3, detail=False)
        mock_conn.get_datasource.return_value = ds
        out = json.loads(server.get_datasource_info(ds.id))
        assert "variables" not in out
        assert list(out["schema"]["data_vars"]) == ["hs_part0", "hs_part1", "hs_part2"]
        assert out["schema"]["data_vars"]["hs_part0"]["attrs"]["units"] == "m"
        assert len(out["schema"]["attrs"]) == formatting.INFO_MAX_GLOBAL_ATTRS

    def test_default_much_smaller_than_full(self, mock_conn):
        ds = make_datasource(n_variables=N_VARIABLES)
        mock_conn.get_datasource.return_value = ds
        default = server.get_datasource_info(ds.id)
        full = server.get_datasource_info(ds.id, detail="full")
        assert len(default) * 4 < len(full)


@pytest.mark.parametrize("tool", ["search_catalog", "get_datasource_info"])
async def test_detail_param_schema(tool):
    tools = {t.name: t for t in await server.mcp.list_tools()}
    prop = tools[tool].parameters["properties"]["detail"]
    assert prop["enum"] == ["summary", "full"]
    assert prop["default"] == "summary"
