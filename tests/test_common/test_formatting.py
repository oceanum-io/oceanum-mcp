"""Tests for shared formatting and summarization helpers."""

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from oceanum_mcp.common.config import set_transport
from oceanum_mcp.common.formatting import human_bytes, summarize_data


def _dataset(n: int = 3) -> xr.Dataset:
    return xr.Dataset(
        {"hs": (("time",), np.arange(n, dtype=float))},
        coords={"time": pd.date_range("2024-01-01", periods=n, freq="h")},
    )


def test_human_bytes():
    assert human_bytes(512) == "512 B"
    assert human_bytes(1_500_000) == "1.5 MB"
    assert human_bytes(2_000_000_000) == "2.0 GB"


def test_none_is_no_data():
    assert summarize_data(None)["status"] == "no_data"


def test_small_dataframe_not_truncated():
    out = summarize_data(pd.DataFrame({"x": [1, 2]}))
    assert out["container"] == "dataframe"
    assert out["truncated"] is False
    assert out["data"] == [{"x": 1}, {"x": 2}]


def test_large_dataframe_truncated_with_note():
    out = summarize_data(pd.DataFrame({"x": range(100)}), max_rows=10)
    assert out["truncated"] is True
    assert len(out["data"]) == 10
    assert "100 rows" in out["note"]


def test_small_dataset_records_include_coordinates():
    out = summarize_data(_dataset())
    assert out["container"] == "dataset"
    assert out["lazy"] is False
    assert "time" in out["data"][0]
    assert out["dims"] == {"time": 3}


def test_lazy_dataset_flagged_without_values():
    out = summarize_data(_dataset().chunk({"time": 1}))
    assert out["lazy"] is True
    assert "data" not in out
    assert "note" in out


def test_lazy_dataset_chunked_coords_not_computed():
    # Chunked (remote) coordinates must not be downloaded for first/last.
    ds = xr.Dataset(
        {"v": (("x",), np.zeros(10))},
        coords={"lon2d": (("x",), np.arange(10.0))},
    ).chunk({"x": 2})
    out = summarize_data(ds)
    assert "first" not in out["coords"]["lon2d"]
    assert out["coords"]["lon2d"]["size"] == 10


def test_midsize_eager_dataset_includes_values():
    # Eager datasets over 1 MB must still show a preview (they are already
    # downloaded); only lazy datasets omit values.
    big = xr.Dataset({"v": (("x",), np.zeros(300_000))})
    assert big.nbytes > 1_000_000
    out = summarize_data(big)
    assert out["lazy"] is False
    assert len(out["data"]) == 100  # DEFAULT_MAX_INLINE_ROWS
    assert out["truncated"] is True


def test_structure_only_mode_has_no_truncation_flags():
    out = summarize_data(pd.DataFrame({"x": range(200)}), max_rows=0)
    assert "data" not in out
    assert "truncated" not in out
    # A completed export must never look partial: no truncation note, full count.
    assert "note" not in out
    assert out["rows"] == 200


def test_default_inline_cap_is_configurable():
    df = pd.DataFrame({"x": range(500)})
    with patch.dict(os.environ, {"OCEANUM_MCP_MAX_INLINE_ROWS": "250"}, clear=False):
        out = summarize_data(df)
    assert len(out["data"]) == 250
    assert out["truncated"] is True


def test_truncation_hint_export_wording_by_transport():
    df = pd.DataFrame({"x": range(500)})
    try:
        set_transport("http")
        http_out = summarize_data(df)
    finally:
        set_transport("stdio")
    # Hosted export_query returns a download link; stdio writes a file. Both
    # name export_query, with transport-appropriate wording.
    assert "export_query" in http_out["note"]
    assert "download link" in http_out["note"]
    stdio_out = summarize_data(df)
    assert "export_query" in stdio_out["note"]
    assert "write the full result to a file" in stdio_out["note"]


def test_geodataframe_records_use_wkt():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame({"name": ["a"]}, geometry=[Point(1.0, 2.0)])
    out = summarize_data(gdf)
    assert out["container"] == "geodataframe"
    assert out["data"][0]["geometry"].startswith("POINT")
    assert out["data"][0]["name"] == "a"


def test_warnings_attached():
    out = summarize_data(None, warnings=["row cap hit"])
    assert out["warnings"] == ["row cap hit"]
