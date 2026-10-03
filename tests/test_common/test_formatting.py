"""Tests for shared formatting and summarization helpers."""

import json
import os
from unittest.mock import patch

import dask.array as da
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


# --- OCE-306: previews are machine-readable and flagged unfit for statistics --


def _assert_stats_warning(note: str) -> None:
    assert "NOT suitable for statistics" in note
    for pointer in ("aggregate_operations", "time_resolution", "export_query"):
        assert pointer in note


def test_truncated_frame_preview_fields():
    out = summarize_data(pd.DataFrame({"x": range(100)}), max_rows=10)
    assert out["preview"] is True
    assert out["returned"] == 10
    assert out["total"] == 100
    _assert_stats_warning(out["note"])
    # Existing fields are kept for current consumers.
    assert out["truncated"] is True
    assert out["rows"] == 100


def test_complete_frame_is_not_a_preview():
    out = summarize_data(pd.DataFrame({"x": [1, 2]}))
    assert out["preview"] is False
    assert out["returned"] == 2
    assert out["total"] == 2
    assert "note" not in out


def test_truncated_eager_dataset_preview_fields():
    out = summarize_data(_dataset(50), max_rows=10)
    assert out["preview"] is True
    assert out["returned"] == 10
    assert out["total"] == 50
    assert out["truncated"] is True
    _assert_stats_warning(out["note"])


def test_complete_eager_dataset_is_not_a_preview():
    out = summarize_data(_dataset(3))
    assert out["preview"] is False
    assert out["returned"] == 3
    assert out["total"] == 3
    assert "note" not in out


def test_lazy_dataset_preview_fields():
    ds = xr.Dataset(
        {"hs": (("time", "x"), np.zeros((4, 5)))},
        coords={"time": pd.date_range("2024-01-01", periods=4, freq="h")},
    ).chunk({"time": 1})
    out = summarize_data(ds)
    assert out["preview"] is True
    assert out["returned"] == 0
    # Record count is known from the dims without downloading values.
    assert out["total"] == 20
    assert out["lazy"] is True
    _assert_stats_warning(out["note"])


def test_structure_only_mode_has_no_preview_fields():
    # After an export the written file is complete: nothing may flag a preview.
    out = summarize_data(pd.DataFrame({"x": range(200)}), max_rows=0)
    for key in ("preview", "returned", "total"):
        assert key not in out


def _zero_d_dataset() -> xr.Dataset:
    # Shape of an aggregate_operations result collapsed over space and time.
    return xr.Dataset(
        {
            "hs": ((), np.float32(1.5), {"units": "m", "long_name": "Hs"}),
            "tp": ((), np.nan),
            "count": ((), np.int64(7)),
            "peak": ((), np.datetime64("2024-01-02T03:00")),
        },
        coords={"time": pd.Timestamp("2024-01-01")},
    )


def test_zero_d_dataset_returns_scalar_records():
    # OCE-320: to_dataframe raised "no valid index for a 0-dimensional object".
    out = summarize_data(_zero_d_dataset())
    assert out["container"] == "dataset"
    assert out["dims"] == {}
    assert out["lazy"] is False
    assert out["data"] == [
        {"name": "hs", "value": 1.5, "units": "m", "long_name": "Hs"},
        {"name": "tp", "value": None},
        {"name": "count", "value": 7},
        {"name": "peak", "value": "2024-01-02T03:00:00.000"},
    ]
    assert out["truncated"] is False
    assert out["preview"] is False
    assert out["returned"] == 1
    assert out["total"] == 1
    assert "note" not in out
    # Scalar coords stay in the coords summary.
    assert "time" in out["coords"]
    # The whole summary must serialize as strict JSON (no NaN, no numpy types).
    json.dumps(out, allow_nan=False)


def test_zero_d_dataset_structure_only_mode():
    out = summarize_data(_zero_d_dataset(), max_rows=0)
    for key in ("data", "truncated", "preview", "returned", "total"):
        assert key not in out


def test_mixed_zero_d_and_dimensioned_variables_broadcast():
    # A 0-d variable alongside dimensioned ones already goes through
    # to_dataframe (it is broadcast per record); keep that behaviour.
    ds = _dataset(2).assign(hs_mean=((), 1.5))
    out = summarize_data(ds)
    assert [r["hs_mean"] for r in out["data"]] == [1.5, 1.5]
    assert out["returned"] == out["total"] == 2


def test_zero_d_values_keep_full_precision():
    # DataFrame.to_json rounds to 10 decimals: a tiny mean must not become 0.0.
    ds = xr.Dataset(
        {
            "small": ((), 5e-12),
            "precise": ((), 1.23456789012345),
            "inf": ((), np.inf),
            "flag": ((), np.bool_(True)),
            "missing": ((), np.datetime64("NaT", "ns")),
        }
    )
    values = {r["name"]: r["value"] for r in summarize_data(ds)["data"]}
    assert values == {
        "small": 5e-12,
        "precise": 1.23456789012345,
        "inf": None,
        "flag": True,
        "missing": None,
    }


def test_zero_d_empty_or_none_attrs_omitted():
    ds = xr.Dataset({"x": ((), 1.0, {"units": None, "long_name": ""})})
    assert summarize_data(ds)["data"] == [{"name": "x", "value": 1.0}]


def _lazy_zero_d_dataset() -> xr.Dataset:
    # Dataset.chunk() is a no-op without dims, so wrap the values in dask.
    ds = _zero_d_dataset()
    for name in ds.data_vars:
        ds[name] = ds[name].copy(data=da.from_array(ds[name].values))
    assert ds["hs"].chunks is not None
    return ds


def test_lazy_zero_d_dataset_is_computed():
    # query_data picks use_dask from the staged size, so an aggregate can come
    # back dask-backed; a 0-d result is tiny and is returned as values.
    ds = _lazy_zero_d_dataset()
    out = summarize_data(ds)
    assert out["lazy"] is False
    assert out["data"][0] == {
        "name": "hs",
        "value": 1.5,
        "units": "m",
        "long_name": "Hs",
    }
    assert (out["preview"], out["returned"], out["total"]) == (False, 1, 1)
    # The caller's dataset is not loaded in place.
    assert ds["hs"].chunks is not None


def test_lazy_zero_d_structure_only_not_computed():
    out = summarize_data(_lazy_zero_d_dataset(), max_rows=0)
    assert out["lazy"] is True
    assert "data" not in out


def test_zero_d_dataset_without_variables():
    out = summarize_data(xr.Dataset(coords={"time": pd.Timestamp("2024-01-01")}))
    assert out["data"] == []
    assert (out["preview"], out["returned"], out["total"]) == (False, 1, 1)


# --- OCE-325: inline records keep full float precision ------------------------

# DataFrame.to_json rounds floats to 10 decimal places by default, and its
# maximum (double_precision=15) is still decimal places, not significant
# digits: 1.234567890123e-14 would come back as 1.2e-14.
_PRECISE = [5e-12, 123456.123456789012, 1.23456789012345, 1.234567890123e-14]


def test_frame_records_keep_full_float_precision():
    df = pd.DataFrame({"v": _PRECISE + [np.nan, np.inf, -np.inf]})
    out = summarize_data(df)
    assert [r["v"] for r in out["data"]] == _PRECISE + [None, None, None]
    # Strict JSON (no NaN/Infinity tokens), and the values survive the
    # tool's own serialization exactly.
    text = json.dumps(out, allow_nan=False)
    assert [r["v"] for r in json.loads(text)["data"]] == _PRECISE + [None] * 3


def test_dataset_records_keep_full_float_precision():
    # Distinct values per column, so a misaligned column cannot pass.
    coord = [1e-13, 2.000000000000001, 3.5, 98765.43210987654]
    ds = xr.Dataset({"v": (("x",), np.array(_PRECISE))}, coords={"x": coord})
    data = summarize_data(ds)["data"]
    assert [r["v"] for r in data] == _PRECISE
    # Float coordinates are record columns too.
    assert [r["x"] for r in data] == coord


def test_float32_records_use_shortest_float32_repr():
    # A float32 widened to float64 carries binary-noise digits
    # (0.1 -> 0.10000000149011612); records use the shortest decimal that
    # round-trips to the same float32 instead.
    f32 = np.array([0.1, 2.5e-7, 1.2345679, 5e-12, np.nan], dtype=np.float32)
    expected = [0.1, 2.5e-07, 1.2345679, 5e-12, None]
    frame = summarize_data(pd.DataFrame({"v": f32}))["data"]
    assert [r["v"] for r in frame] == expected
    dataset = summarize_data(xr.Dataset({"v": (("x",), f32)}))["data"]
    assert [r["v"] for r in dataset] == expected
    for got, orig in zip(expected[:-1], f32[:-1]):
        assert np.float32(got) == orig


def test_zero_d_float32_uses_shortest_float32_repr():
    ds = xr.Dataset({"v": ((), np.float32(0.1))})
    assert summarize_data(ds)["data"] == [{"name": "v", "value": 0.1}]


@pytest.mark.parametrize(
    "values",
    [
        pd.array([5e-12, None], dtype="Float64"),
        pd.arrays.SparseArray([5e-12, np.nan]),
    ],
    ids=["Float64", "sparse"],
)
def test_extension_float_records_keep_full_precision(values):
    df = pd.DataFrame({"v": values})
    assert [r["v"] for r in summarize_data(df)["data"]] == [5e-12, None]


def test_arrow_float_records_keep_full_precision():
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"v": pd.array([5e-12, None], dtype="double[pyarrow]")})
    assert [r["v"] for r in summarize_data(df)["data"]] == [5e-12, None]


def test_narrow_extension_and_float16_use_shortest_repr():
    df = pd.DataFrame(
        {
            "f32": pd.array([0.1, None], dtype="Float32"),
            "f16": np.array([0.1, np.inf], dtype=np.float16),
            "s32": pd.arrays.SparseArray(np.array([0.1, np.nan], dtype=np.float32)),
        }
    )
    assert summarize_data(df)["data"] == [
        {"f32": 0.1, "f16": 0.1, "s32": 0.1},
        {"f32": None, "f16": None, "s32": None},
    ]


def test_object_column_floats_keep_full_precision():
    # Mixed object columns: float cells are exact, the rest keep to_json's
    # conversion.
    df = pd.DataFrame({"o": pd.Series([5e-12, "x", None, np.nan, 7], dtype=object)})
    assert [r["o"] for r in summarize_data(df)["data"]] == [5e-12, "x", None, None, 7]


def test_non_string_column_labels():
    # Keys are str(label), as in the summary's column list.
    df = pd.DataFrame({0: [5e-12], 1.5: [123456.123456789012], "a": [1]})
    out = summarize_data(df)
    assert out["data"] == [{"0": 5e-12, "1.5": 123456.123456789012, "a": 1}]
    assert [c["name"] for c in out["columns"]] == list(out["data"][0])


@pytest.mark.parametrize("labels", [["a", "a"], [1, "1"]], ids=["dup", "str-collision"])
def test_colliding_column_labels_raise(labels):
    # Duplicate labels already made to_json raise; labels that collide only
    # as JSON keys must not silently drop a column either.
    df = pd.DataFrame([[5e-12, 2.0]], columns=labels)
    with pytest.raises(ValueError, match="unique"):
        summarize_data(df)


def test_non_float_records_unchanged():
    # Only float columns are re-serialized; everything else keeps the
    # to_json conversion (ISO datetimes/durations, NaT -> null).
    df = pd.DataFrame(
        {
            "t": pd.to_datetime(["2024-01-02T03:00", None]),
            "d": pd.to_timedelta(["1h", None]),
            "i": [7, 8],
            "b": [True, False],
            "s": ["a", None],
            "f": [5e-12, 1.5],
        }
    )
    assert summarize_data(df)["data"] == [
        {
            "t": "2024-01-02T03:00:00.000",
            "d": "P0DT1H0M0S",
            "i": 7,
            "b": True,
            "s": "a",
            "f": 5e-12,
        },
        {"t": None, "d": None, "i": 8, "b": False, "s": None, "f": 1.5},
    ]
