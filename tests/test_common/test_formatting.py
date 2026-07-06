"""Tests for shared formatting and summarization helpers."""

import numpy as np
import pandas as pd
import xarray as xr

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


def test_warnings_attached():
    out = summarize_data(None, warnings=["row cap hit"])
    assert out["warnings"] == ["row cap hit"]
