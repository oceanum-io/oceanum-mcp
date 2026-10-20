"""Size guards on the data-moving tools (OCE-355, OCE-357, OCE-358).

export_query and query_data run their own size checks whether or not
stage_query was called; export caps are refusals of the transfer, not file
sizes to split around; and stage_query's inline advice counts records the way
query_data does.
"""

import json
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from oceanum.datamesh.query import Container

from oceanum_mcp.common.config import set_transport
from oceanum_mcp.common.formatting import human_bytes
from oceanum_mcp.servers.datamesh import server
from tests.conftest import make_stage

GB = 1_000_000_000
NOT_PER_PART = "not a way round it"


@pytest.fixture
def hosted() -> Iterator[None]:
    set_transport("http")
    yield
    set_transport("stdio")


@pytest.fixture
def roomy_disk() -> Iterator[None]:
    with patch.object(server.shutil, "disk_usage", return_value=MagicMock(free=10**15)):
        yield


def _small_dataset() -> xr.Dataset:
    return xr.Dataset(
        {"hs": (("time",), np.array([1.0, 2.0, 3.0]))},
        coords={"time": pd.date_range("2024-01-01", periods=3, freq="h")},
    )


def _stage(
    container: Container = Container.Dataset,
    size: int = 1000,
    dlen: int = 10,
    coordmap: dict[str, str] | None = None,
):
    """A stage whose variables have the given dims (coordmap codes)."""
    stage = make_stage(container, size=size, dlen=dlen)
    return stage.model_copy(
        update={
            "coordmap": {"hs": "t"} if coordmap is None else coordmap,
            "coordkeys": {"t": "time", "x": "longitude", "y": "latitude"},
        }
    )


def _download_stage(size: int, container: str = "dataset") -> dict:
    return {
        "container": container,
        "size": size,
        "formats": ["nc", "parquet", "csv"],
        "url": "https://datamesh.oceanum.io/oceanql/abc$?dl=SECRET",
    }


class TestExportQueryLocal:
    """OCE-355: export_query sizes the result itself, before writing."""

    def test_over_cap_without_stage_query_refused_before_writing(
        self, mock_conn, mock_stage, tmp_path
    ):
        # The observed T5 call: ~73 GB, overwrite=true, no prior stage_query.
        mock_stage.return_value = make_stage(Container.Dataset, size=73 * GB)
        dest = tmp_path / "era5.nc"

        parsed = json.loads(
            server.export_query(
                datasource_id="era5_wind10m_demo",
                path=str(dest),
                format="netcdf",
                overwrite=True,
                variables=["u10", "v10"],
                time_start="2020-01-01",
                time_end="2020-12-31T23",
            )
        )
        assert parsed["refused"] is True
        assert parsed["size_bytes"] == 73 * GB
        assert "73.0 GB" in parsed["message"]
        assert "Narrow the query" in parsed["message"]
        assert NOT_PER_PART in parsed["message"]
        mock_stage.assert_called_once()
        mock_conn.query.assert_not_called()
        assert list(tmp_path.iterdir()) == []

    def test_over_cap_refused_even_when_confirmed(
        self, mock_conn, mock_stage, tmp_path
    ):
        size = server.MAX_EXPORT_DATASET_BYTES + 1
        mock_stage.return_value = make_stage(Container.Dataset, size=size)

        parsed = json.loads(
            server.export_query(
                datasource_id="test-ds",
                path=str(tmp_path / "out.nc"),
                confirm_size_bytes=size,
            )
        )
        assert parsed["refused"] is True
        assert "confirmation_required" not in parsed
        mock_conn.query.assert_not_called()

    def test_large_unconfirmed_export_refused_before_writing(
        self, mock_conn, mock_stage, tmp_path, roomy_disk
    ):
        mock_stage.return_value = make_stage(Container.Dataset, size=5 * GB)
        dest = tmp_path / "out.nc"

        parsed = json.loads(
            server.export_query(datasource_id="test-ds", path=str(dest))
        )
        assert parsed["refused"] is True
        assert parsed["confirmation_required"] is True
        assert parsed["size_bytes"] == 5 * GB
        assert "ask the user" in parsed["message"]
        assert f"confirm_size_bytes={5 * GB}" in parsed["message"]
        assert "Narrowing it first" in parsed["message"]
        mock_conn.query.assert_not_called()
        assert not dest.exists()

    def test_wrong_confirmation_refused(
        self, mock_conn, mock_stage, tmp_path, roomy_disk
    ):
        mock_stage.return_value = make_stage(Container.Dataset, size=5 * GB)

        parsed = json.loads(
            server.export_query(
                datasource_id="test-ds",
                path=str(tmp_path / "out.nc"),
                confirm_size_bytes=4 * GB,
            )
        )
        assert parsed["confirmation_required"] is True
        assert f"confirm_size_bytes={4 * GB} does not match" in parsed["message"]
        assert f"confirm_size_bytes={5 * GB}." in parsed["message"]
        mock_conn.query.assert_not_called()

    def test_confirmed_large_export_proceeds(
        self, mock_conn, mock_stage, tmp_path, roomy_disk
    ):
        mock_stage.return_value = make_stage(Container.Dataset, size=5 * GB)
        mock_conn.query.return_value = _small_dataset()
        dest = tmp_path / "out.nc"

        parsed = json.loads(
            server.export_query(
                datasource_id="test-ds", path=str(dest), confirm_size_bytes=5 * GB
            )
        )
        assert "refused" not in parsed
        assert dest.exists()

    def test_at_threshold_needs_no_confirmation(
        self, mock_conn, mock_stage, tmp_path, roomy_disk
    ):
        mock_stage.return_value = make_stage(
            Container.Dataset, size=server.LARGE_EXPORT_BYTES
        )
        mock_conn.query.return_value = _small_dataset()
        dest = tmp_path / "out.nc"

        parsed = json.loads(
            server.export_query(datasource_id="test-ds", path=str(dest))
        )
        assert "refused" not in parsed
        assert dest.exists()

    def test_large_frame_needs_confirmation(
        self, mock_conn, mock_stage, tmp_path, roomy_disk
    ):
        size = 1_500_000_000
        mock_stage.return_value = make_stage(Container.DataFrame, size=size)

        parsed = json.loads(
            server.export_query(
                datasource_id="test-ds", path=str(tmp_path / "out.parquet")
            )
        )
        assert parsed["confirmation_required"] is True
        mock_conn.query.assert_not_called()

    def test_frame_cap_refusal_says_splitting_is_no_way_round(
        self, mock_conn, mock_stage, tmp_path
    ):
        mock_stage.return_value = make_stage(
            Container.DataFrame, size=server.MAX_EXPORT_FRAME_BYTES + 1
        )

        parsed = json.loads(
            server.export_query(
                datasource_id="test-ds", path=str(tmp_path / "out.parquet")
            )
        )
        assert parsed["refused"] is True
        assert NOT_PER_PART in parsed["message"]

    def test_limited_slice_confirmed_on_its_own_size(
        self, mock_conn, mock_stage, tmp_path, roomy_disk
    ):
        # The unlimited stage is 73 GB; limit keeps a 1.2 GB lazy slice.
        mock_stage.return_value = make_stage(Container.Dataset, size=73 * GB)
        times = pd.date_range("2020-01-01", periods=366, freq="D")
        mock_conn.query.return_value = xr.Dataset(
            {
                "u10": (
                    ("time", "y", "x"),
                    da.zeros((366, 1000, 1000), chunks=(1, 1000, 1000)),
                )
            },
            coords={"time": times},
        )
        slice_bytes = 150 * 1000 * 1000 * 8 + 150 * 8  # u10 + its time coord
        kwargs = {
            "datasource_id": "test-ds",
            "path": str(tmp_path / "out.nc"),
            "time_start": "2020-01-01",
            "time_end": "2020-12-31",
            "time_resolution": "1D",
            "limit": 150,
        }

        parsed = json.loads(server.export_query(**kwargs))
        assert parsed["confirmation_required"] is True
        assert parsed["size_bytes"] == 73 * GB
        assert parsed["export_size_bytes"] == slice_bytes
        assert f"confirm_size_bytes={slice_bytes}" in parsed["message"]

        # A confirmation below the slice (less tolerance) does not cover it.
        parsed = json.loads(server.export_query(**kwargs, confirm_size_bytes=1 * GB))
        assert parsed["confirmation_required"] is True
        assert list(tmp_path.iterdir()) == []


class TestExportQueryHosted:
    """OCE-355/357 on the hosted download path: no URL until confirmed."""

    def test_large_unconfirmed_download_refused_without_url(self, mock_conn, hosted):
        with patch.object(
            server, "_download_stage", return_value=_download_stage(5 * GB)
        ):
            raw = server.export_query(datasource_id="test-ds")
        parsed = json.loads(raw)
        assert parsed["refused"] is True
        assert parsed["confirmation_required"] is True
        assert parsed["size_bytes"] == 5 * GB
        assert f"confirm_size_bytes={5 * GB}" in parsed["message"]
        assert "download_url" not in parsed
        assert "SECRET" not in raw

    def test_wrong_confirmation_refused(self, mock_conn, hosted):
        with patch.object(
            server, "_download_stage", return_value=_download_stage(5 * GB)
        ):
            raw = server.export_query(datasource_id="test-ds", confirm_size_bytes=1)
        parsed = json.loads(raw)
        assert parsed["confirmation_required"] is True
        assert "SECRET" not in raw

    def test_confirmed_download_returns_url(self, mock_conn, hosted):
        with patch.object(
            server, "_download_stage", return_value=_download_stage(5 * GB)
        ):
            parsed = json.loads(
                server.export_query(datasource_id="test-ds", confirm_size_bytes=5 * GB)
            )
        assert "download_url" in parsed
        assert "refused" not in parsed

    def test_small_download_needs_no_confirmation(self, mock_conn, hosted):
        with patch.object(
            server, "_download_stage", return_value=_download_stage(1234)
        ):
            parsed = json.loads(server.export_query(datasource_id="test-ds"))
        assert "download_url" in parsed

    def test_cap_refusal_says_splitting_is_no_way_round(self, mock_conn, hosted):
        with patch.object(
            server, "_download_stage", return_value=_download_stage(73 * GB)
        ):
            parsed = json.loads(
                server.export_query(datasource_id="test-ds", confirm_size_bytes=73 * GB)
            )
        assert parsed["refused"] is True
        assert "confirmation_required" not in parsed
        assert NOT_PER_PART in parsed["message"]
        assert "Narrow the query" in parsed["message"]
        assert "library" not in parsed["message"]


class TestQueryDataFailsFast:
    """OCE-355: query_data refuses a huge result with stage_query's advice."""

    @pytest.mark.parametrize("container", [Container.Dataset, Container.DataFrame])
    def test_huge_result_refused_without_query(self, mock_conn, mock_stage, container):
        mock_stage.return_value = make_stage(container, size=73 * GB)

        parsed = json.loads(server.query_data(datasource_id="test-ds"))
        assert parsed["refused"] is True
        assert "73.0 GB" in parsed["message"]
        mock_conn.query.assert_not_called()

        advice = json.loads(server.stage_query(datasource_id="test-ds"))[
            "recommendation"
        ]
        assert parsed["message"].endswith(advice)

    def test_huge_result_on_hosted_gives_hosted_advice(
        self, mock_conn, mock_stage, hosted
    ):
        mock_stage.return_value = make_stage(Container.Dataset, size=73 * GB)

        parsed = json.loads(server.query_data(datasource_id="test-ds"))
        assert parsed["refused"] is True
        assert human_bytes(server.MAX_HOSTED_EXPORT_BYTES) in parsed["message"]
        assert NOT_PER_PART in parsed["message"]
        mock_conn.query.assert_not_called()

    def test_midsize_dataset_still_lazy(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(Container.Dataset, size=500_000_000)
        mock_conn.query.return_value = _small_dataset().chunk({"time": 1})

        parsed = json.loads(server.query_data(datasource_id="test-ds"))
        assert parsed["lazy"] is True

    def test_client_limit_keeps_lazy_slice_path(self, mock_conn, mock_stage):
        # A limit applied after resampling may still shrink a huge result.
        mock_stage.return_value = make_stage(Container.Dataset, size=73 * GB)
        mock_conn.query.return_value = _small_dataset().chunk({"time": 1})

        parsed = json.loads(
            server.query_data(
                datasource_id="test-ds",
                time_start="2024-01-01",
                time_end="2024-01-02",
                time_resolution="1h",
                limit=2,
            )
        )
        mock_conn.query.assert_called_once()
        assert "refused" not in parsed


class TestStageAdvice:
    def test_large_result_names_confirmation(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(Container.Dataset, size=5 * GB)

        rec = json.loads(server.stage_query(datasource_id="test-ds"))["recommendation"]
        assert f"confirm_size_bytes={5 * GB}" in rec

    def test_large_hosted_result_names_confirmation(
        self, mock_conn, mock_stage, hosted
    ):
        mock_stage.return_value = make_stage(Container.Dataset, size=5 * GB)

        rec = json.loads(server.stage_query(datasource_id="test-ds"))["recommendation"]
        assert f"confirm_size_bytes={5 * GB}" in rec

    @pytest.mark.parametrize("transport", ["stdio", "http"])
    def test_over_cap_says_splitting_is_no_way_round(
        self, mock_conn, mock_stage, transport
    ):
        # OCE-357: the observed hosted reply read the cap as a file size.
        mock_stage.return_value = make_stage(Container.Dataset, size=73 * GB)
        try:
            set_transport(transport)
            rec = json.loads(server.stage_query(datasource_id="test-ds"))[
                "recommendation"
            ]
        finally:
            set_transport("stdio")
        assert "refuse" in rec
        assert NOT_PER_PART in rec


class TestStageInlineAdvice:
    """OCE-358: the inline recommendation uses query_data's record cap."""

    def test_744_record_series_not_recommended_inline(self, mock_conn, mock_stage):
        # The observed point series: 744 hourly steps, 11.9 kB.
        mock_stage.return_value = _stage(size=11_928, dlen=744)

        rec = json.loads(server.stage_query(datasource_id="test-ds"))["recommendation"]
        assert "Small enough to return inline" not in rec
        assert "744 records" in rec
        assert "first 100" in rec
        assert "preview" in rec
        for alternative in ("time_resolution", "aggregate_operations", "export_query"):
            assert alternative in rec

    def test_at_record_cap_recommended_inline(self, mock_conn, mock_stage):
        mock_stage.return_value = _stage(size=1600, dlen=100)

        rec = json.loads(server.stage_query(datasource_id="test-ds"))["recommendation"]
        assert rec.startswith("Small enough to return inline")

    def test_follows_configured_record_cap(self, mock_conn, mock_stage, monkeypatch):
        monkeypatch.setenv("OCEANUM_MCP_MAX_INLINE_ROWS", "1000")
        mock_stage.return_value = _stage(size=11_928, dlen=744)

        rec = json.loads(server.stage_query(datasource_id="test-ds"))["recommendation"]
        assert rec.startswith("Small enough to return inline")

    def test_tabular_rows_counted(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(Container.DataFrame, size=5000, dlen=500)

        rec = json.loads(server.stage_query(datasource_id="test-ds"))["recommendation"]
        assert "500 records" in rec

    def test_reduction_carries_compute_caveat(self, mock_conn, mock_stage):
        # The observed timeout: a small global annual mean that could not be
        # computed before the client gave up.
        mock_stage.return_value = _stage(size=16_600, dlen=1, coordmap={"u10_mean": ""})

        rec = json.loads(
            server.stage_query(
                datasource_id="test-ds",
                time_start="2020-01-01",
                time_end="2020-12-31",
                aggregate_operations=["mean"],
            )
        )["recommendation"]
        assert rec.startswith("Small enough to return inline")
        assert "reads every native record" in rec
        assert "timeout" in rec

    def test_plain_query_has_no_compute_caveat(self, mock_conn, mock_stage):
        mock_stage.return_value = _stage(size=1000, dlen=10)

        rec = json.loads(server.stage_query(datasource_id="test-ds"))["recommendation"]
        assert "native record" not in rec


class TestConfirmationTolerance:
    """OCE-355 review: the confirmed size may come from a different endpoint
    (stage_query) than the one export_query sizes, so it need not match
    exactly; a confirmation covers any export up to the confirmed size."""

    @pytest.mark.parametrize(
        ("confirmed", "accepted"),
        [
            (5 * GB, True),
            (int(5 * GB * 0.96), True),  # within the 5% tolerance
            (int(5 * GB * 1.04), True),
            (int(5 * GB * 0.94), False),
            (int(5 * GB * 1.06), False),  # bounded above: no blanket value
            (10**12, False),
            (1, False),
        ],
    )
    def test_hosted(self, mock_conn, hosted, confirmed, accepted):
        with patch.object(
            server, "_download_stage", return_value=_download_stage(5 * GB)
        ):
            parsed = json.loads(
                server.export_query(
                    datasource_id="test-ds", confirm_size_bytes=confirmed
                )
            )
        assert ("download_url" in parsed) is accepted
        if not accepted:
            # The refusal states the exact value to pass.
            assert f"confirm_size_bytes={5 * GB}." in parsed["message"]

    def test_hosted_confirmation_refusal_reuses_refusal_shape(self, mock_conn, hosted):
        with patch.object(
            server, "_download_stage", return_value=_download_stage(5 * GB)
        ):
            parsed = json.loads(server.export_query(datasource_id="test-ds"))
        assert parsed["refused"] is True
        assert parsed["container"] == "dataset"
        assert parsed["size_human"] == "5.0 GB"
        assert parsed["query"]["datasource"] == "test-ds"

    def test_confirmation_cannot_be_avoided_by_splitting(self, mock_conn, hosted):
        with patch.object(
            server, "_download_stage", return_value=_download_stage(5 * GB)
        ):
            parsed = json.loads(server.export_query(datasource_id="test-ds"))
        assert "do not split it into parts" in parsed["message"]
        assert "ask the user before retrying" in parsed["message"]


class TestFlattenedExportConfirmation:
    """OCE-355 review: a dataset written as CSV/Parquet is confirmed on its
    flattened table size, which can far exceed the staged size."""

    def test_main_path_confirms_flattened_size(
        self, mock_conn, mock_stage, tmp_path, roomy_disk
    ):
        # 600 MB staged, but 70M rows x 2 columns x 8 bytes = 1.12 GB flat.
        rows = 70_000_000
        mock_stage.return_value = make_stage(Container.Dataset, size=600_000_000)
        mock_conn.query.return_value = xr.Dataset(
            {"hs": (("i",), da.zeros(rows, chunks=10_000_000))}
        )
        flat = rows * 2 * 8
        written = flat * 3  # CSV text, the estimate the caps use
        dest = tmp_path / "out.csv"

        parsed = json.loads(
            server.export_query(datasource_id="test-ds", path=str(dest), format="csv")
        )
        assert parsed["confirmation_required"] is True
        assert parsed["export_size_bytes"] == written
        assert parsed["format"] == "csv"
        assert f"confirm_size_bytes={written}." in parsed["message"]
        # Confirming the staged size does not cover the flattened table.
        parsed = json.loads(
            server.export_query(
                datasource_id="test-ds",
                path=str(dest),
                format="csv",
                confirm_size_bytes=600_000_000,
            )
        )
        assert parsed["confirmation_required"] is True
        assert list(tmp_path.iterdir()) == []

    def test_netcdf_of_same_dataset_needs_no_confirmation(
        self, mock_conn, mock_stage, tmp_path, roomy_disk
    ):
        mock_stage.return_value = make_stage(Container.Dataset, size=600_000_000)
        mock_conn.query.return_value = _small_dataset()

        parsed = json.loads(
            server.export_query(datasource_id="test-ds", path=str(tmp_path / "o.nc"))
        )
        assert "refused" not in parsed

    def test_limited_slice_confirms_flattened_size(
        self, mock_conn, mock_stage, tmp_path, roomy_disk
    ):
        # The limited slice is ~400 MB as a dataset, ~1.2 GB as a table.
        width = 333_334
        mock_stage.return_value = make_stage(Container.Dataset, size=73 * GB)
        mock_conn.query.return_value = xr.Dataset(
            {
                "hs": (
                    ("time", "i"),
                    da.zeros((366, width), chunks=(1, width)),
                )
            },
            coords={"time": pd.date_range("2020-01-01", periods=366, freq="D")},
        )
        flat = 150 * width * 3 * 8  # time, i and hs columns

        parsed = json.loads(
            server.export_query(
                datasource_id="test-ds",
                path=str(tmp_path / "out.parquet"),
                format="parquet",
                time_start="2020-01-01",
                time_end="2020-12-31",
                time_resolution="1D",
                limit=150,
            )
        )
        assert parsed["confirmation_required"] is True
        assert parsed["export_size_bytes"] == flat
        assert list(tmp_path.iterdir()) == []


class TestStageRecordCount:
    """OCE-358 review: records are counted as query_data counts them (the
    product of ALL a dataset's dims), and conservatively when unknown."""

    @staticmethod
    def _rec(**kwargs) -> str:
        return json.loads(server.stage_query(datasource_id="test-ds", **kwargs))[
            "recommendation"
        ]

    def test_gridded_variables_sharing_dims_counted_by_dlen(
        self, mock_conn, mock_stage
    ):
        # A 9x9 grid at one time: u10 on (t, y, x), a mask on (y, x).
        mock_stage.return_value = _stage(dlen=81, coordmap={"u10": "tyx", "lsm": "yx"})
        assert self._rec().startswith("Small enough to return inline")

    def test_gridded_over_record_cap(self, mock_conn, mock_stage):
        mock_stage.return_value = _stage(dlen=60_264, coordmap={"u10": "tyx"})
        rec = self._rec()
        assert "60264 records" in rec
        assert "Small enough" not in rec

    def test_gridded_variables_on_different_dims_not_called_inline(
        self, mock_conn, mock_stage
    ):
        # 50 points per variable, but flattened to records over the union of
        # (t, x) and (t, y) the count is unknown and can exceed the cap.
        mock_stage.return_value = _stage(dlen=50, coordmap={"a": "tx", "b": "ty"})
        rec = self._rec()
        assert "Small enough" not in rec
        assert "not known" in rec
        assert "preview" in rec

    def test_dataset_without_coordmap_not_called_inline(self, mock_conn, mock_stage):
        mock_stage.return_value = _stage(dlen=10, coordmap={})
        assert "Small enough" not in self._rec()

    def test_limit_on_time_series_counted(self, mock_conn, mock_stage):
        mock_stage.return_value = _stage(dlen=366)
        resampled = {
            "time_start": "2020-01-01",
            "time_end": "2020-12-31",
            "time_resolution": "1D",
        }
        assert self._rec(**resampled, limit=50).startswith("Small enough")
        assert "200 records" in self._rec(**resampled, limit=200)

    def test_limit_above_cap_on_grid_still_a_preview(self, mock_conn, mock_stage):
        mock_stage.return_value = _stage(dlen=60_264, coordmap={"u10": "tyx"})
        rec = self._rec(
            time_start="2020-01-01",
            time_end="2020-01-31",
            time_resolution="1D",
            limit=500,
        )
        assert "60264 records" in rec

    def test_small_limit_on_grid_is_unknown(self, mock_conn, mock_stage):
        mock_stage.return_value = _stage(dlen=60_264, coordmap={"u10": "tyx"})
        rec = self._rec(
            time_start="2020-01-01",
            time_end="2020-01-31",
            time_resolution="1D",
            limit=10,
        )
        assert "not known" in rec

    def test_limited_table_counted(self, mock_conn, mock_stage):
        mock_stage.return_value = _stage(Container.DataFrame, dlen=5000, coordmap={})
        rec = self._rec(
            time_start="2020-01-01",
            time_end="2020-12-31",
            aggregate_operations=["mean"],
            limit=20,
        )
        assert rec.startswith("Small enough")

    @pytest.mark.parametrize("size", [500_000_000, 5 * GB])
    def test_compute_caveat_on_larger_branches(self, mock_conn, mock_stage, size):
        mock_stage.return_value = _stage(size=size, dlen=10**8)
        rec = self._rec(
            time_start="2020-01-01", time_end="2020-12-31", time_resolution="1D"
        )
        assert "reads every native record" in rec


class TestQueryDataRefusalShape:
    def test_refusal_reports_variable_dims_from_stage(self, mock_conn, mock_stage):
        mock_stage.return_value = _stage(
            size=73 * GB, dlen=9_119_900_160, coordmap={"u10": "tyx", "v10": "tyx"}
        )

        parsed = json.loads(server.query_data(datasource_id="test-ds"))
        assert parsed["refused"] is True
        assert parsed["variable_dims"] == {
            "u10": ["time", "latitude", "longitude"],
            "v10": ["time", "latitude", "longitude"],
        }
        mock_conn.query.assert_not_called()


class TestWrittenBytesConfirmation:
    """OCE-355 review round 2: the 1 GB threshold, the refusal and the check
    all use the bytes written in the chosen format, exactly on stdio."""

    def test_local_confirmation_must_match_exactly(
        self, mock_conn, mock_stage, tmp_path, roomy_disk
    ):
        mock_stage.return_value = make_stage(Container.Dataset, size=5 * GB)

        parsed = json.loads(
            server.export_query(
                datasource_id="test-ds",
                path=str(tmp_path / "out.nc"),
                confirm_size_bytes=5 * GB + 1,
            )
        )
        assert parsed["confirmation_required"] is True
        assert parsed["export_size_bytes"] == 5 * GB
        mock_conn.query.assert_not_called()

    def test_local_table_as_csv_confirmed_on_text_size(
        self, mock_conn, mock_stage, tmp_path, roomy_disk
    ):
        # 400 MB staged: below the threshold as Parquet, ~1.2 GB as CSV.
        mock_stage.return_value = make_stage(Container.DataFrame, size=400_000_000)
        mock_conn.query.return_value = pd.DataFrame({"x": [1.0]})

        as_csv = json.loads(
            server.export_query(
                datasource_id="test-ds", path=str(tmp_path / "o.csv"), format="csv"
            )
        )
        assert as_csv["confirmation_required"] is True
        assert as_csv["export_size_bytes"] == 1_200_000_000
        assert as_csv["size_bytes"] == 400_000_000

        as_parquet = json.loads(
            server.export_query(
                datasource_id="test-ds", path=str(tmp_path / "o.parquet")
            )
        )
        assert "refused" not in as_parquet

        confirmed = json.loads(
            server.export_query(
                datasource_id="test-ds",
                path=str(tmp_path / "o.csv"),
                format="csv",
                confirm_size_bytes=1_200_000_000,
            )
        )
        assert "refused" not in confirmed

    def test_hosted_csv_confirmed_on_text_size(self, mock_conn, hosted):
        stage = _download_stage(400_000_000)
        with patch.object(server, "_download_stage", return_value=stage):
            as_csv = json.loads(
                server.export_query(datasource_id="test-ds", format="csv")
            )
            as_nc = json.loads(server.export_query(datasource_id="test-ds"))
            confirmed = json.loads(
                server.export_query(
                    datasource_id="test-ds",
                    format="csv",
                    confirm_size_bytes=1_200_000_000,
                )
            )
        assert as_csv["confirmation_required"] is True
        assert as_csv["export_size_bytes"] == 1_200_000_000
        assert "download_url" not in as_csv
        assert "download_url" in as_nc
        assert "download_url" in confirmed

    def test_stage_advice_names_the_value_export_checks(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(Container.Dataset, size=5 * GB)

        rec = json.loads(server.stage_query(datasource_id="test-ds"))["recommendation"]
        assert f"confirm_size_bytes={5 * GB} to export_query for NetCDF" in rec
        assert "flattened" in rec
        assert "the refusal states the value to confirm" in rec

    def test_hosted_stage_advice_names_csv_exception(
        self, mock_conn, mock_stage, hosted
    ):
        mock_stage.return_value = make_stage(Container.Dataset, size=5 * GB)

        rec = json.loads(server.stage_query(datasource_id="test-ds"))["recommendation"]
        assert "for any format but CSV" in rec

    def test_midsize_stage_warns_csv_may_need_confirmation(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(Container.DataFrame, size=400_000_000)

        rec = json.loads(server.stage_query(datasource_id="test-ds"))["recommendation"]
        assert "confirm" in rec
        assert "as CSV" in rec

    def test_small_midsize_stage_has_no_confirmation_note(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(Container.DataFrame, size=100_000_000)

        rec = json.loads(server.stage_query(datasource_id="test-ds"))["recommendation"]
        assert "confirm" not in rec


class TestLimitDims:
    """OCE-358 review round 2: limit is counted only on recognised dims."""

    @staticmethod
    def _rec(limit: int) -> str:
        return json.loads(
            server.stage_query(
                datasource_id="test-ds",
                time_start="2020-01-01",
                time_end="2020-12-31",
                time_resolution="1D",
                limit=limit,
            )
        )["recommendation"]

    def test_ensemble_only_counted(self, mock_conn, mock_stage):
        mock_stage.return_value = _stage(dlen=500, coordmap={"hs": "e"})
        assert self._rec(50).startswith("Small enough")

    def test_unrecognised_dim_not_counted(self, mock_conn, mock_stage):
        mock_stage.return_value = _stage(dlen=500, coordmap={"hs": "z"})
        assert "not known" in self._rec(50)


def test_table_refusal_has_no_empty_variable_dims(mock_conn, mock_stage):
    mock_stage.return_value = _stage(Container.DataFrame, size=73 * GB, coordmap={})

    parsed = json.loads(server.query_data(datasource_id="test-ds"))
    assert parsed["refused"] is True
    assert "variable_dims" not in parsed


class TestStageRecordCapConfig:
    """OCE-358 review round 3: a bad OCEANUM_MCP_MAX_INLINE_ROWS must not
    break the dry run."""

    @pytest.mark.parametrize("raw", ["abc", "0", "-5"])
    def test_invalid_cap_falls_back_to_default(
        self, mock_conn, mock_stage, monkeypatch, raw
    ):
        monkeypatch.setenv("OCEANUM_MCP_MAX_INLINE_ROWS", raw)
        mock_stage.return_value = _stage(size=1000, dlen=10)

        parsed = json.loads(server.stage_query(datasource_id="test-ds"))
        assert parsed["staged"] is True
        assert parsed["recommendation"].startswith("Small enough")
        assert "OCEANUM_MCP_MAX_INLINE_ROWS" in parsed["config_warning"]
        assert "default (100)" in parsed["config_warning"]

    def test_invalid_cap_still_counts_against_default(
        self, mock_conn, mock_stage, monkeypatch
    ):
        monkeypatch.setenv("OCEANUM_MCP_MAX_INLINE_ROWS", "abc")
        mock_stage.return_value = _stage(size=11_928, dlen=744)

        rec = json.loads(server.stage_query(datasource_id="test-ds"))["recommendation"]
        assert "744 records" in rec
        assert "first 100" in rec

    def test_valid_cap_has_no_config_warning(self, mock_conn, mock_stage):
        mock_stage.return_value = _stage(size=1000, dlen=10)

        parsed = json.loads(server.stage_query(datasource_id="test-ds"))
        assert "config_warning" not in parsed


class TestMidsizeFlattenNote:
    """OCE-355 review round 3: a gridded result may exceed 1 GB as a
    flattened table even when 3x its staged size does not."""

    def test_local_dataset_notes_flattened_table(self, mock_conn, mock_stage):
        mock_stage.return_value = _stage(size=100_000_000, dlen=10**7)

        rec = json.loads(server.stage_query(datasource_id="test-ds"))["recommendation"]
        assert "as a flattened CSV/Parquet table" in rec
        assert "as CSV (about 3x)" not in rec

    def test_hosted_dataset_has_no_flatten_note(self, mock_conn, mock_stage, hosted):
        mock_stage.return_value = _stage(size=100_000_000, dlen=10**7)

        rec = json.loads(server.stage_query(datasource_id="test-ds"))["recommendation"]
        assert "confirm" not in rec
