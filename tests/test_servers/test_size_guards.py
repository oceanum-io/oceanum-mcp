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

        # Confirming the unlimited staged size does not confirm the slice.
        parsed = json.loads(server.export_query(**kwargs, confirm_size_bytes=73 * GB))
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
        assert "oceanum Python library" in parsed["message"]


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
        mock_stage.return_value = make_stage(Container.Dataset, size=11_928, dlen=744)

        rec = json.loads(server.stage_query(datasource_id="test-ds"))["recommendation"]
        assert "Small enough to return inline" not in rec
        assert "744 records" in rec
        assert "first 100" in rec
        assert "preview" in rec
        for alternative in ("time_resolution", "aggregate_operations", "export_query"):
            assert alternative in rec

    def test_at_record_cap_recommended_inline(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(Container.Dataset, size=1600, dlen=100)

        rec = json.loads(server.stage_query(datasource_id="test-ds"))["recommendation"]
        assert rec.startswith("Small enough to return inline")

    def test_follows_configured_record_cap(self, mock_conn, mock_stage, monkeypatch):
        monkeypatch.setenv("OCEANUM_MCP_MAX_INLINE_ROWS", "1000")
        mock_stage.return_value = make_stage(Container.Dataset, size=11_928, dlen=744)

        rec = json.loads(server.stage_query(datasource_id="test-ds"))["recommendation"]
        assert rec.startswith("Small enough to return inline")

    def test_tabular_rows_counted(self, mock_conn, mock_stage):
        mock_stage.return_value = make_stage(Container.DataFrame, size=5000, dlen=500)

        rec = json.loads(server.stage_query(datasource_id="test-ds"))["recommendation"]
        assert "500 records" in rec

    def test_reduction_carries_compute_caveat(self, mock_conn, mock_stage):
        # The observed timeout: a small global annual mean that could not be
        # computed before the client gave up.
        mock_stage.return_value = make_stage(Container.Dataset, size=16_600, dlen=1)

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
        mock_stage.return_value = make_stage(Container.Dataset, size=1000, dlen=10)

        rec = json.loads(server.stage_query(datasource_id="test-ds"))["recommendation"]
        assert "native record" not in rec
