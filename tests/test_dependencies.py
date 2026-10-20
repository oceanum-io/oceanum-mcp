"""Guards on the installed data stack and the post-deploy smoke script.

The hosted image installs from pyproject.toml without a lockfile, so a
dependency combination that pip resolves but that cannot read zarr only shows
up at runtime (OCE-354). Opening a zarr store here fails CI on the same
resolution instead.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

SMOKE_SCRIPT = Path(__file__).parent.parent / "scripts" / "smoke_hosted.py"


def test_xarray_opens_zarr_store(tmp_path):
    path = tmp_path / "grid.zarr"
    xr.Dataset({"hs": (("time",), np.arange(3.0))}).to_zarr(path)

    ds = xr.open_zarr(path)

    assert ds.sizes == {"time": 3}
    np.testing.assert_array_equal(ds["hs"].values, np.arange(3.0))


@pytest.fixture(scope="module")
def smoke():
    spec = importlib.util.spec_from_file_location("smoke_hosted", SMOKE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DATASET = json.dumps({"container": "dataset", "dims": {"time": 4}})


@pytest.mark.parametrize(
    ("is_error", "text", "passed"),
    [
        (False, DATASET, True),
        (
            True,
            (
                "Error calling tool 'load_datasource': open_group() got an "
                "unexpected keyword argument 'zarr_format'"
            ),
            False,
        ),
        (False, json.dumps({"error": "boom", "datasource_id": "x"}), False),
        (False, json.dumps({"status": "no_data", "message": "none"}), False),
        (False, json.dumps({"container": "dataframe", "rows": 1}), False),
        (False, "not json", False),
    ],
)
def test_smoke_check_result(smoke, is_error, text, passed):
    assert (smoke.check_result(is_error, text) is None) is passed
