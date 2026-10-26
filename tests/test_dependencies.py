"""Guards on the installed data stack.

The hosted image installs from pyproject.toml without a lockfile, so a
dependency combination that pip resolves but that cannot read zarr only shows
up at runtime (OCE-354: xarray>=2026.9 with oceanum's zarr<3). Opening a zarr
store here fails CI on the same resolution instead.
"""

import numpy as np
import xarray as xr


def test_xarray_opens_zarr_store(tmp_path):
    path = tmp_path / "grid.zarr"
    xr.Dataset({"hs": (("time",), np.arange(3.0))}).to_zarr(path)

    ds = xr.open_zarr(path)

    assert ds.sizes == {"time": 3}
    np.testing.assert_array_equal(ds["hs"].values, np.arange(3.0))
