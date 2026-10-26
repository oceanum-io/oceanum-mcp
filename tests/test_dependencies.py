"""Guards on the installed data stack and the post-deploy smoke script.

The hosted image installs from pyproject.toml without a lockfile, so a
dependency combination that pip resolves but that cannot read zarr only shows
up at runtime (OCE-354: xarray>=2026.9 with oceanum's zarr<3). Opening a zarr
store here fails CI on the same resolution instead.
"""

import importlib.util
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from fastmcp.exceptions import MCPError

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


LAZY = json.dumps({"container": "dataset", "dims": {"time": 8761}, "lazy": True})
INLINE = json.dumps({"container": "dataset", "dims": {"time": 4}, "lazy": False})
ZARR_ERROR = (
    "Error calling tool 'load_datasource': open_group() got an "
    "unexpected keyword argument 'zarr_format'"
)
# zarr 2's GroupNotFoundError on a broken image: says "not found", but is not
# the server's missing-datasource wording.
ZARR_NOT_FOUND = json.dumps({"error": "group not found at path ''"})
REFUSED = json.dumps({"refused": True, "message": "Result is 3 GB, above..."})


def not_found(datasource_id):
    return json.dumps(
        {
            "error": f"Datasource {datasource_id!r} not found or not accessible "
            "with these credentials; did you mean one of the suggestions?",
            "suggestions": [],
        }
    )


def not_shared(datasource_id):
    return json.dumps(
        {
            "error": f"You don't have access to datasource {datasource_id!r}: it "
            "exists but is not shared with your account. Ask its owner or "
            "your Oceanum administrator for access."
        }
    )


@pytest.mark.parametrize(
    ("is_error", "text", "outcome"),
    [
        (False, LAZY, "PASS"),
        (True, ZARR_ERROR, "FAIL"),
        (True, "Error calling tool 'load_datasource': group not found", "FAIL"),
        (False, ZARR_NOT_FOUND, "FAIL"),
        (False, json.dumps({"error": "boom", "datasource_id": "x"}), "FAIL"),
        (False, json.dumps({"container": "dataframe", "rows": 1}), "FAIL"),
        (False, "not json", "FAIL"),
        # Another datasource's not-found is not this check's input problem.
        (False, not_found("other"), "FAIL"),
        (False, INLINE, "INCONCLUSIVE"),
        (False, not_found("x"), "INCONCLUSIVE"),
        (False, not_shared("x"), "INCONCLUSIVE"),
        (False, json.dumps({"status": "no_data", "message": "none"}), "INCONCLUSIVE"),
        (False, REFUSED, "INCONCLUSIVE"),
    ],
)
def test_smoke_check_result(smoke, is_error, text, outcome):
    assert smoke.check_result("query_data", "x", is_error, text)[0] == outcome


def test_smoke_not_lazy_wording_is_per_tool(smoke):
    _, load = smoke.check_result("load_datasource", "x", False, INLINE)
    _, query = smoke.check_result("query_data", "x", False, INLINE)

    assert "not opened lazily" in load
    assert "did not reach the zarr path" in query
    assert "range may be too small" in query


class _StubClient:
    """Stands in for fastmcp.Client: replies to call_tool from a list (an
    exception reply is raised), or raises on connect."""

    def __init__(self, replies=None, connect_error=None):
        self.replies = list(replies or [])
        self.connect_error = connect_error

    def __call__(self, transport, timeout):
        return self

    async def __aenter__(self):
        if self.connect_error is not None:
            raise self.connect_error
        return self

    async def __aexit__(self, *exc):
        return False

    async def call_tool(self, tool, args, raise_on_error):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        is_error, text = reply
        return SimpleNamespace(is_error=is_error, content=[SimpleNamespace(text=text)])


@pytest.mark.parametrize(
    ("replies", "connect_error", "code"),
    [
        ([(False, LAZY), (False, LAZY)], None, 0),
        ([(True, ZARR_ERROR), (False, LAZY)], None, 1),
        ([(False, ZARR_NOT_FOUND), (False, LAZY)], None, 1),
        ([(False, not_found("grid_a")), (True, ZARR_ERROR)], None, 1),
        ([(False, LAZY), (False, INLINE)], None, 3),
        ([(False, not_found("grid_a")), (False, LAZY)], None, 3),
        ([(False, LAZY), (False, REFUSED)], None, 3),
        ([TimeoutError("timed out"), (False, LAZY)], None, 1),
        (None, RuntimeError("Client failed to connect"), 1),
        (None, MCPError(-32603, "Server returned an error response"), 1),
    ],
)
async def test_smoke_run_exit_codes(
    smoke, monkeypatch, capsys, replies, connect_error, code
):
    monkeypatch.setattr(smoke, "Client", _StubClient(replies, connect_error))
    checks = smoke.build_checks(
        SimpleNamespace(
            load_datasource="grid_a",
            query_datasource="grid_b",
            query_variable="hs",
            query_start=date(2020, 1, 1),
            query_end=date(2020, 3, 31),
        )
    )

    assert await smoke.run("http://test/datamesh", "tok", checks) == code
    # Every check that ran prints its own outcome line.
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == (1 if connect_error else len(checks))


# Calibration from a live hosted stage_query on 2026-10-26: hs on
# oceanum_wave_auckl_era5_grid staged at 1_743_896_420 bytes for all of 2020
# (366 days) and 430_153_220 for 2020-01-01..2020-03-31. This only pins the
# default against the server's thresholds; the live smoke run is the real check.
AUCKL_HS_BYTES_PER_DAY = 1_743_896_420 / 366


def test_smoke_default_query_fits_lazy_window(smoke):
    # Imported here so a broken server import does not mask the data-stack
    # guards above.
    from oceanum_mcp.common.config import DEFAULT_MAX_INLINE_BYTES
    from oceanum_mcp.servers.datamesh.server import LARGE_EXPORT_BYTES

    start = date.fromisoformat(smoke.DEFAULT_QUERY_START)
    end = date.fromisoformat(smoke.DEFAULT_QUERY_END)
    # Counts time_end's whole day, though the server stages up to its midnight,
    # so this overestimates by a day (~1%); the window has ~2x headroom.
    size = ((end - start).days + 1) * AUCKL_HS_BYTES_PER_DAY

    # Below the inline limit query_data downloads the result whole (passes on a
    # broken image); above LARGE_EXPORT_BYTES it refuses it (OCE-355), as it
    # did the old full-year default.
    assert DEFAULT_MAX_INLINE_BYTES < size <= LARGE_EXPORT_BYTES


def _sent_checks(smoke, monkeypatch, argv):
    """The checks main(argv) hands to run(), as {tool: arguments}."""
    monkeypatch.setenv("DATAMESH_TOKEN", "tok")
    sent = []

    async def fake_run(url, token, checks):
        sent.extend(checks)
        return 0

    monkeypatch.setattr(smoke, "run", fake_run)
    assert smoke.main(argv) == 0
    return dict(sent)


def test_smoke_default_query_range(smoke, monkeypatch):
    query = _sent_checks(smoke, monkeypatch, [])["query_data"]

    assert query["datasource_id"] == "oceanum_wave_auckl_era5_grid"
    assert query["variables"] == ["hs"]
    assert (query["time_start"], query["time_end"]) == (
        smoke.DEFAULT_QUERY_START,
        smoke.DEFAULT_QUERY_END,
    )


def test_smoke_query_range_overrides(smoke, monkeypatch):
    argv = ["--query-start", "2021-06-01", "--query-end", "2021-06-30"]

    query = _sent_checks(smoke, monkeypatch, argv)["query_data"]

    assert (query["time_start"], query["time_end"]) == ("2021-06-01", "2021-06-30")


@pytest.mark.parametrize(
    "argv",
    [
        ["--query-start", "2020-13-01"],
        ["--query-end", "not-a-date"],
    ],
)
def test_smoke_query_range_rejects_bad_dates(smoke, monkeypatch, argv):
    monkeypatch.setenv("DATAMESH_TOKEN", "tok")

    with pytest.raises(SystemExit) as exc:
        smoke.main(argv)

    assert exc.value.code == 2


def test_smoke_query_range_rejects_reversed(smoke, monkeypatch, capsys):
    monkeypatch.setenv("DATAMESH_TOKEN", "tok")
    monkeypatch.setattr(smoke, "run", None)  # never reached
    argv = ["--query-start", "2020-03-31", "--query-end", "2020-01-01"]

    assert smoke.main(argv) == 2
    assert "is after --query-end" in capsys.readouterr().err


def test_smoke_main_requires_token(smoke, monkeypatch):
    monkeypatch.delenv("DATAMESH_TOKEN", raising=False)

    assert smoke.main([]) == 2


@pytest.mark.parametrize(
    ("url", "code"),
    [
        ("http://mcp.example.com/datamesh", 2),
        ("http://127.0.0.1:8765/datamesh", 0),
        ("http://localhost:8765/datamesh", 0),
        ("https://mcp.example.com/datamesh", 0),
    ],
)
def test_smoke_main_requires_https_for_remote(smoke, monkeypatch, url, code):
    monkeypatch.setenv("DATAMESH_TOKEN", "tok")
    sent = []

    async def fake_run(url, token, checks):
        sent.append(url)
        return 0

    monkeypatch.setattr(smoke, "run", fake_run)

    assert smoke.main([url]) == code
    assert sent == ([] if code == 2 else [url])
