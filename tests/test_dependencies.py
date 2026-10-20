"""Guards on the installed data stack and the post-deploy smoke script.

The hosted image installs from pyproject.toml without a lockfile, so a
dependency combination that pip resolves but that cannot read zarr only shows
up at runtime (OCE-354: xarray>=2026.9 with oceanum's zarr<3). Opening a zarr
store here fails CI on the same resolution instead.
"""

import importlib.util
import json
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
            load_datasource="grid_a", query_datasource="grid_b", query_variable="hs"
        )
    )

    assert await smoke.run("http://test/datamesh", "tok", checks) == code
    # Every check that ran prints its own outcome line.
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == (1 if connect_error else len(checks))


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
