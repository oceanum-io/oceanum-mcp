"""Tool parameter contract tests (OCE-305).

Pins each tool's JSON-schema `required` list to the params that are truly
required by its signature, so a change can't silently make an optional param
required for clients. Also covers normalising the literal strings "null" and
"" (sent by some MCP clients for omitted optional params) to None.
"""

from __future__ import annotations

import importlib
import inspect
import json

import pytest
from fastmcp import Client

from oceanum_mcp.servers.datamesh import server

# Hand-enumerated from the tool signatures: params without a default. A new
# tool, or a param gaining/losing a default, must update this table.
EXPECTED_REQUIRED: dict[str, dict[str, set[str]]] = {
    "oceanum_mcp.servers.datamesh.server": {
        "search_catalog": set(),
        "get_datasource_info": {"datasource_id"},
        "stage_query": {"datasource_id"},
        "query_data": {"datasource_id"},
        "export_query": {"datasource_id"},
        "load_datasource": {"datasource_id"},
        "update_metadata": {"datasource_id"},
    },
    "oceanum_mcp.servers.storage.server": {
        "list_files": set(),
        "file_exists": {"path"},
        "read_file": {"path"},
        "write_file": {"path", "content"},
        "delete_file": {"path"},
        "file_info": {"path"},
    },
}

_COMBINED_EXPECTED = {
    f"{namespace}_{name}": required
    for namespace, module in (
        ("datamesh", "oceanum_mcp.servers.datamesh.server"),
        ("storage", "oceanum_mcp.servers.storage.server"),
    )
    for name, required in EXPECTED_REQUIRED[module].items()
}


async def _schemas(mcp) -> dict[str, dict]:
    """Input schemas as an MCP client sees them."""
    async with Client(mcp) as client:
        return {t.name: t.inputSchema for t in await client.list_tools()}


@pytest.fixture(autouse=True)
def _all_tools_enabled(monkeypatch):
    # Read-only mode hides write tools; the contract covers every tool.
    monkeypatch.delenv("OCEANUM_MCP_READ_ONLY", raising=False)


class TestRequiredParams:
    @pytest.mark.parametrize("module_name", sorted(EXPECTED_REQUIRED))
    async def test_required_lists_match_signatures(self, module_name):
        module = importlib.import_module(module_name)
        schemas = await _schemas(module.mcp)
        expected = EXPECTED_REQUIRED[module_name]

        assert set(schemas) == set(expected), "tool set changed; update the table"
        for name, schema in schemas.items():
            assert set(schema.get("required", [])) == expected[name], name

    async def test_combined_server_required_lists(self):
        combined = importlib.import_module("oceanum_mcp.servers.combined.server")
        schemas = await _schemas(combined.mcp)

        assert set(schemas) == set(_COMBINED_EXPECTED)
        for name, schema in schemas.items():
            assert set(schema.get("required", [])) == _COMBINED_EXPECTED[name], name

    @pytest.mark.parametrize("module_name", sorted(EXPECTED_REQUIRED))
    async def test_optional_params_default_and_accept_null(self, module_name):
        """Every optional param has a schema default; `X | None` ones allow null."""
        module = importlib.import_module(module_name)
        schemas = await _schemas(module.mcp)

        for name, schema in schemas.items():
            tool = getattr(module, name)
            sig = inspect.signature(getattr(tool, "fn", tool))
            for pname, param in sig.parameters.items():
                if param.default is inspect.Parameter.empty:
                    continue
                prop = schema["properties"][pname]
                assert "default" in prop, f"{name}.{pname} has no schema default"
                if param.default is None:
                    types = [s.get("type") for s in prop.get("anyOf", [prop])]
                    assert "null" in types, f"{name}.{pname} does not accept null"


class TestNullStringNormalisation:
    @pytest.mark.parametrize("sentinel", ["null", ""])
    def test_optional_string_params_treated_as_none(self, sentinel):
        q = server._build_query(
            "test-ds",
            time_start=sentinel,
            time_end=sentinel,
            time_resolution=sentinel,
            crs=sentinel,
        )
        assert q.timefilter is None
        assert q.crs is None

    @pytest.mark.parametrize("sentinel", ["null", ""])
    def test_open_ended_range_with_sentinel_end(self, sentinel):
        q = server._build_query("test-ds", time_start="2024-01-01", time_end=sentinel)
        assert q.timefilter.times[1] is None

    def test_sentinel_does_not_trip_resolution_validation(self):
        # time_resolution="null" must not count as a set resolution.
        q = server._build_query("test-ds", times=["2024-01-01"], time_resolution="null")
        assert q.timefilter.type.value == "series"

    def test_required_datasource_id_untouched(self):
        q = server._build_query("null")
        assert q.datasource == "null"

    def test_real_values_pass_through(self):
        q = server._build_query(
            "test-ds",
            time_start="2024-01-01",
            time_end="2024-02-01",
            time_resolution="1D",
            crs="EPSG:4326",
        )
        assert q.timefilter.resolution == "1D"
        assert q.crs == "EPSG:4326"

    async def test_null_string_via_mcp_call(self, mock_conn, mock_stage):
        async with Client(server.mcp) as client:
            result = await client.call_tool(
                "stage_query",
                {"datasource_id": "test-ds", "time_start": "null", "crs": "null"},
            )
        echoed = json.loads(result.content[0].text)["query"]
        assert "timefilter" not in echoed
        assert "crs" not in echoed
