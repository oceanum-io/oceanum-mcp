"""S1-2: variables + aggregate_operations fails inside the Datamesh engine.

The engine renames aggregated variables to <name>_<op> and then selects the
original names, so it reports the requested variable as missing and suggests
the renamed one, which does not work as a variable either. The engine strings
mocked here are the two wordings seen in the agent evaluation.
"""

import json
from unittest.mock import MagicMock

import pytest

from oceanum.datamesh.exceptions import DatameshConnectError

import oceanum_mcp.servers.datamesh.server as server

POINT = {
    "type": "Feature",
    "properties": {},
    "geometry": {"type": "Point", "coordinates": [174.5, -37.8]},
}

# Wordings returned by the engine (bug report IDs shortened).
RENAMED_DID_YOU_MEAN = (
    'Datamesh server error: {"detail": "Error removing irrelevant coordinates: '
    "\\\"No variable named 'u10'. Did you mean one of ('u10_std', 'u10_min', "
    "'u10_max')?\\\". bug report ID 20260920_085257_InternalQueryError_x.json\"}"
)
RENAMED_INCLUDE = (
    "No variable named 'hs'. Variables on the dataset include ['hs_mean', "
    "'tps_mean']. bug report ID 20261007_081250_InternalQueryError_x.json"
)


def _call(tool, variables, **kwargs):
    return json.loads(
        tool(
            datasource_id="test-ds",
            variables=variables,
            time_start="2020-01-01",
            time_end="2020-12-31",
            geofilter_feature=POINT,
            aggregate_operations=["mean"],
            aggregate_spatial=False,
            **kwargs,
        )
    )


@pytest.mark.parametrize(
    "message, name", [(RENAMED_DID_YOU_MEAN, "u10"), (RENAMED_INCLUDE, "hs")]
)
@pytest.mark.parametrize("tool_name", ["stage_query", "query_data"])
def test_engine_rename_error_explains_workaround(
    mock_conn, mock_stage, message, name, tool_name
):
    mock_stage.side_effect = DatameshConnectError(message)
    parsed = _call(getattr(server, tool_name), [name])
    assert "aggregate_operations" in parsed["error"]
    assert "omit variables" in parsed["error"]
    assert f"{name}_mean" in parsed["error"]
    # The engine's text is kept for Oceanum's bug report, but not as the error.
    assert parsed["engine_error"] == message
    assert "suggestions" not in parsed
    assert parsed["query"]["variables"] == [name]
    # Not a missing variable: no schema lookup and no standard_name retry.
    mock_conn.get_datasource.assert_not_called()
    assert mock_stage.call_count == 1


def test_renamed_variable_retry_still_gets_variable_suggestions(mock_conn, mock_stage):
    # An agent that follows the engine's hint and asks for 'hs_mean' gets the
    # ordinary unknown-variable handling (the variable really is unknown).
    mock_stage.side_effect = DatameshConnectError(
        'Datamesh server error: {"detail":"Invalid variable selection - '
        "variable not found: 'hs_mean'\"}"
    )
    ds = MagicMock()
    ds.variables = {"hs": {"attrs": {}}, "tps": {"attrs": {}}}
    ds.dataschema = MagicMock(coords={"time": {}})
    mock_conn.get_datasource.return_value = ds
    parsed = _call(server.stage_query, ["hs_mean"])
    assert "Variable(s) not in datasource" in parsed["error"]
    assert "engine_error" not in parsed


def test_plain_no_variable_named_is_unchanged(mock_conn, mock_stage):
    # "No variable named" without a renamed <name>_<op> candidate is not the
    # aggregation bug, so it keeps the existing missing-variable path.
    message = "No variable named 'foo'. Variables on the dataset include ['hs']"
    mock_stage.side_effect = DatameshConnectError(message)
    ds = MagicMock()
    ds.variables = {"hs": {"attrs": {}}}
    ds.dataschema = MagicMock(coords={"time": {}})
    mock_conn.get_datasource.return_value = ds
    parsed = _call(server.stage_query, ["foo"])
    assert "Variable(s) not in datasource" in parsed["error"]
    assert "engine_error" not in parsed
