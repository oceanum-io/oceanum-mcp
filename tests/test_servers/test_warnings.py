"""Library warnings in tool results (OCE-366).

A pandas "'d' is deprecated" warning surfaced in a query_data result: the
query's process-global warning capture recorded it from a concurrent call's
time_resolution parsing.
"""

import threading
import warnings

import pytest

from oceanum_mcp.servers.datamesh import server

RANGE = {"time_start": "2020-01-01", "time_end": "2021-01-01"}


@pytest.mark.parametrize("resolution", ["1D", "30D", "1d", "1W", "1w-mon"])
def test_concurrent_resolution_parsing_stays_out_of_a_query_result(resolution):
    collected: list[str] = []
    with server._captured_warnings(collected):
        # While a query holds the capture, another call validates its params.
        worker = threading.Thread(
            target=server._build_query,
            args=("test-ds",),
            kwargs={"time_resolution": resolution, **RANGE},
        )
        worker.start()
        worker.join(5)
    assert collected == []


def test_capture_drops_only_pandas_alias_deprecations():
    collected: list[str] = []
    with server._captured_warnings(collected):
        warnings.warn("Query is too large for direct access", UserWarning)
        warnings.warn(
            "'d' is deprecated and will be removed in a future version, "
            "please use 'D' instead.",
            DeprecationWarning,
        )
        warnings.warn(
            "'d' is deprecated and will be removed in a future version. "
            "Please use 'D' instead of 'd'.",
            FutureWarning,
        )
        warnings.warn("The gateway url is deprecated", DeprecationWarning)
        warnings.warn("some API is deprecated", FutureWarning)
    assert collected == [
        "Query is too large for direct access",
        "The gateway url is deprecated",
        "some API is deprecated",
    ]
