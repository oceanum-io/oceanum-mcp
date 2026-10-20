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


@pytest.mark.parametrize("resolution", ["1D", "30D", "7D"])
def test_day_resolutions_parse_without_warnings(resolution):
    """The engine-side simulation parses the canonical "D", not "d", so
    nothing is emitted even with the quieting skipped (lock held)."""
    with server._WARNINGS_LOCK, warnings.catch_warnings():
        warnings.simplefilter("error")
        server._check_time_resolution(resolution)


def test_capture_keeps_library_signals_but_drops_deprecations():
    collected: list[str] = []
    with server._captured_warnings(collected):
        warnings.warn("Query is too large for direct access", UserWarning)
        warnings.warn("'d' is deprecated", DeprecationWarning)
        warnings.warn("'d' is deprecated", FutureWarning)
    assert collected == ["Query is too large for direct access"]
