"""Shared test fixtures.

Tests run against the real fastmcp and oceanum packages; only the network
boundary (Connector / FileSystem / staging) is mocked.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Iterator
from unittest.mock import MagicMock, patch

import httpx2
import pytest

from oceanum.datamesh.query import Container, Query, Stage

import oceanum_mcp.servers.datamesh.server as datamesh_server
from oceanum_mcp.common import notices


def make_stage(
    container: Container = Container.Dataset, size: int = 1000, dlen: int = 10
) -> Stage:
    """Build a real Stage object as returned by the Datamesh staging endpoint."""
    return Stage(
        query=Query(datasource="test-ds"),
        qhash="qhash",
        formats=["nc"],
        size=size,
        dlen=dlen,
        coordmap={},
        coordkeys={},
        container=container,
        sig="sig",
    )


@pytest.fixture
def mock_conn() -> Iterator[MagicMock]:
    """Mock Connector patched into the datamesh server module."""
    conn = MagicMock()
    with patch.object(datamesh_server, "get_datamesh_connector", return_value=conn):
        yield conn


@pytest.fixture
def mock_stage() -> Iterator[MagicMock]:
    """Patch the datamesh server's staging helper; defaults to a small dataset."""
    with patch.object(datamesh_server, "_stage") as stager:
        stager.return_value = make_stage()
        yield stager


def _no_admin_service(request: httpx2.Request) -> httpx2.Response:
    raise AssertionError(f"unexpected notices request to {request.url}")


@pytest.fixture(autouse=True)
def _isolate_notices(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep every test off the real admin service.

    get_datasource_info looks up notices with whatever credential resolves
    (CI sets DATAMESH_TOKEN), so the lookup is disabled by default, which
    subprocess servers inherit through the environment, and the in-process
    HTTP client refuses any request. Notice tests re-enable the lookup and
    patch the client with their own transport. Service URL overrides from
    the developer's environment are dropped so the admin host is the default.
    """
    monkeypatch.setenv("OCEANUM_MCP_DISABLE_NOTICES", "1")
    for name in ("OCEANUM_ADMIN_SERVICE", "DATAMESH_SERVICE", "OCEANUM_DOMAIN"):
        monkeypatch.delenv(name, raising=False)
    client = notices.new_client(httpx2.MockTransport(_no_admin_service))
    monkeypatch.setattr(notices, "_http_client", lambda: client)
    executor = ThreadPoolExecutor(max_workers=8)
    monkeypatch.setattr(notices, "_executor", executor)
    notices._cache.clear()
    notices._warned.clear()
    notices._inflight.clear()
    yield
    # A lookup a test stopped waiting for must not fill the next test's cache.
    executor.shutdown(wait=True)
    notices._cache.clear()
    notices._warned.clear()
    client.close()
