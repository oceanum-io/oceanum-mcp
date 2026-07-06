"""Shared test fixtures.

Tests run against the real fastmcp and oceanum packages; only the network
boundary (Connector / FileSystem / staging) is mocked.
"""

from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from oceanum.datamesh.query import Container, Query, Stage

import oceanum_mcp.servers.datamesh.server as datamesh_server


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
