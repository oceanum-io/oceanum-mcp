"""Datasource notices in the Datamesh tools: get_datasource_notices and the
notices get_datasource_info shows (mocked admin service)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import httpx2
import pytest
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import AccessToken

from oceanum_mcp.servers.datamesh import server
from oceanum_mcp.common import notices
from oceanum_mcp.common.client import CREDENTIAL_CLAIM
from tests.test_common.test_notices import JWT_CREDENTIAL, NOTICE, TOKEN, Admin
from tests.test_servers.test_datamesh import _mock_datasource


@pytest.fixture
def admin(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Admin]:
    def install(**kwargs: Any) -> Admin:
        fake = Admin(**kwargs)
        client = notices.new_client(httpx2.MockTransport(fake))
        monkeypatch.setattr(notices, "_http_client", lambda: client)
        return fake

    return install


@pytest.fixture(autouse=True)
def _stdio_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stdio server holding its own Datamesh token; notices enabled."""
    monkeypatch.setenv("DATAMESH_TOKEN", TOKEN)
    monkeypatch.delenv("OCEANUM_MCP_DISABLE_NOTICES", raising=False)


def _hosted_jwt():
    """An authenticated hosted request whose credential is an Auth0 JWT."""
    access = AccessToken(
        token="jwt",
        client_id="user",
        scopes=[],
        claims={CREDENTIAL_CLAIM: JWT_CREDENTIAL},
    )
    return patch("fastmcp.server.dependencies.get_access_token", return_value=access)


class TestGetDatasourceNotices:
    def test_returns_notices(self, admin):
        admin()
        parsed = json.loads(server.get_datasource_notices("era5"))
        assert parsed == {
            "datasource_id": "era5",
            "notices_note": notices.NOTICES_NOTE,
            "notices": [{k: NOTICE[k] for k in notices.NOTICE_FIELDS}],
        }

    def test_none(self, admin):
        admin(handler=lambda r: httpx2.Response(404, json={"detail": "No notices"}))
        parsed = json.loads(server.get_datasource_notices("era5"))
        assert parsed["notices"] == []

    def test_stdio_token_header(self, admin):
        fake = admin()
        server.get_datasource_notices("era5")
        assert fake.requests[0].headers["X-DATAMESH-TOKEN"] == TOKEN
        assert "Authorization" not in fake.requests[0].headers

    def test_hosted_jwt_header(self, admin):
        fake = admin()
        with _hosted_jwt():
            server.get_datasource_notices("era5")
        headers = fake.requests[0].headers
        assert headers["Authorization"] == JWT_CREDENTIAL
        assert "X-DATAMESH-TOKEN" not in headers

    def test_error_reported(self, admin):
        admin(handler=lambda r: httpx2.Response(503, headers={"Retry-After": "7"}))
        parsed = json.loads(server.get_datasource_notices("era5"))
        assert "busy" in parsed["error"]
        assert parsed["retry_after_s"] == 7.0
        assert parsed["datasource_id"] == "era5"
        assert TOKEN not in json.dumps(parsed)

    def test_empty_id_refused(self, admin):
        fake = admin()
        with pytest.raises(ToolError):
            server.get_datasource_notices(" ")
        assert fake.requests == []

    def test_fresh_answer_refreshes_info_cache(self, admin, mock_conn):
        mock_conn.get_datasource.return_value = _mock_datasource(id="era5")
        fake = admin(items=[])
        assert "notices" not in json.loads(server.get_datasource_info("era5"))
        fake._items = [NOTICE]
        server.get_datasource_notices("era5")
        parsed = json.loads(server.get_datasource_info("era5"))
        assert parsed["notices"][0]["title"] == NOTICE["title"]
        assert len(fake.requests) == 2


class TestDatasourceInfoNotices:
    @pytest.fixture(autouse=True)
    def _datasource(self, mock_conn):
        mock_conn.get_datasource.return_value = _mock_datasource(id="era5")

    @pytest.mark.parametrize("detail", ["summary", "full"])
    def test_notices_come_first(self, admin, detail: str):
        admin()
        raw = server.get_datasource_info("era5", detail=detail)
        parsed = json.loads(raw)
        assert list(parsed)[:2] == ["notices_note", "notices"]
        assert parsed["notices"][0]["severity"] == "action_required"
        assert parsed["id"] == "era5"

    def test_no_notices_leaves_record_unchanged(self, admin):
        admin(items=[])
        parsed = json.loads(server.get_datasource_info("era5"))
        assert "notices" not in parsed and "notices_hint" not in parsed
        assert parsed["id"] == "era5"

    def test_cached_across_calls(self, admin):
        fake = admin()
        for _ in range(3):
            server.get_datasource_info("era5")
        assert len(fake.requests) == 1

    @pytest.mark.parametrize(
        "handler",
        [
            lambda r: httpx2.Response(500),
            lambda r: httpx2.Response(401),
            lambda r: (_ for _ in ()).throw(httpx2.ConnectTimeout("t")),
            lambda r: (_ for _ in ()).throw(httpx2.ConnectError("refused")),
        ],
        ids=["500", "401", "timeout", "unreachable"],
    )
    def test_admin_failure_does_not_break_info(self, admin, handler):
        admin(handler=handler)
        parsed = json.loads(server.get_datasource_info("era5"))
        assert parsed["id"] == "era5"
        assert "notices" not in parsed

    def test_hosted_jwt_header(self, admin):
        fake = admin()
        with _hosted_jwt():
            server.get_datasource_info("era5")
        assert fake.requests[0].headers["Authorization"] == JWT_CREDENTIAL

    def test_not_looked_up_for_failed_info(self, admin, mock_conn):
        from oceanum.datamesh.exceptions import DatameshConnectError

        fake = admin()
        mock_conn.get_datasource.side_effect = DatameshConnectError("boom")
        parsed = json.loads(server.get_datasource_info("era5"))
        assert "error" in parsed
        assert fake.requests == []

    def test_opt_out(self, admin, monkeypatch):
        fake = admin()
        monkeypatch.setenv("OCEANUM_MCP_DISABLE_NOTICES", "1")
        parsed = json.loads(server.get_datasource_info("era5"))
        assert "notices" not in parsed
        assert fake.requests == []


def test_redirect_not_followed_by_tool(admin):
    fake = admin(
        handler=lambda r: httpx2.Response(
            307, headers={"Location": "https://evil.example/"}
        )
    )
    with _hosted_jwt():
        parsed = json.loads(server.get_datasource_notices("era5"))
    assert "error" in parsed
    assert [r.url.host for r in fake.requests] == ["admin.oceanum.io"]
