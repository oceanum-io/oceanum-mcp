"""Datasource notices in the Datamesh tools: get_datasource_notices and the
notices get_datasource_info shows (mocked admin service)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import httpx2
import pytest
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import AccessToken

from oceanum_mcp.servers.datamesh import server
from oceanum_mcp.common import formatting, notices
from oceanum_mcp.common.client import CREDENTIAL_CLAIM
from tests.test_common.test_notices import JWT_CREDENTIAL, NOTICE, TOKEN, Admin
from tests.test_servers.test_catalog_detail import make_datasource
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

    def test_shares_cache_with_info(self, admin, mock_conn):
        mock_conn.get_datasource.return_value = _mock_datasource(id="era5")
        fake = admin()
        assert json.loads(server.get_datasource_info("era5"))["notices"]
        parsed = json.loads(server.get_datasource_notices("era5"))
        assert parsed["notices"][0]["title"] == NOTICE["title"]
        assert len(fake.requests) == 1

    def test_recent_failure_holds_off(self, admin):
        fake = admin(handler=lambda r: httpx2.Response(502))
        assert "HTTP 502" in json.loads(server.get_datasource_notices("era5"))["error"]
        parsed = json.loads(server.get_datasource_notices("era5"))
        assert parsed["notices"] == []  # held off: no notices, no request
        assert len(fake.requests) == 1


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

    @pytest.mark.parametrize("datasource_id", ["", "  "])
    def test_no_lookup_for_empty_id(self, admin, datasource_id: str):
        fake = admin()
        server.get_datasource_info(datasource_id)
        assert fake.requests == []

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

    def test_failed_info_has_no_notices(self, admin, mock_conn):
        from oceanum.datamesh.exceptions import DatameshConnectError

        admin()
        mock_conn.get_datasource.side_effect = DatameshConnectError("boom")
        parsed = json.loads(server.get_datasource_info("era5"))
        assert "error" in parsed
        assert not set(parsed) & set(notices.RESERVED_KEYS)

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


class TestToolErrors:
    """get_datasource_notices answers every failure with its JSON error."""

    def test_no_credential(self, admin, monkeypatch):
        fake = admin()
        monkeypatch.delenv("DATAMESH_TOKEN")
        monkeypatch.delenv("DATAMESH_TOKEN_FILE", raising=False)
        monkeypatch.delenv("DATAMESH_TOKEN_KEYRING", raising=False)
        parsed = json.loads(server.get_datasource_notices("era5"))
        assert "No Datamesh credential" in parsed["error"]
        assert fake.requests == []

    def test_unexpected_exception(self, admin):
        def boom(request: httpx2.Request) -> httpx2.Response:
            raise RuntimeError("bug")

        admin(handler=boom)
        parsed = json.loads(server.get_datasource_notices("era5"))
        assert parsed == {
            "datasource_id": "era5",
            "error": "The notices lookup failed (RuntimeError).",
        }

    def test_other_value_error_gets_generic_message(self, admin, monkeypatch):
        def broken(datasource_id: str, credential: str):
            raise ValueError("internal detail")

        monkeypatch.setattr(notices, "lookup", broken)
        parsed = json.loads(server.get_datasource_notices("era5"))
        assert parsed["error"] == "The notices lookup failed (ValueError)."

    @pytest.mark.parametrize(
        "datamesh", ["http://localhost:8000", "https://datamesh.staging.oceanum.io"]
    )
    def test_no_known_admin_service(self, admin, monkeypatch, datamesh: str):
        monkeypatch.setenv("DATAMESH_SERVICE", datamesh)
        fake = admin()
        parsed = json.loads(server.get_datasource_notices("era5"))
        assert "OCEANUM_ADMIN_SERVICE" in parsed["error"]
        assert fake.requests == []

    def test_stripped_id(self, admin):
        fake = admin()
        parsed = json.loads(server.get_datasource_notices("  era5 "))
        assert parsed["datasource_id"] == "era5"
        assert dict(fake.requests[0].url.params) == {"datasource": "era5"}

    def test_limits(self, admin):
        items = [
            {**NOTICE, "title": f"N{i}", "publish_at": f"2026-11-01T{i:02d}:00:00Z"}
            for i in range(24)
        ]
        items[23]["body"] = "b" * 5000
        admin(items=items)
        parsed = json.loads(server.get_datasource_notices("era5"))
        assert len(parsed["notices"]) == notices.MAX_NOTICES
        assert parsed["notices"][0]["title"] == "N23"
        assert len(parsed["notices"][0]["body"]) == notices.BODY_CHARS


class TestInfoMerge:
    @pytest.fixture(autouse=True)
    def _datasource(self, mock_conn):
        mock_conn.get_datasource.return_value = _mock_datasource(id="era5")

    def _spoofing_record(self, monkeypatch):
        record = {
            "id": "era5",
            "notices": [{"severity": "outage", "title": "spoofed"}],
            "notices_note": "spoofed note",
            "notices_hint": "spoofed hint",
        }
        monkeypatch.setattr(
            server, "format_datasource_bounded", lambda ds, reserve=0: dict(record)
        )

    def test_record_cannot_override_notices(self, admin, monkeypatch):
        self._spoofing_record(monkeypatch)
        admin()
        parsed = json.loads(server.get_datasource_info("era5"))
        assert parsed["notices"][0]["title"] == NOTICE["title"]
        assert parsed["notices_note"] == notices.NOTICES_NOTE
        assert parsed["record_notices"][0]["title"] == "spoofed"
        assert parsed["record_notices_hint"] == "spoofed hint"
        assert "notices_hint" not in parsed

    def test_record_cannot_pose_as_notices(self, admin, monkeypatch):
        self._spoofing_record(monkeypatch)
        admin(items=[])
        parsed = json.loads(server.get_datasource_info("era5"))
        assert not set(parsed) & set(notices.RESERVED_KEYS)
        assert parsed["record_notices_note"] == "spoofed note"

    def test_bounded_view_stays_within_budget(self, admin, mock_conn, monkeypatch):
        ds = make_datasource(n_variables=40)
        mock_conn.get_datasource.return_value = ds
        admin(items=[])
        whole = len(server.get_datasource_info(ds.id))  # fits without notices
        budget = whole + 50
        monkeypatch.setattr(formatting, "INFO_BUDGET_CHARS", budget)
        notices._cache.clear()
        admin(items=[{**NOTICE, "severity": "outage", "body": "x" * 1000}] * 5)
        raw = server.get_datasource_info(ds.id)
        parsed = json.loads(raw)
        assert len(parsed["notices"]) == notices.INFO_MAX_NOTICES
        assert len(raw) <= budget
        assert "budget_exceeded" not in parsed

    def test_runs_alongside_metadata_request(self, admin, mock_conn):
        def slow_metadata(datasource_id):
            time.sleep(0.5)
            return _mock_datasource(id=datasource_id)

        def slow_admin(request: httpx2.Request) -> httpx2.Response:
            time.sleep(0.5)
            return httpx2.Response(200, json={"items": [NOTICE]})

        mock_conn.get_datasource.side_effect = slow_metadata
        admin(handler=slow_admin)
        start = time.monotonic()
        parsed = json.loads(server.get_datasource_info("era5"))
        assert time.monotonic() - start < 0.95  # one after the other: >= 1.0
        assert parsed["notices"]

    def test_slow_admin_bounded_by_deadline(self, admin, monkeypatch):
        monkeypatch.setattr(notices, "DEADLINE_S", 0.3)

        def drip(request: httpx2.Request) -> httpx2.Response:
            def body():
                for _ in range(40):
                    time.sleep(0.05)
                    yield b" "

            return httpx2.Response(200, content=body())

        admin(handler=drip)
        start = time.monotonic()
        parsed = json.loads(server.get_datasource_info("era5"))
        assert time.monotonic() - start < 0.6
        assert parsed["id"] == "era5"
        assert "notices" not in parsed
