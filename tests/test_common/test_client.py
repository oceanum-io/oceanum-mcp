"""Tests for per-request credential resolution and client caching."""

import os
from unittest.mock import MagicMock, patch

import pytest

from fastmcp.server.auth import AccessToken

from oceanum_mcp.common.client import (
    CREDENTIAL_CLAIM,
    KEYRING_SERVICE,
    KEYRING_USERNAME,
    _ClientCache,
    _keyring_lookup,
    _datamesh_cache,
    _storage_cache,
    get_datamesh_connector,
    is_unresolved_variable,
    resolve_credential,
)


def make_access_token(credential: str) -> AccessToken:
    return AccessToken(
        token=credential,
        client_id="user",
        scopes=[],
        claims={CREDENTIAL_CLAIM: credential},
    )


@pytest.fixture(autouse=True)
def clean_caches():
    _datamesh_cache.clear()
    _storage_cache.clear()
    yield
    _datamesh_cache.clear()
    _storage_cache.clear()


def test_resolve_credential_prefers_auth_context():
    """An authenticated request's credential wins over the environment."""
    with patch(
        "fastmcp.server.dependencies.get_access_token",
        return_value=make_access_token("request-token"),
    ):
        with patch.dict(os.environ, {"DATAMESH_TOKEN": "env-token"}, clear=False):
            assert resolve_credential() == "request-token"


def test_resolve_credential_env_fallback():
    """Outside an auth context (stdio) the environment token is used."""
    with patch("fastmcp.server.dependencies.get_access_token", return_value=None):
        with patch.dict(os.environ, {"DATAMESH_TOKEN": "env-token"}, clear=False):
            assert resolve_credential() == "env-token"


def test_resolve_credential_missing_raises():
    with patch("fastmcp.server.dependencies.get_access_token", return_value=None):
        env = {k: v for k, v in os.environ.items() if k != "DATAMESH_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="credential"):
                resolve_credential()


def test_resolve_credential_rejects_missing_claim():
    """An authenticated request whose provider set no credential claim fails
    loudly instead of degrading to the raw bearer."""
    access = AccessToken(token="raw-jwt", client_id="c", scopes=[], claims={})
    with patch("fastmcp.server.dependencies.get_access_token", return_value=access):
        with pytest.raises(ValueError, match=CREDENTIAL_CLAIM):
            resolve_credential()


def test_resolve_credential_refuses_env_fallback_on_network_request():
    """An unauthenticated network request must not run as the server's own
    identity, even with DATAMESH_TOKEN set (bypassed-auth launch paths)."""
    with patch("fastmcp.server.dependencies.get_access_token", return_value=None):
        with patch(
            "fastmcp.server.dependencies.get_http_request", return_value=object()
        ):
            with patch.dict(os.environ, {"DATAMESH_TOKEN": "env-token"}, clear=False):
                with pytest.raises(ValueError, match="[Uu]nauthenticated"):
                    resolve_credential()


def test_resolve_credential_env_fallback_allowed_with_auth_none():
    """OCEANUM_MCP_AUTH=none is the explicit opt-in for env-token serving."""
    env = {"DATAMESH_TOKEN": "env-token", "OCEANUM_MCP_AUTH": "none"}
    with patch("fastmcp.server.dependencies.get_access_token", return_value=None):
        with patch(
            "fastmcp.server.dependencies.get_http_request", return_value=object()
        ):
            with patch.dict(os.environ, env, clear=False):
                assert resolve_credential() == "env-token"


def test_multi_tenant_connector_isolation():
    """Different request credentials must never share a Connector."""
    connector_cls = MagicMock(side_effect=lambda **kw: MagicMock(name=kw["token"]))
    with patch("oceanum.datamesh.Connector", connector_cls):
        with patch(
            "fastmcp.server.dependencies.get_access_token",
            return_value=make_access_token("token-a"),
        ):
            conn_a1 = get_datamesh_connector()
            conn_a2 = get_datamesh_connector()
        with patch(
            "fastmcp.server.dependencies.get_access_token",
            return_value=make_access_token("token-b"),
        ):
            conn_b = get_datamesh_connector()

    assert conn_a1 is conn_a2, "same credential must reuse the cached client"
    assert conn_a1 is not conn_b, "different credentials must get different clients"
    tokens = [c.kwargs["token"] for c in connector_cls.call_args_list]
    assert tokens == ["token-a", "token-b"]


def test_client_cache_ttl_expiry():
    cache = _ClientCache(max_entries=8, ttl_s=0.0)
    first = cache.get_or_create(("t", "s"), lambda: object())
    second = cache.get_or_create(("t", "s"), lambda: object())
    assert first is not second, "expired entries must be rebuilt"


def test_client_cache_bounded():
    cache = _ClientCache(max_entries=2, ttl_s=60.0)
    a = cache.get_or_create(("a", "s"), lambda: "A")
    cache.get_or_create(("b", "s"), lambda: "B")
    cache.get_or_create(("c", "s"), lambda: "C")
    rebuilt = cache.get_or_create(("a", "s"), lambda: "A2")
    assert a == "A" and rebuilt == "A2", "oldest entry must be evicted at the bound"


# --- server token sources (OCE-362) ------------------------------------------


@pytest.fixture
def stdio_env(monkeypatch):
    """A stdio call (no auth context) with every server token source unset."""
    for key in ("DATAMESH_TOKEN", "DATAMESH_TOKEN_FILE", "DATAMESH_TOKEN_KEYRING"):
        monkeypatch.delenv(key, raising=False)
    _keyring_lookup.cache_clear()
    with patch("fastmcp.server.dependencies.get_access_token", return_value=None):
        yield monkeypatch
    _keyring_lookup.cache_clear()


@pytest.mark.parametrize(
    "value", ["${input:datamesh-token}", "${env:DATAMESH_TOKEN}", "$DATAMESH_TOKEN"]
)
def test_unresolved_client_variable_is_named(stdio_env, value):
    """VS Code passed the literal "${input:datamesh-token}" (23 chars) as the
    token; the error must say so rather than "Invalid token"."""
    stdio_env.setenv("DATAMESH_TOKEN", value)
    with pytest.raises(ValueError, match="unresolved client variable") as exc:
        resolve_credential()
    assert "did not substitute it" in str(exc.value)


@pytest.mark.parametrize("value", ["a" * 40, "eyJh.eyJz.c2ln", "tok-a"])
def test_real_tokens_are_not_flagged(value):
    assert not is_unresolved_variable(value)


def test_token_read_from_file(stdio_env, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("file-token\n")
    token_file.chmod(0o600)
    stdio_env.setenv("DATAMESH_TOKEN_FILE", str(token_file))
    assert resolve_credential() == "file-token"


def test_token_file_rotation_needs_no_restart(stdio_env, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("old")
    token_file.chmod(0o600)
    stdio_env.setenv("DATAMESH_TOKEN_FILE", str(token_file))
    assert resolve_credential() == "old"
    token_file.write_text("new")
    assert resolve_credential() == "new"


def test_token_file_expands_home(stdio_env, tmp_path):
    (tmp_path / "token").write_text("home-token")
    (tmp_path / "token").chmod(0o600)
    stdio_env.setenv("HOME", str(tmp_path))
    stdio_env.setenv("DATAMESH_TOKEN_FILE", "~/token")
    assert resolve_credential() == "home-token"


def test_datamesh_token_wins_over_file(stdio_env, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("file-token")
    token_file.chmod(0o600)
    stdio_env.setenv("DATAMESH_TOKEN", "env-token")
    stdio_env.setenv("DATAMESH_TOKEN_FILE", str(token_file))
    assert resolve_credential() == "env-token"


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
@pytest.mark.parametrize("mode", [0o640, 0o604, 0o620, 0o644])
def test_token_file_accessible_to_others_is_refused(stdio_env, tmp_path, mode):
    token_file = tmp_path / "token"
    token_file.write_text("secret-value")
    token_file.chmod(mode)
    stdio_env.setenv("DATAMESH_TOKEN_FILE", str(token_file))
    with pytest.raises(ValueError, match="chmod 600") as exc:
        resolve_credential()
    assert "secret-value" not in str(exc.value)


def test_token_file_missing(stdio_env, tmp_path):
    stdio_env.setenv("DATAMESH_TOKEN_FILE", str(tmp_path / "absent"))
    with pytest.raises(ValueError, match="could not be read"):
        resolve_credential()


def test_token_file_empty(stdio_env, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("\n")
    token_file.chmod(0o600)
    stdio_env.setenv("DATAMESH_TOKEN_FILE", str(token_file))
    with pytest.raises(ValueError, match="is empty"):
        resolve_credential()


@pytest.mark.parametrize("raw", ["$HOME/.config/token", "${input:token-file}"])
def test_token_file_path_not_expanded(stdio_env, raw):
    stdio_env.setenv("DATAMESH_TOKEN_FILE", raw)
    with pytest.raises(ValueError, match="was not expanded") as exc:
        resolve_credential()
    assert "absolute path" in str(exc.value)


def test_token_from_keyring(stdio_env):
    import keyring

    stdio_env.setenv("DATAMESH_TOKEN_KEYRING", "1")
    with patch.object(keyring, "get_password", return_value="kr-token") as get:
        assert resolve_credential() == "kr-token"
    get.assert_called_once_with(KEYRING_SERVICE, KEYRING_USERNAME)


def test_keyring_looked_up_once_per_process(stdio_env):
    import keyring

    stdio_env.setenv("DATAMESH_TOKEN_KEYRING", "1")
    with patch.object(keyring, "get_password", return_value="kr-token") as get:
        assert resolve_credential() == "kr-token"
        assert resolve_credential() == "kr-token"
    get.assert_called_once()


def test_keyring_failure_not_cached(stdio_env):
    """Storing a missing token takes effect without a restart."""
    import keyring

    stdio_env.setenv("DATAMESH_TOKEN_KEYRING", "1")
    with patch.object(keyring, "get_password", side_effect=[None, "kr-token"]):
        with pytest.raises(ValueError, match="no token is stored"):
            resolve_credential()
        assert resolve_credential() == "kr-token"


@pytest.mark.parametrize("value", ["disabled", "enabled", "2", "a" * 40])
def test_keyring_unrecognised_value_is_refused_without_echo(stdio_env, value):
    stdio_env.setenv("DATAMESH_TOKEN_KEYRING", value)
    with pytest.raises(ValueError, match="DATAMESH_TOKEN_KEYRING must be") as exc:
        resolve_credential()
    assert value not in str(exc.value)


@pytest.mark.parametrize("value", ["on", " On "])
def test_keyring_on_enables_lookup(stdio_env, value):
    import keyring

    stdio_env.setenv("DATAMESH_TOKEN_KEYRING", value)
    with patch.object(keyring, "get_password", return_value="kr-token"):
        assert resolve_credential() == "kr-token"


@pytest.mark.parametrize("value", ["0", "false", "no", "", "off", "Off"])
def test_keyring_disabled_values(stdio_env, value):
    stdio_env.setenv("DATAMESH_TOKEN_KEYRING", value)
    with pytest.raises(ValueError, match="No Datamesh credential"):
        resolve_credential()


def test_keyring_without_stored_token(stdio_env):
    import keyring

    stdio_env.setenv("DATAMESH_TOKEN_KEYRING", "true")
    with (
        patch.object(keyring, "get_password", return_value=None),
        pytest.raises(ValueError, match="keyring set oceanum-mcp"),
    ):
        resolve_credential()


def test_keyring_backend_failure(stdio_env):
    import keyring
    from keyring.errors import NoKeyringError

    stdio_env.setenv("DATAMESH_TOKEN_KEYRING", "1")
    with (
        patch.object(keyring, "get_password", side_effect=NoKeyringError("none")),
        pytest.raises(ValueError, match="credential store could not be read"),
    ):
        resolve_credential()


def test_keyring_not_installed(stdio_env):
    stdio_env.setenv("DATAMESH_TOKEN_KEYRING", "1")
    with (
        patch.dict("sys.modules", {"keyring": None}),
        pytest.raises(ValueError, match=r"oceanum-mcp\[keyring\]"),
    ):
        resolve_credential()


def test_keyring_not_consulted_unless_enabled(stdio_env):
    import keyring

    with (
        patch.object(keyring, "get_password") as get,
        pytest.raises(ValueError, match="No Datamesh credential"),
    ):
        resolve_credential()
    get.assert_not_called()


def test_token_file_with_byte_order_mark(stdio_env, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_bytes("﻿file-token\r\n".encode())
    token_file.chmod(0o600)
    stdio_env.setenv("DATAMESH_TOKEN_FILE", str(token_file))
    assert resolve_credential() == "file-token"


def test_token_file_holding_a_variable(stdio_env, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("${input:datamesh-token}")
    token_file.chmod(0o600)
    stdio_env.setenv("DATAMESH_TOKEN_FILE", str(token_file))
    with pytest.raises(ValueError, match="unsubstituted variable"):
        resolve_credential()


def test_keyring_holding_a_variable(stdio_env):
    import keyring

    stdio_env.setenv("DATAMESH_TOKEN_KEYRING", "1")
    with (
        patch.object(keyring, "get_password", return_value="$DATAMESH_TOKEN"),
        pytest.raises(ValueError, match="unsubstituted variable"),
    ):
        resolve_credential()


def test_keyring_concurrent_first_calls_share_one_lookup(stdio_env):
    """Concurrent first calls must not each hit the store (one prompt)."""
    import threading
    import time

    import keyring

    def slow_lookup(*_: str) -> str:
        time.sleep(0.05)
        return "kr-token"

    stdio_env.setenv("DATAMESH_TOKEN_KEYRING", "1")
    results: list[str] = []
    with patch.object(keyring, "get_password", side_effect=slow_lookup) as get:
        threads = [
            threading.Thread(target=lambda: results.append(resolve_credential()))
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
    assert results == ["kr-token"] * 4
    get.assert_called_once()


@pytest.mark.parametrize(
    "content", ["tok-a\ntok-b\n", "tok a", "tok\x00a"], ids=["lines", "space", "nul"]
)
def test_token_file_with_whitespace_or_control_chars(stdio_env, tmp_path, content):
    token_file = tmp_path / "token"
    token_file.write_text(content)
    token_file.chmod(0o600)
    stdio_env.setenv("DATAMESH_TOKEN_FILE", str(token_file))
    with pytest.raises(ValueError, match="whitespace or control") as exc:
        resolve_credential()
    assert content.strip() not in str(exc.value)  # never echo the token


def test_env_token_stripped(stdio_env):
    stdio_env.setenv("DATAMESH_TOKEN", "  env-token\n")
    assert resolve_credential() == "env-token"


def test_keyring_unexpected_backend_error_wrapped(stdio_env):
    import keyring

    stdio_env.setenv("DATAMESH_TOKEN_KEYRING", "1")
    with (
        patch.object(keyring, "get_password", side_effect=RuntimeError("dbus")),
        pytest.raises(ValueError, match="credential store failed: RuntimeError"),
    ):
        resolve_credential()
