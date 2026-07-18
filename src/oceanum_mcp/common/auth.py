"""Auth providers for network transports.

Both verifiers deposit a connector-ready credential string in the access
token claims under CREDENTIAL_CLAIM; common.client requires it per request.
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any

import httpx
from fastmcp.server.auth import AccessToken, AuthProvider, TokenVerifier
from fastmcp.server.auth.providers.jwt import JWTVerifier
from fastmcp.utilities.token_cache import TokenCache

from oceanum_mcp.common.client import CREDENTIAL_CLAIM
from oceanum_mcp.common.config import (
    auth0_audience,
    auth0_domain,
    auth_mode,
    datamesh_service,
)

# Verification results are cached so a chatty MCP session does not hit the
# gateway's /user/ endpoint on every request. Gateway-confirmed rejections
# (401/403) are cached briefly to blunt retry storms with a bad token;
# transport failures and unexpected statuses are never cached.
_VALID_TTL_S = 300
_INVALID_TTL_S = 60.0
_CACHE_MAX = 256


class DatameshTokenVerifier(TokenVerifier):
    """Validates the presented bearer as a Datamesh token against the gateway.

    The gateway's /user/ endpoint returns 200 for a valid token, 401/403
    otherwise, and identifies the account, which becomes the token subject.
    Outcome handling:
    - 200 -> valid; cached (fastmcp TokenCache: hashed keys, TTL).
    - 401/403 -> invalid; cached briefly under a hashed key.
    - any other status, or a transport error -> fail closed, UNCACHED, so a
      gateway blip never locks a valid token out past the blip itself.
    """

    def __init__(self, service: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._service = service or datamesh_service()
        # One client for the verifier's lifetime: per-verification clients
        # would pay TCP+TLS setup on every cache miss. Never closed — the
        # verifier lives as long as the server process.
        self._http = httpx.AsyncClient(timeout=10.0)
        self._valid = TokenCache(ttl_seconds=_VALID_TTL_S, max_size=_CACHE_MAX)
        self._lock = threading.Lock()
        # sha256(token) -> monotonic expiry; only gateway-confirmed rejections.
        self._invalid: dict[str, float] = {}

    @staticmethod
    def _key(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def _is_known_invalid(self, key: str) -> bool:
        with self._lock:
            expiry = self._invalid.get(key)
            return expiry is not None and time.monotonic() < expiry

    def _mark_invalid(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            if len(self._invalid) >= _CACHE_MAX:
                self._invalid = {k: v for k, v in self._invalid.items() if now < v}
                if len(self._invalid) >= _CACHE_MAX:
                    self._invalid.pop(next(iter(self._invalid)))
            self._invalid[key] = now + _INVALID_TTL_S

    async def verify_token(self, token: str) -> AccessToken | None:
        hit, cached = self._valid.get(token)
        if hit:
            return cached
        key = self._key(token)
        if self._is_known_invalid(key):
            return None
        try:
            result = await self._verify_with_gateway(token)
        except httpx.HTTPError:
            # Gateway unreachable or in an unexpected state: fail closed
            # without caching, so recovery is immediate.
            return None
        if result is None:
            self._mark_invalid(key)
        else:
            self._valid.set(token, result)
        return result

    async def _verify_with_gateway(self, token: str) -> AccessToken | None:
        """Returns None only for a gateway-confirmed rejection (401/403).

        Any other non-200 status raises httpx.HTTPStatusError so the caller
        treats it as an outage, not an invalid token.
        """
        resp = await self._http.get(
            f"{self._service}/user/",
            headers={"X-DATAMESH-TOKEN": token},
        )
        if resp.status_code in (401, 403):
            return None
        if resp.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Datamesh gateway returned {resp.status_code} during token "
                "verification",
                request=resp.request,
                response=resp,
            )
        subject = None
        try:
            users = resp.json()
            if isinstance(users, list) and users:
                subject = users[0].get("username")
        except ValueError:
            pass
        return AccessToken(
            token=token,
            client_id=subject or "datamesh-user",
            subject=subject,
            scopes=[],
            # Bounds the revocation window: a token revoked at the gateway is
            # honored for at most _VALID_TTL_S after its last verification.
            expires_at=int(time.time()) + _VALID_TTL_S,
            claims={CREDENTIAL_CLAIM: token},
        )


class Auth0JWTVerifier(JWTVerifier):
    """Validates Auth0-issued JWTs and forwards them to the gateway.

    The Datamesh gateway accepts "Bearer <jwt>" credentials directly, so the
    verified JWT is stored as the connector credential without exchange.
    """

    def __init__(
        self, domain: str | None = None, audience: str | None = None, **kwargs: Any
    ) -> None:
        domain = domain or auth0_domain()
        super().__init__(
            jwks_uri=f"https://{domain}/.well-known/jwks.json",
            issuer=f"https://{domain}/",
            audience=audience or auth0_audience(),
            **kwargs,
        )

    async def verify_token(self, token: str) -> AccessToken | None:
        result = await super().verify_token(token)
        if result is not None:
            result.claims[CREDENTIAL_CLAIM] = f"Bearer {token}"
        return result


def build_auth_provider() -> AuthProvider | None:
    """Build the auth provider for a network transport, per OCEANUM_MCP_AUTH."""
    mode = auth_mode()
    if mode == "datamesh":
        return DatameshTokenVerifier()
    if mode == "auth0":
        return Auth0JWTVerifier()
    return None
