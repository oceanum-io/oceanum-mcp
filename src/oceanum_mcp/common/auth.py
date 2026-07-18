"""Auth providers for network transports.

Both verifiers deposit a connector-ready credential string in the access
token claims under CREDENTIAL_CLAIM; common.client picks it up per request.
"""

from __future__ import annotations

import threading
import time

import httpx
from fastmcp.server.auth import AccessToken, AuthProvider, TokenVerifier
from fastmcp.server.auth.providers.jwt import JWTVerifier

from oceanum_mcp.common.client import CREDENTIAL_CLAIM
from oceanum_mcp.common.config import (
    auth0_audience,
    auth0_domain,
    auth_mode,
    datamesh_service,
)

# Verification results are cached so a chatty MCP session does not hit the
# gateway's /user/ endpoint on every request. Bounded; failures are cached
# briefly to blunt retry storms with a bad token.
_VALID_TTL_S = 300.0
_INVALID_TTL_S = 60.0
_CACHE_MAX = 256


class DatameshTokenVerifier(TokenVerifier):
    """Validates the presented bearer as a Datamesh token against the gateway.

    The gateway's /user/ endpoint returns 200 for a valid token and 401
    otherwise, and identifies the account, which becomes the token subject.
    """

    def __init__(self, service: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._service = service or datamesh_service()
        self._lock = threading.Lock()
        # token -> (expiry_monotonic, AccessToken | None)
        self._cache: dict[str, tuple[float, AccessToken | None]] = {}

    def _cached(self, token: str) -> tuple[bool, AccessToken | None]:
        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(token)
            if entry is not None and now < entry[0]:
                return True, entry[1]
        return False, None

    def _store(self, token: str, result: AccessToken | None) -> None:
        ttl = _VALID_TTL_S if result is not None else _INVALID_TTL_S
        with self._lock:
            if len(self._cache) >= _CACHE_MAX:
                # Drop expired entries first; if still full, drop arbitrary
                # entries — correctness only needs the cache to be a cache.
                now = time.monotonic()
                self._cache = {k: v for k, v in self._cache.items() if now < v[0]}
                while len(self._cache) >= _CACHE_MAX:
                    self._cache.pop(next(iter(self._cache)))
            self._cache[token] = (time.monotonic() + ttl, result)

    async def verify_token(self, token: str) -> AccessToken | None:
        hit, cached = self._cached(token)
        if hit:
            return cached
        try:
            result = await self._verify_with_gateway(token)
        except httpx.HTTPError:
            # Gateway unreachable: fail closed but do not cache the outcome,
            # so recovery is immediate once the gateway is back.
            return None
        self._store(token, result)
        return result

    async def _verify_with_gateway(self, token: str) -> AccessToken | None:
        """Raises httpx.HTTPError when the gateway cannot be reached."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{self._service}/user/",
                headers={"X-DATAMESH-TOKEN": token},
            )
        if resp.status_code != 200:
            return None
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
            claims={CREDENTIAL_CLAIM: token},
        )


class Auth0JWTVerifier(JWTVerifier):
    """Validates Auth0-issued JWTs and forwards them to the gateway.

    The Datamesh gateway accepts "Bearer <jwt>" credentials directly, so the
    verified JWT is stored as the connector credential without exchange.
    """

    def __init__(
        self, domain: str | None = None, audience: str | None = None, **kwargs
    ):
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


def build_auth_provider(mode: str | None = None) -> AuthProvider | None:
    """Build the auth provider for a network transport, per OCEANUM_MCP_AUTH."""
    mode = mode or auth_mode()
    if mode == "datamesh":
        return DatameshTokenVerifier()
    if mode == "auth0":
        return Auth0JWTVerifier()
    return None
