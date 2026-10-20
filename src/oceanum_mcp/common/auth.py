"""Auth providers for network transports.

Both verifiers deposit a connector-ready credential string in the access
token claims under CREDENTIAL_CLAIM; common.client requires it per request.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from contextvars import ContextVar
from typing import Any

import httpx2
from fastmcp.server.auth import (
    AccessToken,
    AuthProvider,
    MultiAuth,
    RemoteAuthProvider,
    TokenVerifier,
)
from fastmcp.server.auth.providers.jwt import JWTVerifier
from fastmcp.utilities.token_cache import TokenCache
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from oceanum_mcp.common.client import (
    CREDENTIAL_CLAIM,
    UNRESOLVED_VARIABLE_MESSAGE,
    is_unresolved_variable,
)
from oceanum_mcp.common.config import (
    auth0_audience,
    auth0_domain,
    auth_mode,
    datamesh_service,
    public_url,
)

# Verification results are cached so a chatty MCP session does not hit the
# gateway's /user/ endpoint on every request. Gateway-confirmed rejections
# (401/403) are cached briefly to blunt retry storms with a bad token;
# transport failures and unexpected statuses are never cached.
_VALID_TTL_S = 300
_INVALID_TTL_S = 60.0
_CACHE_MAX = 256


def _is_jwt_shaped(token: str) -> bool:
    """Whether the bearer has JWT structure (three dot-separated segments).

    Opaque Datamesh tokens never contain dots, so this cleanly routes each
    credential type to its verifier.
    """
    return token.count(".") == 2


HEADER_TOKEN_REJECTED = (
    "The Datamesh token in the X-DATAMESH-TOKEN header was not accepted "
    "(invalid, expired or revoked). Update the token in your MCP client "
    "configuration (get one from https://oceanum.io); header tokens need no "
    "sign-in."
)

GATEWAY_UNAVAILABLE = (
    "The Datamesh token in the X-DATAMESH-TOKEN header could not be verified: "
    "the Datamesh service did not respond as expected. This is not a "
    "problem with the token; retry shortly."
)

# A per-request holder DatameshHeaderMiddleware installs, recording why a
# credential failed: "placeholder" (set by the middleware), or "rejected" /
# "unavailable" (set by DatameshTokenVerifier). Only a 401 with a recorded
# reason is rewritten. Mutable, so marks survive any context copy in between.
_verification_outcome: ContextVar[dict[str, str] | None] = ContextVar(
    "_verification_outcome", default=None
)


def _mark_outcome(result: str) -> None:
    outcome = _verification_outcome.get()
    if outcome is not None:
        outcome["result"] = result


async def _send_invalid_token(send: Send, description: str) -> None:
    """A 401 invalid_token response with no OAuth discovery challenge.

    No resource_metadata: for a credential the client supplied itself, that
    pointer only sends clients (VS Code) into an OAuth flow that cannot fix
    the token. RFC 6750 §3.1 only asks for error="invalid_token" here; the
    challenge with resource_metadata stays on credential-less 401s.
    """
    body = json.dumps({"error": "invalid_token", "error_description": description})
    # error_description must be printable ASCII without '"' or '\' (§3).
    header_description = description.replace("…", "...")
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body.encode())).encode()),
                (
                    b"www-authenticate",
                    f'Bearer error="invalid_token", '
                    f'error_description="{header_description}"'.encode(),
                ),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body.encode()})


async def _send_unavailable(send: Send) -> None:
    """A 503 for a header token the gateway outage left unverified."""
    body = json.dumps(
        {"error": "temporarily_unavailable", "error_description": GATEWAY_UNAVAILABLE}
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 503,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"retry-after", b"5"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _replace_401(send: Send, outcome: dict[str, str], from_header: bool) -> Send:
    """Wrap send so a 401 from inside says why the credential failed, when
    the reason was recorded: a placeholder credential; or, for a header
    token, a gateway rejection (401) or a gateway outage (503). Any other
    401 passes through untouched."""
    replaced = False

    async def wrapped(message: Message) -> None:
        nonlocal replaced
        if message["type"] == "http.response.start" and message["status"] == 401:
            result = outcome.get("result")
            if result == "placeholder":
                await _send_invalid_token(send, UNRESOLVED_VARIABLE_MESSAGE)
            elif from_header and result == "rejected":
                await _send_invalid_token(send, HEADER_TOKEN_REJECTED)
            elif from_header and result == "unavailable":
                await _send_unavailable(send)
            else:
                await send(message)
                return
            replaced = True
        elif not (replaced and message["type"] == "http.response.body"):
            await send(message)

    return wrapped


class DatameshHeaderMiddleware:
    """Promotes X-DATAMESH-TOKEN to the Authorization bearer FastMCP reads.

    Datamesh clients conventionally send their token in the X-DATAMESH-TOKEN
    header (as the gateway itself expects), while JWTs arrive as
    Authorization bearers — so the header itself distinguishes the two
    credential types. Promotion only happens when no Authorization header is
    present, keeping the channels unambiguous when both are sent.

    Also explains the 401s FastMCP's generic wording gets wrong (OCE-362);
    requests that end in no 401 (e.g. public /.well-known routes) are
    untouched:
    - a credential that is an unsubstituted client variable such as
      "${input:datamesh-token}" gets a 401 saying so;
    - a rejected header token gets a 401 saying the header token is invalid,
      instead of FastMCP's "clear authentication tokens and reconnect" with
      an OAuth challenge, which header-token clients cannot act on; one left
      unverified by a gateway outage gets a 503 saying to retry.
    Only added when an auth provider is attached (see create_http_app).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers: list[tuple[bytes, bytes]] = scope["headers"]
        authorization = next((v for k, v in headers if k == b"authorization"), None)
        header_token = None
        if authorization is None:
            header_token = next(
                (v for k, v in headers if k == b"x-datamesh-token"), None
            )
            if header_token:
                scope = dict(scope)
                scope["headers"] = [
                    *headers,
                    (b"authorization", b"Bearer " + header_token),
                ]
        if header_token:
            credential = header_token
        elif authorization is not None and authorization[:7].lower() == b"bearer ":
            credential = authorization[7:]
        else:
            await self.app(scope, receive, send)
            return
        outcome: dict[str, str] = {}
        if is_unresolved_variable(credential.decode("latin-1").strip()):
            outcome["result"] = "placeholder"
        reset = _verification_outcome.set(outcome)
        try:
            await self.app(
                scope, receive, _replace_401(send, outcome, bool(header_token))
            )
        finally:
            _verification_outcome.reset(reset)


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
        self._http = httpx2.AsyncClient(timeout=10.0)
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
        if _is_jwt_shaped(token):
            # A JWT is never a Datamesh token: skip the gateway round trip
            # and leave the negative cache unpolluted (matters in auto mode,
            # where MultiAuth consults this verifier after the JWT one).
            return None
        if is_unresolved_variable(token):
            # Never a token (DatameshHeaderMiddleware reports it): skip the
            # gateway round trip and the negative cache.
            return None
        hit, cached = self._valid.get(token)
        if hit:
            return cached
        key = self._key(token)
        if self._is_known_invalid(key):
            _mark_outcome("rejected")
            return None
        try:
            result = await self._verify_with_gateway(token)
        except httpx2.HTTPError:
            # Gateway unreachable or in an unexpected state: fail closed
            # without caching, so recovery is immediate.
            _mark_outcome("unavailable")
            return None
        if result is None:
            _mark_outcome("rejected")
            self._mark_invalid(key)
        else:
            self._valid.set(token, result)
        return result

    async def _verify_with_gateway(self, token: str) -> AccessToken | None:
        """Returns None only for a gateway-confirmed rejection (401/403).

        Any other non-200 status raises httpx2.HTTPStatusError so the caller
        treats it as an outage, not an invalid token.
        """
        resp = await self._http.get(
            f"{self._service}/user/",
            headers={"X-DATAMESH-TOKEN": token},
        )
        if resp.status_code in (401, 403):
            return None
        if resp.status_code != 200:
            raise httpx2.HTTPStatusError(
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


def _jwt_provider() -> AuthProvider:
    """The Auth0 verifier, with OAuth discovery when a public URL is set.

    RemoteAuthProvider serves RFC 9728 Protected Resource Metadata naming the
    Auth0 tenant, which is how OAuth clients (claude.ai custom connectors)
    discover where to authorize. Requires OCEANUM_MCP_PUBLIC_URL, since the
    metadata must state the server's externally visible resource URL.
    """
    verifier = Auth0JWTVerifier()
    base = public_url()
    if base is None:
        return verifier
    return RemoteAuthProvider(
        token_verifier=verifier,
        authorization_servers=[f"https://{auth0_domain()}/"],
        base_url=base,
        resource_name="Oceanum MCP",
    )


def build_auth_provider() -> AuthProvider | None:
    """Build the auth provider for a network transport, per OCEANUM_MCP_AUTH."""
    mode = auth_mode()
    if mode == "auto":
        # Per-request credential detection: the JWT provider is consulted
        # first and fails fast (locally) on non-JWT bearers; the Datamesh
        # verifier declines JWT-shaped bearers, so each credential type is
        # only ever checked against its own backend. When the JWT side is a
        # RemoteAuthProvider it goes in the server slot, which contributes
        # the discovery routes as well as verification.
        jwt_provider = _jwt_provider()
        if isinstance(jwt_provider, RemoteAuthProvider):
            return MultiAuth(server=jwt_provider, verifiers=[DatameshTokenVerifier()])
        return MultiAuth(verifiers=[jwt_provider, DatameshTokenVerifier()])
    if mode == "datamesh":
        return DatameshTokenVerifier()
    if mode == "auth0":
        return _jwt_provider()
    return None
