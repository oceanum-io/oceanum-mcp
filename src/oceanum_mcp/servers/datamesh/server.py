"""Oceanum Datamesh MCP server.

Exposes the Oceanum Datamesh API as MCP tools for AI assistants.

Error conventions:
- Invalid parameter combinations raise ToolError (the caller should fix the
  tool call).
- Datamesh runtime errors return a JSON object with an "error" key and the
  canonical query echoed back, so the caller can self-correct. An unknown
  datasource id or variable adds a "suggestions" list of close matches; a
  datasource that exists but is not shared with the caller says so instead.
  A query variable named by the exact CF standard_name of exactly one of the
  datasource's variables is resolved to it, reported in "resolved_variables".
- Results that are too large to return inline return a JSON object with a
  "refused" key, the staged size, and concrete alternatives.
"""

from __future__ import annotations

import difflib
import json
import logging
import math
import os
import re
import shutil
import threading
import time
import uuid
import warnings as _warnings
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial, wraps
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import pandas as pd
import requests
import xarray as xr
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pandas.tseries.frequencies import to_offset
from urllib3.exceptions import ReadTimeoutError

from oceanum.datamesh import Connector
from oceanum.datamesh.exceptions import (
    DatameshConnectError,
    DatameshQueryError,
    DatameshSessionError,
)
from oceanum.datamesh.query import Container, CoordSelector, GeoFilter, Query, Stage
from oceanum.datamesh.session import Session
from oceanum.datamesh.utils import (
    DATAMESH_CONNECT_TIMEOUT,
    DATAMESH_STAGE_READ_TIMEOUT,
)

from oceanum_mcp.common.client import get_datamesh_connector
from oceanum_mcp.common.config import (
    export_dir,
    is_network_transport,
    is_read_only,
    max_inline_bytes,
    stage_timeout,
)
from oceanum_mcp.common.formatting import (
    export_clause,
    format_datasource,
    format_datasource_bounded,
    format_datasource_summary,
    human_bytes,
    summarize_data,
    to_json,
)

# Everything the oceanum library can raise on a gateway interaction.
# Session.acquire wraps all its failures (auth, network) in DatameshSessionError.
_DATAMESH_ERRORS = (DatameshConnectError, DatameshQueryError, DatameshSessionError)

# Server caps tabular (dataframe/geodataframe) query results at this many rows.
DATAMESH_ROW_CAP = 2_000_000

# Ceiling on bytes loaded into memory for a tabular export (local/stdio path).
MAX_EXPORT_FRAME_BYTES = 2_000_000_000

# Ceiling on a local (stdio) dataset export. Real cancellation of an
# in-flight write is not possible yet (the to_netcdf call runs to completion in
# a worker thread), so this bounds the damage of a runaway export: at the
# ~145 MB/s observed in OCE-296 it is ~70 s of disk writing, versus the 36.5 GB
# written before that export was killed. 10x the "ask the user" threshold, so
# multi-GB exports the user has agreed to still work; anything larger belongs
# on the hosted download link or in direct oceanum library code.
MAX_EXPORT_DATASET_BYTES = 10_000_000_000

# Above this staged size stage_query leads with narrowing, states the cost of
# exporting as-is, and tells the agent to ask the user before exporting.
LARGE_EXPORT_BYTES = 1_000_000_000

# Free space required before a local export writes, as a multiple of the staged
# size. NetCDF/Parquet land at or below the staged (in-memory) size; CSV text
# runs well above binary floats.
_EXPORT_DISK_FACTOR = {"netcdf": 1.1, "parquet": 1.1, "csv": 3.0}

# Above this staged size the hosted download link is flagged as a large
# transfer. Not a refusal (that is MAX_HOSTED_EXPORT_BYTES): the URL streams
# from the gateway at no cost to this server, and exporting large results is
# the whole point of the download path.
LARGE_DOWNLOAD_BYTES = 2_000_000_000

# Above this staged size hosted export refuses to return a download link
# (OCE-299). The link is a bearer URL: anyone holding it can fetch the data
# with no credential until it expires, so until the gateway issues scoped
# links (OCE-317) this bounds what one leaked link, or one runaway agent call,
# can pull. Equal to the local dataset cap (MAX_EXPORT_DATASET_BYTES) so an
# export that works on stdio also works hosted; 2-10 GB stays warn-and-allow.
MAX_HOSTED_EXPORT_BYTES = MAX_EXPORT_DATASET_BYTES

# The tool's format names mapped to the gateway's `&f=` download tokens.
_GATEWAY_FORMAT = {"netcdf": "nc", "parquet": "parquet", "csv": "csv"}

READ_TOOL = {"readOnlyHint": True, "openWorldHint": True}

# Instructions are transport-neutral: they describe the stage -> narrow
# workflow without naming export_query, whose behaviour differs by transport
# (download link on hosted, local file on stdio). The runtime "result too
# large" messages (via export_clause()) and the export_query tool's own
# docstring carry the transport-specific wording, so nothing here freezes a
# transport-specific string at import.
mcp = FastMCP(
    "Oceanum Datamesh",
    instructions=(
        "Access the Oceanum Datamesh platform for ocean and environmental data.\n"
        "Workflow: search_catalog to discover datasets -> get_datasource_info for "
        "schema and coverage -> stage_query to learn the result size WITHOUT "
        "downloading -> query_data for small results inline, narrowing large "
        "results with filters, aggregation, or time_resolution downsampling.\n"
        "Never pull large data inline: stage first, then shrink results with "
        "time/geo/level filters, aggregation, or time_resolution downsampling. "
        "Times are ISO 8601 (UTC assumed if naive); sizes are bytes."
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WARNINGS_LOCK = threading.Lock()


@contextmanager
def _captured_warnings(collected: list[str]) -> Iterator[None]:
    """Capture Python warnings raised by the oceanum library.

    The library signals silent degradation (row caps, lazy dask fallback)
    via warnings.warn, which would otherwise be lost to stderr.
    catch_warnings mutates process-global state and FastMCP runs sync tools
    in worker threads, so captures are serialized with a lock.
    """
    with _WARNINGS_LOCK:
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            yield
        collected.extend(str(w.message) for w in caught)


class GatewayTimeout(DatameshConnectError):
    """A Datamesh request exceeded OCEANUM_MCP_STAGE_TIMEOUT (OCE-294).

    Subclasses DatameshConnectError so the tools' existing runtime-error path
    reports it as {"error": ..., "query": ...}. Deliberately NOT a
    requests.RequestException: the SDK's retried_request retries those (up to
    8 attempts), which would multiply the wait instead of ending it.
    """


@dataclass(frozen=True)
class _TimeoutBudget:
    """The OCEANUM_MCP_STAGE_TIMEOUT budget of one bounded block."""

    cap: float
    deadline: float  # time.monotonic() value
    message: str


# Budget for gateway requests made by the current tool call. FastMCP runs each
# sync tool call in its own worker thread, so the context variable scopes the
# budget to one call even though connectors are shared between calls.
_TIMEOUT_BUDGET: ContextVar[_TimeoutBudget | None] = ContextVar(
    "_TIMEOUT_BUDGET", default=None
)
_CAP_INSTALL_LOCK = threading.Lock()


def _is_read_timeout(exc: requests.RequestException) -> bool:
    """True for a read timeout, whether raised before the response headers
    (ReadTimeout) or while reading the body (requests wraps that in a
    ConnectionError around urllib3's ReadTimeoutError)."""
    if isinstance(exc, requests.ReadTimeout):
        return True
    return bool(exc.args) and isinstance(exc.args[0], ReadTimeoutError)


class _CappedHTTPSession:
    """Wraps a Connector's HTTPSession to bound long requests inside
    _gateway_timeout().

    The oceanum SDK sends stage and query requests with a 900 s read timeout
    and retries a failed request up to 8 times (sleeping 30 s after each 502).
    Inside _gateway_timeout(), a request whose SDK read timeout exceeds the
    cap gets the time left before the block's deadline as its read timeout
    (requests closes the socket when it fires), and the timeout, or a retry
    attempted after the deadline, raises GatewayTimeout, which the SDK does
    not retry. Requests with a shorter SDK timeout (session acquire/close)
    and requests outside the block pass through untouched.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        # Read _inner via __dict__: during copy/unpickle it is not set yet,
        # and self._inner would recurse back into __getattr__.
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def request(self, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
        budget = _TIMEOUT_BUDGET.get()
        if budget is None:
            return self._inner.request(method, url, *args, **kwargs)
        timeout = kwargs.get("timeout")
        connect, read = timeout if isinstance(timeout, tuple) else (timeout, timeout)
        if read is not None and read < budget.cap:
            # A short request (e.g. session acquire/close): the SDK's own
            # timeout and retries already bound it.
            return self._inner.request(method, url, *args, **kwargs)
        remaining = budget.deadline - time.monotonic()
        if remaining <= 0:
            raise GatewayTimeout(budget.message)
        kwargs["timeout"] = (connect, remaining)
        try:
            return self._inner.request(method, url, *args, **kwargs)
        except requests.RequestException as exc:
            if not _is_read_timeout(exc):
                raise
            raise GatewayTimeout(budget.message) from exc


@contextmanager
def _gateway_timeout(
    conn: Connector,
    what: str,
    hint: str = "The stage may be cold or slow: retrying once may succeed. If "
    "it times out again, narrow the query (shorter time range, smaller bbox, "
    "fewer variables) or use a coarser time_resolution",
) -> Iterator[None]:
    """Bound the long gateway requests `conn` makes in this block (OCE-294).

    Requests the SDK sends with a read timeout at or above
    OCEANUM_MCP_STAGE_TIMEOUT (stage, download stage, query, load) share a
    deadline of that many seconds from entering the block: each waits at most
    the time left, and the SDK's retries stop once it has passed. On expiry
    the HTTP socket is closed and GatewayTimeout raised, freeing this worker
    thread within about the timeout (plus at most one in-flight 30 s SDK
    back-off after a 502). The deadline is checked when a request starts and
    by the read timeout, so a response that keeps trickling bytes is only
    bounded by inactivity. Whatever the gateway was computing is not
    cancelled by this server: whether it stops when the client disconnects
    is up to the gateway. Session acquire/close go through the same session:
    while the timeout is above their SDK read timeout (DATAMESH_READ_TIMEOUT,
    10 s by default) they pass through untouched; with a lower timeout they
    are capped and count against the deadline like any other request. Lazy
    zarr chunk reads use their own session and are not bounded here.
    """
    cap = stage_timeout()
    with _CAP_INSTALL_LOCK:
        # Connectors are cached and shared across calls: wrap each one once.
        if not isinstance(conn.http_session, _CappedHTTPSession):
            conn.http_session = _CappedHTTPSession(conn.http_session)
    budget = _TimeoutBudget(
        cap=cap,
        deadline=time.monotonic() + cap,
        message=f"Datamesh {what} timed out after {cap:g}s. {hint}. (Server "
        "operators can raise this limit with OCEANUM_MCP_STAGE_TIMEOUT.)",
    )
    token = _TIMEOUT_BUDGET.set(budget)
    try:
        yield
    finally:
        _TIMEOUT_BUDGET.reset(token)


def _stage(conn: Connector, query: Query) -> Stage | None:
    """Stage a query on the Datamesh gateway without downloading data.

    Uses the connector's private staging request — oceanum<2 has no public
    staging API; the dependency pin in pyproject.toml guards this. Once
    oceanum grows a public Connector.stage() (and a way to execute a query
    from an existing stage), switch to it: that also removes the second
    staging round-trip conn.query() currently performs internally.

    Bounded by OCEANUM_MCP_STAGE_TIMEOUT (see _gateway_timeout): raises
    GatewayTimeout instead of waiting out the SDK's 900 s x retries.
    """
    session = Session.acquire(conn)
    try:
        with _gateway_timeout(conn, "staging"):
            return conn._stage_request(query, session)
    except AttributeError as exc:  # private API drift within the 1.x pin
        raise ToolError(
            "The installed oceanum version no longer exposes the staging "
            "internals this server relies on; install a version matching "
            "the pyproject.toml pin."
        ) from exc
    finally:
        session.close()


def _download_stage(conn: Connector, query: Query) -> dict[str, Any] | None:
    """Stage a query for download, returning the gateway's response dict.

    POSTs to the gateway's /oceanql/download/ endpoint, which the oceanum<2
    library does not wrap. The response carries a self-authenticating signed
    `url` (append `&f=<format>` to pick a format), plus `formats`, `size`, and
    `container`. Mirrors Connector._stage_request's auth/error handling.
    Returns None when no data matches (HTTP 204). Bounded by
    OCEANUM_MCP_STAGE_TIMEOUT like _stage.
    """
    session = Session.acquire(conn)
    try:
        with _gateway_timeout(conn, "download staging"):
            resp = conn._retried_request(
                f"{conn._gateway}/oceanql/download/",
                method="POST",
                headers=session.header,
                data=query.model_dump_json(warnings=False),
                # Match _stage_request's long read timeout: preparing a
                # download stage for a large export can far exceed the 10s
                # default. _gateway_timeout caps it at the stage timeout.
                timeout=(DATAMESH_CONNECT_TIMEOUT, DATAMESH_STAGE_READ_TIMEOUT),
            )
    except AttributeError as exc:  # private API drift within the 1.x pin
        raise ToolError(
            "The installed oceanum version no longer exposes the connection "
            "internals this server relies on; install a version matching "
            "the pyproject.toml pin."
        ) from exc
    finally:
        session.close()
    if resp.status_code == 204:
        return None
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail")
        except ValueError:
            detail = None
        if detail:
            raise DatameshQueryError(detail)
        raise DatameshConnectError("Datamesh server error: " + resp.text)
    return resp.json()


def _query_echo(query: Query) -> dict[str, Any]:
    """Canonical JSON form of a query, for echoing in responses."""
    return query.model_dump(mode="json", exclude_none=True, warnings=False)


# ---------------------------------------------------------------------------
# "Did you mean" suggestions for unknown datasources and variables (OCE-303)
# ---------------------------------------------------------------------------

# Catalog entries fetched to rank datasource suggestions (one request).
SUGGEST_CATALOG_LIMIT = 20
# Suggestions returned per unknown datasource id or variable.
MAX_SUGGESTIONS = 5
# Seconds a tool call waits for its suggestions before returning the original
# error. The lookups are SDK metadata requests (10 s read timeout, up to 8
# retries, 30 s back-off after a 502), so without this cap a degraded metadata
# server could hold an already-failed tool call for minutes.
SUGGEST_TIMEOUT = 10.0
# Minimum similarity (0-1) for a candidate to be suggested.
_SUGGEST_CUTOFF = 0.7
# Score of a candidate containing every word of the requested name (e.g.
# "significant_wave_height" in "sea_surface_wave_significant_height"): above
# the cutoff, below a close typo.
_CONTAINS_ALL_WORDS_SCORE = 0.8
# Words the catalog's keyword search (Postgres websearch_to_tsquery) treats as
# operators or stop words; dropped from the suggestion search.
_SEARCH_OPERATOR_WORDS = frozenset({"or", "and", "not"})

# The SDK's get_datasource raises DatameshConnectError with exactly
# "Datasource <id> not found" for a metadata-server 404, and with the server's
# detail for a 403 (the datasource exists but is not shared with the caller):
# "You do not have permission to access this datasource". A 401 ("not
# Authorized") is a credential problem and 5xx bodies are server faults:
# neither is matched.
_FORBIDDEN_RE = re.compile(r"You do not have permission\b")
# The query engine reports a requested variable missing from the datasource as
# "Invalid variable selection - variable not found: '<name>'".
_VARIABLE_MISSING_RE = re.compile(
    r"variables? not found|no variable named", re.IGNORECASE
)

# Lookups run in daemon threads, at most this many at once: a lookup that
# times out keeps its thread until the SDK gives up, and must neither delay
# interpreter exit nor pile up. With every slot busy, lookups are skipped.
_SUGGEST_SLOTS = threading.BoundedSemaphore(4)
logger = logging.getLogger(__name__)

# Variables substituted by exact CF standard_name in the current tool call
# (see _stage_resolving), reported by _reporting_resolved_variables.
_RESOLVED_VARIABLES: ContextVar[dict[str, str] | None] = ContextVar(
    "_RESOLVED_VARIABLES", default=None
)


def _run_bounded(fn: Callable[[], Any], datasource_id: str) -> Any:
    """fn() run in a daemon thread for at most SUGGEST_TIMEOUT seconds.

    Returns None, logging a warning, if fn raised, timed out, or no lookup
    slot was free: a best-effort lookup must never replace the error it
    enriches, nor hold the tool call or interpreter exit.
    """
    if not _SUGGEST_SLOTS.acquire(blocking=False):
        logger.warning("Suggestions for %r skipped: lookups busy", datasource_id)
        return None
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["result"] = fn()
        except Exception as exc:
            box["error"] = exc
        finally:
            _SUGGEST_SLOTS.release()

    thread = threading.Thread(target=target, name="datamesh-suggest", daemon=True)
    try:
        thread.start()
    except RuntimeError:  # e.g. "can't start new thread"
        _SUGGEST_SLOTS.release()
        logger.warning("Suggestions for %r skipped", datasource_id, exc_info=True)
        return None
    thread.join(SUGGEST_TIMEOUT)
    if thread.is_alive():
        logger.warning(
            "Suggestions for %r timed out after %ss", datasource_id, SUGGEST_TIMEOUT
        )
        return None
    if "error" in box:
        logger.warning(
            "Suggestions for %r failed", datasource_id, exc_info=box["error"]
        )
        return None
    return box.get("result")


def _engine_missing_re(datasource_id: str) -> re.Pattern[str]:
    """The query engine's 404 for a datasource id, which it returns both for a
    missing datasource and for one not shared with the caller ("Datasource
    <id> not found or not authorized")."""
    return re.compile(
        rf"Datasource {re.escape(datasource_id)} (?:not found|does not exist)",
        re.IGNORECASE,
    )


def _norm(text: str) -> str:
    """Lower-case text with every run of non-alphanumerics folded to "_"."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _similarity(wanted: str, candidate: str) -> tuple[float, float]:
    """(score, ratio) of a requested name against a candidate name or label.

    ratio is difflib's character-sequence ratio (catches typos). score is
    ratio, raised to _CONTAINS_ALL_WORDS_SCORE when the candidate contains
    every word of the request (catches reordered or longer names). A partial
    word overlap does not count: "sea_surface_temperature" must not suggest
    "sea_surface_salinity".
    """
    a, b = _norm(wanted), _norm(candidate)
    if not a or not b:
        return 0.0, 0.0
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    if set(a.split("_")) <= set(b.split("_")):
        return max(ratio, _CONTAINS_ALL_WORDS_SCORE), ratio
    return ratio, ratio


def _best_matches(wanted: str, candidates: dict[str, list[str]]) -> list[str]:
    """Up to MAX_SUGGESTIONS candidate keys most similar to wanted.

    candidates maps each key to the labels it is matched on (the key itself
    plus e.g. a datasource name or a variable's standard_name/long_name).
    Ranked by score, then by character similarity, then by key.
    """
    scored = []
    for key, labels in candidates.items():
        score, ratio = max(
            (_similarity(wanted, label) for label in labels if label),
            default=(0.0, 0.0),
        )
        if score >= _SUGGEST_CUTOFF:
            scored.append((-score, -ratio, key))
    return [key for *_, key in sorted(scored)[:MAX_SUGGESTIONS]]


def _datasource_suggestions(
    conn: Connector, datasource_id: str
) -> list[dict[str, Any]]:
    """Visible datasources whose id or name resembles datasource_id.

    One bounded catalog search on the id's words (OR-ed, so a single typo'd
    word does not empty a keyword search); the catalog only lists datasources
    shared with the caller, so entitlement is respected.
    """
    words = [
        w for w in _norm(datasource_id).split("_") if w not in _SEARCH_OPERATOR_WORDS
    ]
    if not words:
        return []
    catalog = conn.get_catalog(search=" or ".join(words), limit=SUGGEST_CATALOG_LIMIT)
    names: dict[str, str] = {}
    for ds in catalog:
        if ds is not None and ds.id != datasource_id:
            names[ds.id] = ds.name or ""
    ranked = _best_matches(datasource_id, {i: [i, n] for i, n in names.items()})
    return [{"id": i, "name": names[i]} for i in ranked]


def _metadata_error_fields(
    conn: Connector, datasource_id: str, message: str
) -> dict[str, Any] | None:
    """Error fields for a get_datasource failure on datasource_id: a "no
    access" message for a 403, "did you mean" suggestions for a 404, and None
    for anything else."""
    if _FORBIDDEN_RE.match(message):
        return {
            "error": f"You don't have access to datasource {datasource_id!r}: it "
            "exists but is not shared with your account. Ask its owner or "
            "your Oceanum administrator for access."
        }
    if message != f"Datasource {datasource_id} not found":
        return None
    suggestions = _datasource_suggestions(conn, datasource_id)
    hint = (
        "did you mean one of the suggestions?"
        if suggestions
        else "no similar ids found; use search_catalog to find one."
    )
    return {
        "error": f"Datasource {datasource_id!r} not found; {hint}",
        "suggestions": suggestions,
    }


def _engine_missing_fields(
    conn: Connector, datasource_id: str
) -> dict[str, Any] | None:
    """Error fields for the query engine's "not found or not authorized": the
    metadata server tells a missing datasource (404) from one not shared with
    the caller (403); the engine does not."""
    try:
        conn.get_datasource(datasource_id)
    except _DATAMESH_ERRORS as exc:
        return _metadata_error_fields(conn, datasource_id, str(exc))
    return None  # visible to the caller: keep the engine's own error


def _best_effort(fn: Callable[[], Any], datasource_id: str) -> Any:
    """fn() (in-memory work, no I/O), or None, logging a warning, if it
    raised: a bug in building suggestions must not replace the error."""
    try:
        return fn()
    except Exception:
        logger.warning("Suggestions for %r failed", datasource_id, exc_info=True)
        return None


def _datasource_schema(conn: Connector, datasource_id: str, exc: Exception) -> Any:
    """The caller's own get_datasource record for datasource_id, fetched at
    most once per error (cached on exc) and bounded by _run_bounded; None if
    the lookup failed."""
    if not hasattr(exc, "_datamesh_schema_lookup"):
        exc._datamesh_schema_lookup = _run_bounded(  # type: ignore[attr-defined]
            partial(conn.get_datasource, datasource_id), datasource_id
        )
    return exc._datamesh_schema_lookup  # type: ignore[attr-defined]


def _unknown_variables(datasource: Any, variables: list[str]) -> list[str]:
    """Requested variables that are neither data variables nor coordinates of
    datasource; empty if its schema is unknown."""
    data_vars = datasource.variables or {}
    if not data_vars:
        return []
    coords = getattr(datasource.dataschema, "coords", None) or {}
    return [v for v in variables if v not in data_vars and v not in coords]


def _variable_attrs(schema: Any) -> dict[str, Any]:
    return (schema or {}).get("attrs") or {}


def _variable_fields(
    datasource_id: str, datasource: Any, variables: list[str]
) -> dict[str, Any] | None:
    """Error fields for requested variables the datasource lacks, with the
    closest variable names (matched on name, standard_name and long_name).
    None if every requested variable exists, or the schema is unknown."""
    unknown = _unknown_variables(datasource, variables)
    if not unknown:
        return None
    candidates: dict[str, list[str]] = {}
    for name, schema in datasource.variables.items():
        attrs = _variable_attrs(schema)
        labels = [str(name)]
        labels += [
            str(attrs[k]) for k in ("standard_name", "long_name") if attrs.get(k)
        ]
        candidates[str(name)] = labels
    return {
        "error": f"Variable(s) not in datasource {datasource_id!r}: "
        f"{', '.join(unknown)}. See the suggestions; get_datasource_info lists "
        "every variable.",
        "suggestions": [
            {"variable": v, "did_you_mean": _best_matches(v, candidates)}
            for v in unknown
        ],
    }


def _standard_name_aliases(
    datasource: Any, variables: list[str]
) -> dict[str, str] | None:
    """{requested: variable} for every unknown requested variable that
    EXACTLY equals the CF standard_name of exactly one of the datasource's
    variables. None if any unknown variable has no such unique match: an
    ambiguous or fuzzy alias is never resolved."""
    unknown = _unknown_variables(datasource, variables)
    if not unknown:
        return None
    aliases: dict[str, str] = {}
    for wanted in unknown:
        matches = [
            str(name)
            for name, schema in datasource.variables.items()
            if _variable_attrs(schema).get("standard_name") == wanted
        ]
        if len(matches) != 1:
            return None
        aliases[wanted] = matches[0]
    return aliases


def _substituted(query: Query, aliases: dict[str, str]) -> Query:
    """query with its variables renamed per aliases (duplicates dropped)."""
    if not aliases or not query.variables:
        return query
    variables = list(dict.fromkeys(aliases.get(v, v) for v in query.variables))
    return query.model_copy(update={"variables": variables})


def _stage_resolving(
    conn: Connector, query: Query, stage_fn: Callable[[Connector, Query], Any]
) -> tuple[Any, dict[str, str]]:
    """stage_fn(conn, query), retried once with variables resolved by CF
    standard_name if the engine reports a requested variable missing.

    Returns (result, aliases); aliases is empty unless the retry ran and
    succeeded, in which case the caller must apply _substituted(query,
    aliases) to the query it goes on to use. Only the caller's own
    get_datasource record is consulted (one bounded fetch, reused by
    _error_json on failure). If nothing resolves, or the retry still reports
    a missing variable, the ORIGINAL error is raised; any other retry error
    (e.g. GatewayTimeout) is raised as is. The retry is a second stage with
    its own OCEANUM_MCP_STAGE_TIMEOUT, so this error path can take up to
    twice the stage timeout plus SUGGEST_TIMEOUT.
    """
    try:
        return stage_fn(conn, query), {}
    except _DATAMESH_ERRORS as exc:
        if (
            isinstance(exc, GatewayTimeout)
            or not query.variables
            or not _VARIABLE_MISSING_RE.search(str(exc))
        ):
            raise
        datasource = _datasource_schema(conn, query.datasource, exc)
        aliases = (
            _best_effort(
                partial(_standard_name_aliases, datasource, query.variables),
                query.datasource,
            )
            if datasource is not None
            else None
        )
        if not aliases:
            raise
        try:
            result = stage_fn(conn, _substituted(query, aliases))
        except _DATAMESH_ERRORS as retry_exc:
            if _VARIABLE_MISSING_RE.search(str(retry_exc)):
                raise exc from None
            # Anything else (a timeout, a 5xx) is the real problem now: the
            # variables did resolve, so do not report them as missing.
            raise
    recorder = _RESOLVED_VARIABLES.get()
    if recorder is not None:
        recorder.update(aliases)
    return result, aliases


def _reporting_resolved_variables(tool: Callable[..., str]) -> Callable[..., str]:
    """Wrap a query tool so a response whose variables were resolved by
    _stage_resolving reports them as "resolved_variables" ({requested:
    used}); responses without a resolution are returned unchanged."""

    @wraps(tool)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        resolved: dict[str, str] = {}
        token = _RESOLVED_VARIABLES.set(resolved)
        try:
            out = tool(*args, **kwargs)
        finally:
            _RESOLVED_VARIABLES.reset(token)
        if not resolved:
            return out
        try:
            payload = json.loads(out)
        except ValueError:
            return out
        if not isinstance(payload, dict):
            return out
        return to_json({**payload, "resolved_variables": resolved})

    return wrapper


def _error_json(
    conn: Connector,
    datasource_id: str,
    exc: Exception,
    context: dict[str, Any],
    *,
    variables: list[str] | None = None,
    from_metadata: bool = False,
) -> str:
    """JSON for a Datamesh error on datasource_id, plus context (the echoed
    query, or the datasource_id).

    If the error says the datasource or a requested variable does not exist,
    it is restated with "did you mean" suggestions; a datasource that exists
    but is not shared with the caller is reported as such, without
    suggestions. from_metadata marks exc as raised by get_datasource itself;
    otherwise it is a query-engine error. The lookups (at most one
    get_datasource and one bounded catalog search) run only on those error
    paths, each bounded by _run_bounded; if they fail or time out the
    original error is returned unchanged.
    """
    message = str(exc)
    fields = None
    if from_metadata:
        fields = _run_bounded(
            partial(_metadata_error_fields, conn, datasource_id, message),
            datasource_id,
        )
    elif isinstance(exc, GatewayTimeout):
        pass
    elif _engine_missing_re(datasource_id).search(message):
        fields = _run_bounded(
            partial(_engine_missing_fields, conn, datasource_id), datasource_id
        )
    elif variables and _VARIABLE_MISSING_RE.search(message):
        datasource = _datasource_schema(conn, datasource_id, exc)
        if datasource is not None:
            fields = _best_effort(
                partial(_variable_fields, datasource_id, datasource, variables),
                datasource_id,
            )
    return to_json({**(fields or {"error": message}), **context})


# Coordinate keys Datamesh applies `limit` along (time, ensemble, quantile):
# it keeps the last N steps of each, mirrored by _limit_result.
_LIMIT_COORD_KEYS = ("t", "e", "q")


def _resamples_or_aggregates(query: Query) -> bool:
    """True if Datamesh resamples (time_resolution) or aggregates this query."""
    tf = query.timefilter
    resamples = tf is not None and getattr(tf, "resolution", None) not in (
        None,
        "native",
    )
    return resamples or query.aggregate is not None


def _split_limit(query: Query) -> tuple[Query, int | None]:
    """The query to send to Datamesh, and any limit to apply to its result.

    OCE-298: Datamesh applies `limit` to native records BEFORE time_resolution
    resampling or aggregation, so with those the limit is stripped from the
    query sent (and from the size check, which then sees the full
    resampled/aggregated result) and applied afterwards via _limit_result.
    Responses keep echoing the caller's query, limit included.
    """
    if query.limit is not None and _resamples_or_aggregates(query):
        return query.model_copy(update={"limit": None}), query.limit
    return query, None


# Hosted export cannot apply limit itself (the gateway streams the file), and
# sending it would apply it before resampling/aggregation (OCE-319).
_HOSTED_LIMIT_UNSUPPORTED = (
    "limit with time_resolution/aggregate_operations is not supported for "
    "hosted export until the Datamesh engine fix ships: Datamesh would apply "
    "limit to the native records before resampling/aggregating, so the file "
    "would be wrong. To proceed, drop limit or narrow the time range instead."
)


def _record_count(data: Any) -> int | None:
    """Records in a result, counted as summarize_data counts them."""
    if isinstance(data, pd.DataFrame):
        return int(data.shape[0])
    if isinstance(data, xr.Dataset):
        return math.prod(int(n) for n in data.sizes.values())
    return None


def _limit_result(data: Any, limit: int, coordkeys: dict[str, str]) -> tuple[Any, str]:
    """Apply Datamesh `limit` semantics (keep the last N) to a returned result.

    Used when limit cannot be sent to Datamesh because it would be applied to
    native records before resampling/aggregation (OCE-298). Returns the
    limited data and a note describing what was kept.
    """
    lead = (
        f"limit={limit} was applied by this server AFTER time_resolution "
        "resampling / aggregation (it is not sent to Datamesh with those, "
        "which would apply it to the native records first)"
    )
    if isinstance(data, pd.DataFrame):
        return data.tail(limit), f"{lead}: kept the last {limit} rows as returned."
    if isinstance(data, xr.Dataset):
        dims = []
        for key in _LIMIT_COORD_KEYS:
            name = coordkeys.get(key)
            if name in data.coords and data.coords[name].dims:
                dims.append(data.coords[name].dims[0])
            elif name in data.dims:
                # A dimension without a coordinate variable (OCE-326), e.g.
                # an ensemble member index.
                dims.append(name)
        if not dims:
            # Stage without coordinate keys: fall back to datetime dims.
            dims = [
                d
                for d in data.dims
                if d in data.coords and np.issubdtype(data[d].dtype, np.datetime64)
            ]
        if dims:
            dims = list(dict.fromkeys(dims))
            limited = data.isel({d: slice(-limit, None) for d in dims})
            return limited, f"{lead}: kept the last {limit} steps along {dims}."
    return data, (
        f"{lead}; the result has no time/ensemble dimension to limit, so limit "
        "had no effect."
    )


def _stage_summary(stage: Stage) -> dict[str, Any]:
    return {
        "container": stage.container.value,
        "size_bytes": stage.size,
        "size_human": human_bytes(stage.size),
        "domain_length": stage.dlen,
    }


def _refusal(stage: Stage, message: str, **extra: Any) -> str:
    return to_json(
        {"refused": True, **_stage_summary(stage), "message": message, **extra}
    )


def _resolve_export_path(path: str) -> Path:
    """Resolve an export destination, confined to OCEANUM_MCP_EXPORT_DIR if set."""
    dest = Path(path).expanduser()
    root = export_dir()
    if root is None:
        return dest
    dest = (dest if dest.is_absolute() else root / dest).resolve()
    if not dest.is_relative_to(root):
        raise ToolError(
            f"path must be inside OCEANUM_MCP_EXPORT_DIR ({root}) on this server."
        )
    return dest


def _unset_sentinel(value: Any) -> Any:
    """None for the literal "null" or "" that some MCP clients send in place
    of an omitted optional param.

    Only str-typed params can carry these into a tool: schema validation
    rejects them for typed params. Never apply to required params.
    """
    return None if value in ("null", "") else value


# Offset types whose meaning survives the engine's lower-casing of
# time_resolution on every pandas version (pandas 2 upper-cases unknown
# aliases back, pandas 3 does not): fixed durations and weeks. Calendar
# month/quarter/year and business offsets are excluded.
_ENGINE_SAFE_OFFSETS = (pd.offsets.Tick, pd.offsets.Day, pd.offsets.Week)

# Resampling finer than this estimates absurd bin counts over any real range.
_MIN_TIME_RESOLUTION = pd.Timedelta(minutes=1)


@contextmanager
def _quiet_pandas_aliases() -> Iterator[None]:
    """Silence pandas' alias-deprecation FutureWarnings (e.g. "'d' is
    deprecated" on pandas 3) while parsing a time_resolution.

    catch_warnings mutates process-global state, so it is only used when
    _WARNINGS_LOCK is free. The lock is held by _captured_warnings for a
    whole query, and validating a parameter must not wait behind one; when
    it is busy the parse runs unsuppressed (a stray deprecation message is
    harmless).
    """
    if not _WARNINGS_LOCK.acquire(blocking=False):
        yield
        return
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            yield
    finally:
        _WARNINGS_LOCK.release()


# Legacy pandas aliases, matched on the alias tokens of a time_resolution
# independently of the local pandas version (OCE-294 gate). pandas 3 removed
# them, so they no longer parse here and would otherwise be passed through;
# the engine lower-cases them and, on its pandas, may still parse them, e.g.
# "1L" -> "1l" = one millisecond, the same hang as "1MS".
# Sub-minute units, compared lower-cased: L/ms, U/us, N/ns, S/s.
_SUB_MINUTE_ALIASES = frozenset({"l", "ms", "u", "us", "n", "ns", "s"})
# Calendar month/quarter/year aliases, incl. legacy A/AS/Y/M/Q, anchored
# forms (the anchor is a separate token), and business (B, CB) and
# semi-month variants. Compared upper-cased, except that "ms" written in
# lower case is milliseconds (handled above), not month start.
_CALENDAR_ALIAS_RE = re.compile(r"(C?B)?[MQYA][SE]?|SM[SE]?")
_ALIAS_TOKEN_RE = re.compile(r"[A-Za-z]+")


def _resolution_error(
    resolution: str, reason: Literal["calendar", "fine"]
) -> ToolError:
    if reason == "calendar":
        return ToolError(
            f"time_resolution {resolution!r} is not supported yet: Datamesh "
            "currently lower-cases time_resolution, which changes the meaning "
            'of calendar aliases (e.g. "1MS", month start, becomes "1ms", one '
            "millisecond, and the query hangs). Monthly, quarterly and yearly "
            "aliases (MS, ME, QS, YS, M, Q, Y, A, ...) are not supported until "
            'that engine fix ships: use a fixed length instead, e.g. "30D" '
            '(about monthly), "90D" (about quarterly) or "365D" (about yearly).'
        )
    return ToolError(
        f"time_resolution {resolution!r} is finer than one minute, which "
        "makes an enormous number of time bins over any real range. Use "
        '"1min" or coarser (e.g. "1h", "1D"), or omit time_resolution to '
        "get the data at its native resolution."
    )


def _denied_alias(resolution: str) -> Literal["calendar", "fine"] | None:
    """Why a time_resolution uses a denylisted alias, or None.

    Pandas-version independent: looks only at the alias tokens (the letters
    after the multiplier; an anchor such as "-DEC" is a separate token).
    """
    for token in _ALIAS_TOKEN_RE.findall(resolution):
        if token != "ms" and _CALENDAR_ALIAS_RE.fullmatch(token.upper()):
            return "calendar"
        if token.lower() in _SUB_MINUTE_ALIASES:
            return "fine"
    return None


def _check_time_resolution(resolution: str) -> None:
    """Reject time_resolution values the current Datamesh engine mishandles.

    OCE-294: the engine lower-cases time_resolution before parsing it, so a
    case-sensitive alias silently changes meaning: "1MS" (month start)
    becomes "1ms" (one millisecond), and a year of data at 1 ms hangs the
    stage. Other calendar aliases (ME, QS, YS, BMS, ...) work or fail
    depending on the engine's pandas version. Until the engine fix (OCE-324)
    is deployed, only offsets whose meaning is case-insensitive are allowed,
    and none finer than a minute. Relax this guard once OCE-324 ships.

    Legacy aliases are denylisted first (_denied_alias), so the result does
    not depend on whether this host's pandas still parses them. Any other
    value pandas cannot parse here is left for the engine to validate.
    """
    resolution = resolution.strip()
    if resolution.lower() == "native":
        return
    denied = _denied_alias(resolution)
    if denied is not None:
        raise _resolution_error(resolution, denied)
    with _quiet_pandas_aliases():
        try:
            offset = to_offset(resolution)
        except ValueError:
            offset = None
        try:
            engine_offset = to_offset(resolution.lower())
        except ValueError:
            engine_offset = None
    if offset is None and engine_offset is None:
        return
    if engine_offset is not None and engine_offset.n <= 0:
        raise ToolError(
            f"time_resolution {resolution!r} must be a positive duration, "
            'e.g. "1h", "1D" or "30D".'
        )
    if (offset is not None and offset != engine_offset) or not isinstance(
        engine_offset, _ENGINE_SAFE_OFFSETS
    ):
        raise _resolution_error(resolution, "calendar")
    if (
        isinstance(engine_offset, pd.offsets.Tick)
        and pd.Timedelta(engine_offset) < _MIN_TIME_RESOLUTION
    ):
        raise _resolution_error(resolution, "fine")


def _build_query(
    datasource_id: str,
    *,
    variables: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    times: list[str] | None = None,
    time_resolution: str | None = None,
    time_resample: Literal["mean", "nearest", "linear"] | None = None,
    bbox: list[float] | None = None,
    geofilter_feature: dict[str, Any] | None = None,
    geofilter_interp: Literal["nearest", "linear"] | None = None,
    geofilter_resolution: float | None = None,
    level_min: float | None = None,
    level_max: float | None = None,
    levels: list[float] | None = None,
    level_interp: Literal["nearest", "linear"] | None = None,
    coord_filters: list[CoordSelector] | None = None,
    crs: str | int | None = None,
    aggregate_operations: list[Literal["mean", "min", "max", "std", "sum"]]
    | None = None,
    aggregate_spatial: bool = True,
    aggregate_temporal: bool = True,
    limit: int | None = None,
) -> Query:
    """Build a validated Datamesh Query from flat tool parameters."""
    time_start = _unset_sentinel(time_start)
    time_end = _unset_sentinel(time_end)
    time_resolution = _unset_sentinel(time_resolution)
    if isinstance(time_resolution, str):
        # Surrounding whitespace never belongs in a pandas frequency string.
        time_resolution = time_resolution.strip() or None
    crs = _unset_sentinel(crs)
    q: dict[str, Any] = {"datasource": datasource_id}

    if variables:
        q["variables"] = variables

    if times and (time_start or time_end):
        raise ToolError(
            "Provide either times (series selection) or time_start/time_end "
            "(range selection), not both."
        )
    if (time_resolution or time_resample) and (times or not (time_start or time_end)):
        raise ToolError(
            "time_resolution/time_resample apply only to a time_start/time_end range."
        )
    if times:
        q["timefilter"] = {"type": "series", "times": times}
    elif time_start or time_end:
        timefilter: dict[str, Any] = {
            "type": "range",
            "times": [time_start, time_end],
        }
        if time_resolution:
            _check_time_resolution(time_resolution)
            timefilter["resolution"] = time_resolution
        if time_resample:
            timefilter["resample"] = time_resample
        q["timefilter"] = timefilter

    if bbox and geofilter_feature:
        raise ToolError("Provide either bbox or geofilter_feature, not both.")
    if bbox or geofilter_feature:
        geofilter: dict[str, Any] = (
            {"type": "bbox", "geom": bbox}
            if bbox
            else {"type": "feature", "geom": geofilter_feature}
        )
        if geofilter_interp:
            geofilter["interp"] = geofilter_interp
        if geofilter_resolution is not None:
            geofilter["resolution"] = geofilter_resolution
        q["geofilter"] = geofilter

    if levels and (level_min is not None or level_max is not None):
        raise ToolError(
            "Provide either levels (series selection) or level_min/level_max "
            "(range selection), not both."
        )
    if levels or level_min is not None or level_max is not None:
        levelfilter: dict[str, Any] = (
            {"type": "series", "levels": levels}
            if levels
            else {"type": "range", "levels": [level_min, level_max]}
        )
        if level_interp:
            levelfilter["interp"] = level_interp
        q["levelfilter"] = levelfilter

    if coord_filters:
        q["coordfilter"] = coord_filters
    if crs is not None:
        q["crs"] = crs
    if aggregate_operations:
        q["aggregate"] = {
            "operations": aggregate_operations,
            "spatial": aggregate_spatial,
            "temporal": aggregate_temporal,
        }
    if limit is not None:
        if limit < 1:
            raise ToolError("limit must be at least 1.")
        q["limit"] = limit

    try:
        return Query(**q)
    except (ValueError, TypeError) as exc:
        raise ToolError(f"Invalid query parameters: {exc}") from exc


# Shared Args documentation for the three query-shaped tools. Assembled into
# each tool's __doc__ before registration so the MCP parameter descriptions
# stay byte-identical across tools (FastMCP parses the docstring Args section).
_QUERY_PARAM_DOCS = """\
        datasource_id: The datasource to query.
        variables: List of variable names to select (e.g. ["temperature", "salinity"]).
        time_start: ISO 8601 start of a time range (e.g. "2023-01-01T00:00:00Z"). Open-ended if omitted.
        time_end: ISO 8601 end of a time range. Open-ended if omitted.
        times: Discrete times to select (series selection). Mutually exclusive with time_start/time_end.
        time_resolution: Downsample a time range server-side to this resolution (pandas frequency string, e.g. "1h", "6h", "1D", "7D", or "30D" for roughly monthly). Drastically shrinks long time series. Monthly/quarterly/yearly calendar aliases (MS, ME, QS, YS) are not supported until a Datamesh engine fix ships: use "30D", "90D" or "365D". Finer than 1 minute is refused.
        time_resample: Resampling method when time_resolution is set: mean, nearest, or linear.
        bbox: Bounding box [xmin, ymin, xmax, ymax] in WGS84 (or crs units if crs is set).
        geofilter_feature: GeoJSON Feature object (Point, MultiPoint, or Polygon geometry) for spatial selection/interpolation. Mutually exclusive with bbox.
        geofilter_interp: Interpolation for feature selection: nearest or linear (default linear).
        geofilter_resolution: Maximum spatial resolution for downsampling, in CRS units.
        level_min: Minimum vertical level of a range.
        level_max: Maximum vertical level of a range.
        levels: Discrete vertical levels to select (series selection). Mutually exclusive with level_min/level_max.
        level_interp: Interpolation for level series selection: nearest or linear.
        coord_filters: Additional coordinate selections, e.g. [{"coord": "station", "values": ["A1", "B2"]}].
        crs: CRS for filter coordinates and returned data (EPSG code or CRS string).
        aggregate_operations: Aggregations to apply after filtering: mean, min, max, std, sum.
        aggregate_spatial: Aggregate over spatial dimensions (default true).
        aggregate_temporal: Aggregate over the temporal dimension (default true).
        limit: Keep only the last N records (Datamesh semantics: the last N steps along time/ensemble). Combined with time_resolution or aggregate_operations it is applied AFTER resampling/aggregation (by this server, not sent to Datamesh), so stage_query sizes the unlimited result as an upper bound; hosted export_query refuses that combination."""


# ---------------------------------------------------------------------------
# Catalog & Discovery
# ---------------------------------------------------------------------------


# Ceiling on the serialized results of one search_catalog call, in characters
# (~4 per token). Matching datasources beyond it are dropped with a note to
# refine the search. Live summaries run ~0.7 kB each (~1.2 kB with 20
# variable names), so the default limit of 20 always fits; full records of
# live catalog hits run ~2 kB, but a record carrying a schema can reach
# hundreds of kB, in which case it is returned alone.
SEARCH_BUDGET_CHARS = {"summary": 30_000, "full": 100_000}

_SEARCH_HINT = (
    "These are summaries. Call get_datasource_info(datasource_id) for one "
    "datasource's variables (units, long names, dims), coordinates and "
    "attributes."
)


def _within_budget(results: list[dict[str, Any]], budget: int) -> int:
    """How many leading results fit in budget characters (always at least one).

    Sizes are measured as each result serializes inside the response's
    "results" array: every line indented 4 more spaces, plus the separator
    and the array's closing line.
    """
    used = 2
    for n, result in enumerate(results):
        used += len(to_json(result).replace("\n", "\n    ")) + 6
        if n and used > budget:
            return n
    return len(results)


@mcp.tool(annotations=READ_TOOL)
def search_catalog(
    search: str | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    bbox: list[float] | None = None,
    limit: int = 20,
    detail: Literal["summary", "full"] = "summary",
) -> str:
    """Search the Oceanum Datamesh catalog for datasets.

    Args:
        search: Text search string to filter datasources by name, description, or tags.
        time_start: ISO 8601 datetime for start of time range filter (e.g. "2023-01-01").
        time_end: ISO 8601 datetime for end of time range filter (e.g. "2023-12-31").
        bbox: Bounding box as [xmin, ymin, xmax, ymax] in WGS84 coordinates.
        limit: Maximum number of datasources to return (default 20, minimum 1).
        detail: "summary" (default) returns id, name, a short description, time range, bounds, and variable names when known. "full" returns each datasource's complete catalog record; prefer get_datasource_info for one datasource's details.

    Returns:
        JSON with count and matching datasources. If count equals limit, more
        results may exist. The total output is bounded: matches beyond the
        bound are dropped and counted in "omitted", with a note to refine the
        search.
    """
    if limit < 1:
        raise ToolError("limit must be at least 1.")
    search = _unset_sentinel(search)
    time_start = _unset_sentinel(time_start)
    time_end = _unset_sentinel(time_end)

    conn = get_datamesh_connector()

    timefilter = None
    if time_start or time_end:
        timefilter = [time_start, time_end]

    geofilter = None
    if bbox:
        geofilter = GeoFilter(type="bbox", geom=bbox)

    catalog = conn.get_catalog(
        search=search,
        timefilter=timefilter,
        geofilter=geofilter,
        limit=limit,
    )

    formatter = format_datasource if detail == "full" else format_datasource_summary
    results = [formatter(ds) for ds in catalog if ds is not None]
    shown = _within_budget(results, SEARCH_BUDGET_CHARS[detail])
    omitted = len(results) - shown
    out: dict[str, Any] = {"count": shown, "results": results[:shown]}
    if not results:
        out["message"] = "No datasources found matching the search criteria."
    elif omitted:
        out["omitted"] = omitted
        out["note"] = (
            f"{omitted} more results matched but are not shown, to keep this "
            "response small"
            + (
                f" (and the {limit}-result limit was reached, so further "
                "matches may exist)"
                if len(results) >= limit
                else ""
            )
            + "; refine the search (more specific search text, a time "
            "range, or a bbox) to see them."
        )
    elif len(results) >= limit:
        out["note"] = (
            f"Result count equals the limit ({limit}); more matches may exist. "
            "Raise limit or refine the search."
        )
    if results and detail == "summary":
        out["hint"] = _SEARCH_HINT
    return to_json(out)


@mcp.tool(annotations=READ_TOOL)
def get_datasource_info(
    datasource_id: str, detail: Literal["summary", "full"] = "summary"
) -> str:
    """Get the metadata for one datasource: coverage, coordinates and variables.

    The default view keeps every field of the full record and, per variable
    and coordinate, its dims, shape, dtype, units, long_name and
    standard_name; it caps the other attributes, clips long attribute values,
    and on very large records keeps only those core fields per variable.

    Args:
        datasource_id: The unique ID of the datasource.
        detail: "summary" (default) returns the bounded view described above. "full" returns the complete record, including every attribute; it can be very large.

    Returns:
        Datasource metadata as JSON. On failure, JSON with an "error" (and
        the datasource_id); an unknown id adds "suggestions", close matches
        as [{id, name}].
    """
    conn = get_datamesh_connector()
    try:
        ds = conn.get_datasource(datasource_id)
    except _DATAMESH_ERRORS as exc:
        return _error_json(
            conn,
            datasource_id,
            exc,
            {"datasource_id": datasource_id},
            from_metadata=True,
        )
    if detail == "full":
        return to_json(format_datasource(ds))
    return to_json(format_datasource_bounded(ds))


# ---------------------------------------------------------------------------
# Data Access
# ---------------------------------------------------------------------------


def stage_query(
    datasource_id: str,
    variables: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    times: list[str] | None = None,
    time_resolution: str | None = None,
    time_resample: Literal["mean", "nearest", "linear"] | None = None,
    bbox: list[float] | None = None,
    geofilter_feature: dict[str, Any] | None = None,
    geofilter_interp: Literal["nearest", "linear"] | None = None,
    geofilter_resolution: float | None = None,
    level_min: float | None = None,
    level_max: float | None = None,
    levels: list[float] | None = None,
    level_interp: Literal["nearest", "linear"] | None = None,
    coord_filters: list[CoordSelector] | None = None,
    crs: str | int | None = None,
    aggregate_operations: list[Literal["mean", "min", "max", "std", "sum"]]
    | None = None,
    aggregate_spatial: bool = True,
    aggregate_temporal: bool = True,
    limit: int | None = None,
) -> str:
    """Dry-run a query: report the result size WITHOUT downloading any data.

    Always stage before retrieving data you have not sized. The response says
    whether the result is small enough for query_data to return inline, and
    echoes the canonical query. With limit plus time_resolution or
    aggregate_operations the reported size is that of the unlimited result
    (limit is applied after resampling), so it is an upper bound.

    A variable given as the exact CF standard_name of exactly one of the
    datasource's variables (e.g. "sea_surface_wave_significant_height" for
    "hs") is resolved to that variable; the response then includes
    resolved_variables ({requested: used}) and echoes the resolved names.
    """
    conn = get_datamesh_connector()
    query = _build_query(
        datasource_id,
        variables=variables,
        time_start=time_start,
        time_end=time_end,
        times=times,
        time_resolution=time_resolution,
        time_resample=time_resample,
        bbox=bbox,
        geofilter_feature=geofilter_feature,
        geofilter_interp=geofilter_interp,
        geofilter_resolution=geofilter_resolution,
        level_min=level_min,
        level_max=level_max,
        levels=levels,
        level_interp=level_interp,
        coord_filters=coord_filters,
        crs=crs,
        aggregate_operations=aggregate_operations,
        aggregate_spatial=aggregate_spatial,
        aggregate_temporal=aggregate_temporal,
        limit=limit,
    )
    sent, client_limit = _split_limit(query)
    try:
        stage, aliases = _stage_resolving(conn, sent, _stage)
        sent, query = _substituted(sent, aliases), _substituted(query, aliases)
    except _DATAMESH_ERRORS as exc:
        return _error_json(
            conn,
            query.datasource,
            exc,
            {"query": _query_echo(query)},
            variables=query.variables,
        )

    if stage is None:
        return to_json(
            {
                "staged": False,
                "message": "No data matches this query.",
                "query": _query_echo(query),
            }
        )

    out: dict[str, Any] = {"staged": True, **_stage_summary(stage)}
    inline_limit = max_inline_bytes()
    if stage.size <= inline_limit:
        out["recommendation"] = (
            "Small enough to return inline: call query_data with these parameters."
        )
    elif stage.size > LARGE_EXPORT_BYTES:
        size = human_bytes(stage.size)
        local_cap = (
            MAX_EXPORT_DATASET_BYTES
            if stage.container == Container.Dataset
            else MAX_EXPORT_FRAME_BYTES
        )
        if is_network_transport():
            if client_limit is not None:
                cost = (
                    "export_query will refuse it with limit plus "
                    "time_resolution/aggregate_operations (see limit_note)."
                )
            elif stage.size > MAX_HOSTED_EXPORT_BYTES:
                cost = (
                    "export_query will refuse it: hosted downloads are capped "
                    f"at {human_bytes(MAX_HOSTED_EXPORT_BYTES)}."
                )
            else:
                cost = (
                    f"Exporting as-is makes export_query return a {size} "
                    "download; ask the user before exporting."
                )
        elif client_limit is not None and stage.container == Container.Dataset:
            # OCE-326: export_query sizes a limited dataset on the sliced
            # result, not on this unlimited stage, so it may well export it.
            cost = (
                f"With limit={client_limit}, export_query keeps only the last "
                f"{client_limit} time/ensemble steps of the resampled or "
                "aggregated result and applies the local export cap "
                f"({human_bytes(local_cap)}) and the disk-space check to that "
                f"slice, which can be far smaller than {size} (if the result "
                "has no time/ensemble dimension, limit cannot shrink it); ask "
                "the user before exporting if it is still large."
            )
        elif stage.size > local_cap:
            cost = (
                f"export_query will refuse it: local exports are capped at "
                f"{human_bytes(local_cap)}."
            )
        else:
            cost = (
                f"Exporting as-is makes export_query write about {size} to the "
                "user's disk (several times more as CSV); ask the user before "
                "exporting."
            )
        out["recommendation"] = (
            f"Large result ({size}). Narrow it first: a shorter time range, a "
            "smaller bbox, fewer variables, time_resolution downsampling, or "
            f"aggregation. {cost}"
        )
    else:
        detail = (
            "query_data will return only a lazy structure summary; shrink the "
            "result with filters, aggregation, or time_resolution"
            if stage.container == Container.Dataset
            else "narrow the query with filters or aggregation"
        )
        out["recommendation"] = (
            f"Larger than the inline limit ({human_bytes(inline_limit)}): "
            f"{detail}{export_clause()}."
        )
    if (
        stage.container in (Container.DataFrame, Container.GeoDataFrame)
        and stage.dlen >= DATAMESH_ROW_CAP
    ):
        out["warnings"] = [
            f"Datamesh caps tabular results at {DATAMESH_ROW_CAP} rows; this "
            "result would be truncated. Narrow the query."
        ]
    if client_limit is not None:
        limit_note = (
            f"limit={client_limit} is applied AFTER time_resolution resampling / "
            "aggregation (by this server, not sent to Datamesh), so size_bytes "
            "is the unlimited result: an upper bound on what is returned."
        )
        if is_network_transport():
            limit_note += (
                " export_query refuses this combination on this hosted server "
                "(not supported until the Datamesh engine fix ships): drop "
                "limit or narrow the time range to export."
            )
        out["limit_note"] = limit_note
    out["query"] = _query_echo(query)
    return to_json(out)


def query_data(
    datasource_id: str,
    variables: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    times: list[str] | None = None,
    time_resolution: str | None = None,
    time_resample: Literal["mean", "nearest", "linear"] | None = None,
    bbox: list[float] | None = None,
    geofilter_feature: dict[str, Any] | None = None,
    geofilter_interp: Literal["nearest", "linear"] | None = None,
    geofilter_resolution: float | None = None,
    level_min: float | None = None,
    level_max: float | None = None,
    levels: list[float] | None = None,
    level_interp: Literal["nearest", "linear"] | None = None,
    coord_filters: list[CoordSelector] | None = None,
    crs: str | int | None = None,
    aggregate_operations: list[Literal["mean", "min", "max", "std", "sum"]]
    | None = None,
    aggregate_spatial: bool = True,
    aggregate_temporal: bool = True,
    limit: int | None = None,
) -> str:
    """Query a datasource and return small results inline.

    The query is staged first; results larger than the inline limit are not
    downloaded (datasets are summarized lazily, tabular queries are refused
    with alternatives). Use stage_query to size a query before calling this.

    Inline values are a preview when `preview` is true (`returned` of `total`
    records): never compute statistics from a preview; use
    aggregate_operations or time_resolution instead.

    limit keeps the last N records (Datamesh semantics: the last N steps along
    time/ensemble). Combined with time_resolution or aggregate_operations,
    limit is applied by this server AFTER resampling/aggregation (to the last
    N resampled steps, or the last N rows of a table) rather than sent to
    Datamesh. A limited result is a subset of the requested range, so it is
    flagged as a preview, with a limit_note.

    A variable given as the exact CF standard_name of exactly one of the
    datasource's variables (e.g. "sea_surface_wave_significant_height" for
    "hs") is resolved to that variable; the response then includes
    resolved_variables ({requested: used}) and echoes the resolved names.
    """
    conn = get_datamesh_connector()
    query = _build_query(
        datasource_id,
        variables=variables,
        time_start=time_start,
        time_end=time_end,
        times=times,
        time_resolution=time_resolution,
        time_resample=time_resample,
        bbox=bbox,
        geofilter_feature=geofilter_feature,
        geofilter_interp=geofilter_interp,
        geofilter_resolution=geofilter_resolution,
        level_min=level_min,
        level_max=level_max,
        levels=levels,
        level_interp=level_interp,
        coord_filters=coord_filters,
        crs=crs,
        aggregate_operations=aggregate_operations,
        aggregate_spatial=aggregate_spatial,
        aggregate_temporal=aggregate_temporal,
        limit=limit,
    )
    sent, client_limit = _split_limit(query)
    warnings: list[str] = []
    try:
        stage, aliases = _stage_resolving(conn, sent, _stage)
        sent, query = _substituted(sent, aliases), _substituted(query, aliases)
        if stage is None:
            return to_json(
                {
                    "status": "no_data",
                    "message": "No data matches this query.",
                    "query": _query_echo(query),
                }
            )
        inline_limit = max_inline_bytes()
        use_dask = False
        if stage.size > inline_limit:
            if stage.container == Container.Dataset:
                # Lazy zarr access: structure only, no data download.
                use_dask = True
            else:
                extra: dict[str, Any] = {}
                if client_limit is not None:
                    extra["limit_note"] = (
                        f"limit={client_limit} cannot shrink this download: with "
                        "time_resolution/aggregate_operations it is applied "
                        "after the full result is fetched."
                    )
                return _refusal(
                    stage,
                    f"Result is {human_bytes(stage.size)}, above the inline "
                    f"limit of {human_bytes(inline_limit)}. Narrow the query "
                    f"with filters or aggregation{export_clause()}.",
                    query=_query_echo(query),
                    **extra,
                )
        # Inline results are small (or lazy), so the whole query, including
        # the SDK's internal re-stage, is held to the stage timeout.
        with _gateway_timeout(conn, "query"), _captured_warnings(warnings):
            data = conn.query(sent, use_dask=use_dask)
            full_records = _record_count(data)
            limit_note = None
            if client_limit is not None and data is not None:
                data, limit_note = _limit_result(data, client_limit, stage.coordkeys)
                if use_dask and data.nbytes <= inline_limit:
                    # The limited slice is small: fetch just its chunks so
                    # it comes back inline rather than as a lazy summary.
                    data = data.load()
    except _DATAMESH_ERRORS as exc:
        return _error_json(
            conn,
            query.datasource,
            exc,
            {"query": _query_echo(query)},
            variables=query.variables,
        )

    out = summarize_data(data, warnings=warnings)
    out["staged_size_bytes"] = stage.size
    if query.limit is not None and "preview" in out:
        # A limited result is a subset of what the query selects, so it is a
        # preview too: never let it pass as the full range (OCE-306).
        if client_limit is None:
            # Datamesh applied it; the unlimited size is unknown.
            out["preview"], out["total"] = True, None
            limit_note = (
                f"limit={query.limit} was applied by Datamesh (it keeps the "
                "last N steps); the full result may be larger."
            )
        elif full_records is not None and _record_count(data) < full_records:
            out["preview"], out["total"] = True, full_records
        if out["preview"]:
            limit_note += (
                " This limited result is a subset of the requested range: NOT "
                "suitable for statistics over that range; use "
                "aggregate_operations or time_resolution instead."
            )
    if limit_note:
        out["limit_note"] = limit_note
    return to_json(out)


def _export_download_url(
    conn: Connector,
    query: Query,
    format: Literal["netcdf", "parquet", "csv"] | None,
    requested_path: str | None = None,
) -> str:
    """Hosted export: return a signed gateway download URL, no local file.

    The gateway stages the result and returns a self-authenticating URL; the
    data never touches this server's disk or the conversation. The caller (or
    their code) fetches the URL out-of-band.
    """
    try:
        stage, aliases = _stage_resolving(conn, query, _download_stage)
        query = _substituted(query, aliases)
    except _DATAMESH_ERRORS as exc:
        return _error_json(
            conn,
            query.datasource,
            exc,
            {"query": _query_echo(query)},
            variables=query.variables,
        )
    if stage is None or not stage.get("url"):
        return to_json(
            {
                "status": "no_data",
                "message": "No data matches this query; nothing to download.",
                "query": _query_echo(query),
            }
        )

    # The gateway renders any format it advertises regardless of container
    # (a point-extracted dataset serves CSV fine), so validate against the
    # advertised list rather than the local path's xarray/pandas constraints.
    is_dataset = stage.get("container") == Container.Dataset.value
    fmt = format or ("netcdf" if is_dataset else "parquet")
    gateway_fmt = _GATEWAY_FORMAT[fmt]
    available = stage.get("formats") or []
    if available and gateway_fmt not in available:
        raise ToolError(
            f"Format {fmt!r} is not available for this query; the gateway "
            f"offers: {', '.join(available)}."
        )

    try:
        size = int(stage["size"])
    except (KeyError, TypeError, ValueError):
        # Fail closed: without a size the hosted cap cannot be enforced.
        return to_json(
            {
                "error": "The gateway did not report the download size, so "
                "the hosted export limit cannot be checked; no link returned.",
                "query": _query_echo(query),
            }
        )
    # size is the staged (in-memory) size, which NetCDF/Parquet land at or
    # below; CSV text runs well above it, so the cap applies to its estimate.
    est = int(size * _EXPORT_DISK_FACTOR["csv"]) if fmt == "csv" else size
    if est > MAX_HOSTED_EXPORT_BYTES:
        # Same shape as _refusal; the gateway's download stage has no dlen.
        # The signed URL is withheld: it is a bearer credential (OCE-299).
        as_fmt = f" (about {human_bytes(est)} as {fmt})" if fmt == "csv" else ""
        return to_json(
            {
                "refused": True,
                "container": stage.get("container"),
                "size_bytes": size,
                "size_human": human_bytes(size),
                "message": f"Result is {human_bytes(size)}{as_fmt}, above the "
                f"hosted export limit of {human_bytes(MAX_HOSTED_EXPORT_BYTES)}. "
                "Narrow the query: a shorter time range, a smaller bbox, fewer "
                "variables, time_resolution downsampling, or aggregation.",
                "query": _query_echo(query),
            }
        )
    # The signed URL is a capability: append the format token, keeping the
    # signature intact. Use '?' if the URL has no query string yet, else '&'.
    url = stage["url"]
    sep = "&" if "?" in url else "?"
    download_url = f"{url}{sep}f={gateway_fmt}"
    note = (
        "Time-limited, self-authenticating download link (it needs no "
        "token, so treat it like a password). Fetch it out-of-band; the "
        "data is not returned inline."
    )
    if requested_path is not None:
        note += " (The path argument is ignored on hosted servers.)"
    out: dict[str, Any] = {
        "download_url": download_url,
        "format": fmt,
        "container": stage.get("container"),
        "size_bytes": size,
        "size_human": human_bytes(size),
        "available_formats": stage.get("formats", []),
        "note": note,
        "query": _query_echo(query),
    }
    if size > LARGE_DOWNLOAD_BYTES:
        out["warning"] = (
            f"Large download ({human_bytes(size)}); ensure the consumer can "
            "stream it. Narrow the query to shrink it."
        )
    return to_json(out)


def _flat_frame_bytes(ds: xr.Dataset) -> int:
    """Approximate in-memory size of ds.to_dataframe().reset_index().

    Every variable is broadcast over the union of the dataset's dims and every
    coordinate becomes a column repeated on each row, so the flat table can be
    far larger than the dataset itself (8 bytes per cell is the estimate).
    """
    rows = math.prod(ds.sizes.values())
    cols = len(ds.dims) + len(ds.data_vars) + len(set(ds.coords) - set(ds.dims))
    return rows * cols * 8


def _flatten_refusal(stage: Stage, nbytes: int, fmt: str, query: Query) -> str:
    return _refusal(
        stage,
        f"This query returns a gridded dataset of about {human_bytes(nbytes)} "
        f"as a table, too large to flatten to {fmt} (limit "
        f"{human_bytes(MAX_EXPORT_FRAME_BYTES)}). Use format='netcdf', or "
        "narrow the query (e.g. a point geofilter_feature, fewer variables, a "
        "shorter time range).",
        query=_query_echo(query),
    )


def _dataset_cap_refusal(stage: Stage, nbytes: int, query: Query) -> str:
    return _refusal(
        stage,
        f"Dataset result is {human_bytes(nbytes)}, above the local export "
        f"limit of {human_bytes(MAX_EXPORT_DATASET_BYTES)}. Narrow the query: "
        "a shorter time range, a smaller bbox, fewer variables, "
        "time_resolution downsampling, or aggregation.",
        query=_query_echo(query),
    )


def _disk_refusal(
    stage: Stage, dest: Path, fmt: str, nbytes: int, query: Query
) -> str | None:
    """Refusal JSON if dest's filesystem lacks room for nbytes as fmt.

    Checked before writing a byte: a local export cannot be cancelled once it
    starts (OCE-296). Probes the nearest existing ancestor, since the
    destination directory may not exist yet. An existing file being
    overwritten is not credited: the new file is written alongside it first.
    Returns error JSON if free space cannot be determined.
    """
    try:
        probe = dest.parent
        while not probe.exists():
            probe = probe.parent
        free = shutil.disk_usage(probe).free
    except OSError as exc:
        return to_json(
            {
                "error": f"Could not check free disk space for {dest}: {exc}",
                "query": _query_echo(query),
            }
        )
    needed = int(nbytes * _EXPORT_DISK_FACTOR[fmt])
    if needed <= free:
        return None
    return _refusal(
        stage,
        f"Not enough disk space: writing this as {fmt} needs about "
        f"{human_bytes(needed)}, but {probe} has {human_bytes(free)} free. "
        "Narrow the query or choose a path on a larger disk.",
        free_bytes=free,
        query=_query_echo(query),
    )


def export_query(
    datasource_id: str,
    path: str | None = None,
    format: Literal["netcdf", "parquet", "csv"] | None = None,
    overwrite: bool = False,
    variables: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    times: list[str] | None = None,
    time_resolution: str | None = None,
    time_resample: Literal["mean", "nearest", "linear"] | None = None,
    bbox: list[float] | None = None,
    geofilter_feature: dict[str, Any] | None = None,
    geofilter_interp: Literal["nearest", "linear"] | None = None,
    geofilter_resolution: float | None = None,
    level_min: float | None = None,
    level_max: float | None = None,
    levels: list[float] | None = None,
    level_interp: Literal["nearest", "linear"] | None = None,
    coord_filters: list[CoordSelector] | None = None,
    crs: str | int | None = None,
    aggregate_operations: list[Literal["mean", "min", "max", "std", "sum"]]
    | None = None,
    aggregate_spatial: bool = True,
    aggregate_temporal: bool = True,
    limit: int | None = None,
) -> str:
    """Export the FULL result of a query — the data never enters the conversation.

    Behavior depends on how the server is running:
    - Hosted (http/sse): returns a time-limited, self-authenticating gateway
      download_url (choose format via `format`); `path` is ignored. Fetch the
      URL out-of-band — it needs no credential, so treat it as a secret.
      Refused above the hosted export cap, and for limit combined with
      time_resolution/aggregate_operations.
    - Local (stdio): writes the result to the local file `path` (required) and
      returns that path. Gridded datasets stream to NetCDF; tabular results
      write Parquet or CSV, as do datasets small enough to flatten into a
      table (e.g. point time series). Refused up front if the result exceeds
      the local export cap or the destination's free disk space. With limit
      plus time_resolution/aggregate_operations, the last N resampled steps
      are written.

    A variable given as the exact CF standard_name of exactly one of the
    datasource's variables (e.g. "sea_surface_wave_significant_height" for
    "hs") is resolved to that variable; the response then includes
    resolved_variables ({requested: used}) and echoes the resolved names.
    """
    # Some MCP clients send "null"/"" for an omitted path (never a file name).
    path = _unset_sentinel(path)
    conn = get_datamesh_connector()
    query = _build_query(
        datasource_id,
        variables=variables,
        time_start=time_start,
        time_end=time_end,
        times=times,
        time_resolution=time_resolution,
        time_resample=time_resample,
        bbox=bbox,
        geofilter_feature=geofilter_feature,
        geofilter_interp=geofilter_interp,
        geofilter_resolution=geofilter_resolution,
        level_min=level_min,
        level_max=level_max,
        levels=levels,
        level_interp=level_interp,
        coord_filters=coord_filters,
        crs=crs,
        aggregate_operations=aggregate_operations,
        aggregate_spatial=aggregate_spatial,
        aggregate_temporal=aggregate_temporal,
        limit=limit,
    )

    if is_network_transport():
        # Hosted: broker a gateway download link; there is no client-visible
        # local filesystem. path/overwrite do not apply.
        if query.limit is not None and _resamples_or_aggregates(query):
            raise ToolError(_HOSTED_LIMIT_UNSUPPORTED)
        return _export_download_url(conn, query, format, requested_path=path)

    if path is None:
        raise ToolError("path is required for local (stdio) export.")
    dest = _resolve_export_path(path)
    if dest.exists():
        if dest.is_dir():
            raise ToolError(f"path is an existing directory: {dest}")
        if not overwrite:
            raise ToolError(f"File exists: {dest}. Pass overwrite=true to replace it.")

    sent, client_limit = _split_limit(query)
    warnings: list[str] = []
    try:
        stage, aliases = _stage_resolving(conn, sent, _stage)
        sent, query = _substituted(sent, aliases), _substituted(query, aliases)
        if stage is None:
            return to_json(
                {
                    "status": "no_data",
                    "message": "No data matches this query; nothing written.",
                    "query": _query_echo(query),
                }
            )

        is_dataset = stage.container == Container.Dataset
        # stage.size is the unlimited result. A limited dataset is sliced
        # lazily before any chunk is fetched, so its size checks run on the
        # limited slice once it is open, below. (A frame downloads in full
        # before the limit applies, so the unlimited size is the right bound.)
        size_limited_slice = is_dataset and client_limit is not None
        if is_dataset:
            fmt = format or "netcdf"
            # A point series (or any small dataset) flattens to a table via
            # to_dataframe(), which materializes it in memory: hold it to the
            # same ceiling as a tabular export.
            # Cheap pre-download bound; the flattened size is checked again
            # from the lazy structure once it is open.
            if not size_limited_slice:
                if fmt != "netcdf" and stage.size > MAX_EXPORT_FRAME_BYTES:
                    return _flatten_refusal(stage, stage.size, fmt, query)
                if stage.size > MAX_EXPORT_DATASET_BYTES:
                    return _dataset_cap_refusal(stage, stage.size, query)
        else:
            fmt = format or "parquet"
            if fmt not in ("parquet", "csv"):
                raise ToolError(
                    "This query returns tabular data; use format='parquet' or 'csv'."
                )
            if stage.size > MAX_EXPORT_FRAME_BYTES:
                return _refusal(
                    stage,
                    f"Tabular result is {human_bytes(stage.size)}, above the "
                    f"export limit of {human_bytes(MAX_EXPORT_FRAME_BYTES)}. "
                    "Narrow the query.",
                    query=_query_echo(query),
                )

        if not size_limited_slice:
            refusal = _disk_refusal(stage, dest, fmt, stage.size, query)
            if refusal is not None:
                return refusal

        # A dataset's conn.query only re-stages and opens lazy zarr, so it is
        # held to the stage timeout. A frame's downloads the whole (up to
        # MAX_EXPORT_FRAME_BYTES) result, which can legitimately take longer
        # to start streaming, so it keeps the SDK's timeouts; its re-stage
        # follows the stage above, which just succeeded within the cap.
        bound = _gateway_timeout(conn, "query") if is_dataset else nullcontext()
        with bound, _captured_warnings(warnings):
            # Datasets stream chunk-wise from lazy zarr; frames download fully.
            data = conn.query(sent, use_dask=is_dataset)
    except _DATAMESH_ERRORS as exc:
        return _error_json(
            conn,
            query.datasource,
            exc,
            {"query": _query_echo(query)},
            variables=query.variables,
        )

    if data is None:
        # The gateway can report no data on the download staging even after a
        # successful dry-run stage (data changed in between).
        return to_json(
            {
                "status": "no_data",
                "message": "No data matches this query; nothing written.",
                "query": _query_echo(query),
            }
        )

    limit_note = None
    if client_limit is not None:
        # Lazy datasets are sliced before any chunk is fetched.
        data, limit_note = _limit_result(data, client_limit, stage.coordkeys)
        if size_limited_slice:
            # The checks skipped before download, on the limited lazy slice
            # (nbytes comes from shape and dtype; nothing is fetched). The
            # flatten path below checks the flattened size and disk itself.
            nbytes = int(data.nbytes)
            if nbytes > MAX_EXPORT_DATASET_BYTES:
                return _dataset_cap_refusal(stage, nbytes, query)
            if fmt == "netcdf":
                refusal = _disk_refusal(stage, dest, fmt, nbytes, query)
                if refusal is not None:
                    return refusal

    if is_dataset and fmt != "netcdf":
        # Size the flattened table from the lazy structure (no data fetched
        # yet) before materializing it in memory.
        flat_bytes = _flat_frame_bytes(data)
        if flat_bytes > MAX_EXPORT_FRAME_BYTES:
            return _flatten_refusal(stage, flat_bytes, fmt, query)
        refusal = _disk_refusal(stage, dest, fmt, flat_bytes, query)
        if refusal is not None:
            return refusal
        try:
            with _captured_warnings(warnings):
                data = data.to_dataframe().reset_index()
        except (*_DATAMESH_ERRORS, OSError) as exc:
            return to_json({"error": str(exc), "query": _query_echo(query)})

    # Write into a private sibling directory under dest's own file name, then
    # rename into place on success: a failure never leaves a partial file at
    # dest nor destroys a file being overwritten, and pandas/xarray still see
    # the real name, from which they infer compression (out.csv.gz) and the
    # archive member name (out.csv.zip).
    tmp_dir = dest.with_name(f".{dest.name}.{uuid.uuid4().hex[:8]}.partial")
    tmp = tmp_dir / dest.name
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir()
        if fmt == "netcdf":
            data.to_netcdf(tmp)
        elif fmt == "parquet":
            data.to_parquet(tmp)
        else:
            data.to_csv(tmp, index=False)
        os.replace(tmp, dest)
    except (*_DATAMESH_ERRORS, OSError) as exc:
        return to_json(
            {
                "error": f"Export failed while writing {dest}: {exc}",
                "query": _query_echo(query),
            }
        )
    finally:
        # Any failure (zarr chunk fetch, disk, an unexpected exception,
        # KeyboardInterrupt) leaves a partial file in tmp_dir; on success it is
        # empty. Ignore removal errors so they never mask the original failure.
        shutil.rmtree(tmp_dir, ignore_errors=True)

    summary = summarize_data(data, max_rows=0, warnings=warnings)
    out: dict[str, Any] = {
        "path": str(dest),
        "format": fmt,
        "bytes_written": dest.stat().st_size,
        "summary": summary,
    }
    if limit_note:
        out["limit_note"] = limit_note
    return to_json(out)


# Assemble the shared Args docs into each query tool's docstring BEFORE
# registration — FastMCP parses __doc__ at registration time.
stage_query.__doc__ = f"""{stage_query.__doc__}
    Args:
{_QUERY_PARAM_DOCS}

    Returns:
        JSON with staged flag, container type, size_bytes, domain_length,
        the canonical query, and a recommendation for the next step.
    """
query_data.__doc__ = f"""{query_data.__doc__}
    Args:
{_QUERY_PARAM_DOCS}

    Returns:
        JSON with the result data (coordinate-attributed records) or a
        structure summary, explicit truncated/lazy flags, preview/returned/
        total record counts, staged size, any limit_note, and any server
        warnings.
    """
export_query.__doc__ = f"""{export_query.__doc__}
    Args:
        path: Local (stdio) destination file path (required on stdio; parent directories are created; confined to OCEANUM_MCP_EXPORT_DIR when set). Ignored on hosted servers, which return a download URL.
        format: Output format: netcdf (datasets), parquet or csv (tabular, or datasets small enough to flatten into a table such as point time series). Defaults by container: dataset -> netcdf, tabular -> parquet.
        overwrite: Overwrite an existing local file (default false). Local export only.
{_QUERY_PARAM_DOCS}

    Returns:
        Hosted: JSON with a signed download_url, format, size, container, and
        available_formats (or a refused object, with no URL, above the hosted
        cap). Local: JSON with the written path, format, bytes_written, a
        structure summary, and any limit_note. Neither returns inline values.
    """

# Wrapped so a response reports variables resolved by CF standard_name
# (OCE-303); functools.wraps keeps the signature and docstring FastMCP parses.
stage_query = mcp.tool(annotations=READ_TOOL)(
    _reporting_resolved_variables(stage_query)
)
query_data = mcp.tool(annotations=READ_TOOL)(_reporting_resolved_variables(query_data))
export_query = mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)(_reporting_resolved_variables(export_query))


@mcp.tool(annotations=READ_TOOL)
def load_datasource(datasource_id: str) -> str:
    """Summarize an entire datasource.

    Gridded datasources are opened lazily (no data download). Tabular
    datasources are downloaded only if under the inline size limit; use
    query_data with filters to retrieve larger ones.

    Args:
        datasource_id: The datasource to load.

    Returns:
        JSON structure summary with shape, variables, and preview values for
        small datasources.
    """
    conn = get_datamesh_connector()
    try:
        query = Query(datasource=datasource_id)
    except (ValueError, TypeError) as exc:
        raise ToolError(f"Invalid datasource_id: {exc}") from exc
    warnings: list[str] = []
    try:
        stage = _stage(conn, query)
        if stage is None:
            return to_json(
                {
                    "status": "no_data",
                    "message": "Datasource contains no data.",
                    "datasource_id": datasource_id,
                }
            )
        inline_limit = max_inline_bytes()
        if (
            stage.container in (Container.DataFrame, Container.GeoDataFrame)
            and stage.size > inline_limit
        ):
            return _refusal(
                stage,
                f"Datasource is {human_bytes(stage.size)}, above the inline "
                f"limit of {human_bytes(inline_limit)}. Use query_data with "
                f"filters{export_clause()}.",
                datasource_id=datasource_id,
            )
        bound = _gateway_timeout(
            conn,
            "load",
            hint="The datasource may be cold or slow: retrying once may "
            "succeed. If it times out again, use query_data with time/space "
            "filters to fetch a subset",
        )
        with bound, _captured_warnings(warnings):
            data = conn.load_datasource(datasource_id)
    except _DATAMESH_ERRORS as exc:
        return _error_json(conn, datasource_id, exc, {"datasource_id": datasource_id})
    return to_json(summarize_data(data, warnings=warnings))


# ---------------------------------------------------------------------------
# Data Management
# ---------------------------------------------------------------------------


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
def update_metadata(
    datasource_id: str,
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    labels: list[str] | None = None,
    info: dict[str, Any] | None = None,
    details: str | None = None,
) -> str:
    """Update metadata properties on an existing datasource.

    Only the provided fields will be updated; others remain unchanged.
    Not available when the server runs with OCEANUM_MCP_READ_ONLY set.

    Args:
        datasource_id: The datasource to update.
        name: New human-readable name (max 128 chars).
        description: New description (max 1500 chars).
        tags: New list of keyword tags.
        labels: New list of metadata labels.
        info: Additional metadata as a JSON object.
        details: URL with further details about the datasource.

    Returns:
        Updated datasource metadata.
    """
    conn = get_datamesh_connector()
    # A "null"/"" sentinel must not overwrite live metadata.
    name = _unset_sentinel(name)
    description = _unset_sentinel(description)
    details = _unset_sentinel(details)

    props: dict[str, Any] = {}
    if name is not None:
        props["name"] = name
    if description is not None:
        props["description"] = description
    if tags is not None:
        props["tags"] = tags
    if labels is not None:
        props["labels"] = labels
    if info is not None:
        props["info"] = info
    if details is not None:
        props["details"] = details

    ds = conn.update_metadata(datasource_id, **props)
    return to_json(format_datasource(ds))


if is_read_only():
    # Native visibility transform: composes through combined-server mounts
    # and keeps the tool registered (re-enable is possible at runtime).
    mcp.disable(names={"update_metadata"})

# export_query stays enabled on every transport: on stdio it writes a local
# file, on hosted it brokers a gateway download URL (both branches live in the
# tool itself), so there is no transport for which it is unavailable.
