"""Oceanum Datamesh MCP server.

Exposes the Oceanum Datamesh API as MCP tools for AI assistants.

Error conventions:
- Invalid parameter combinations raise ToolError (the caller should fix the
  tool call).
- Datamesh runtime errors return a JSON object with an "error" key and the
  canonical query echoed back, so the caller can self-correct.
- Results that are too large to return inline return a JSON object with a
  "refused" key, the staged size, and concrete alternatives.
"""

from __future__ import annotations

import threading
import warnings as _warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from oceanum.datamesh import Connector
from oceanum.datamesh.exceptions import (
    DatameshConnectError,
    DatameshQueryError,
    DatameshSessionError,
)
from oceanum.datamesh.query import Container, CoordSelector, GeoFilter, Query, Stage
from oceanum.datamesh.session import Session

from oceanum_mcp.common.client import get_datamesh_connector
from oceanum_mcp.common.config import (
    export_dir,
    is_network_transport,
    is_read_only,
    max_inline_bytes,
)
from oceanum_mcp.common.formatting import (
    format_datasource,
    human_bytes,
    summarize_data,
    to_json,
)

# Everything the oceanum library can raise on a gateway interaction.
# Session.acquire wraps all its failures (auth, network) in DatameshSessionError.
_DATAMESH_ERRORS = (DatameshConnectError, DatameshQueryError, DatameshSessionError)

# Server caps tabular (dataframe/geodataframe) query results at this many rows.
DATAMESH_ROW_CAP = 2_000_000

# Ceiling on bytes loaded into memory for a tabular export.
MAX_EXPORT_FRAME_BYTES = 2_000_000_000

READ_TOOL = {"readOnlyHint": True, "openWorldHint": True}

mcp = FastMCP(
    "Oceanum Datamesh",
    instructions=(
        "Access the Oceanum Datamesh platform for ocean and environmental data.\n"
        "Workflow: search_catalog to discover datasets -> get_datasource_info for "
        "schema and coverage -> stage_query to learn the result size WITHOUT "
        "downloading -> query_data for small results inline, or export_query to "
        "write large results to a file for code to consume.\n"
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


def _stage(conn: Connector, query: Query) -> Stage | None:
    """Stage a query on the Datamesh gateway without downloading data.

    Uses the connector's private staging request — oceanum<2 has no public
    staging API; the dependency pin in pyproject.toml guards this. Once
    oceanum grows a public Connector.stage() (and a way to execute a query
    from an existing stage), switch to it: that also removes the second
    staging round-trip conn.query() currently performs internally.
    """
    session = Session.acquire(conn)
    try:
        return conn._stage_request(query, session)
    except AttributeError as exc:  # private API drift within the 1.x pin
        raise ToolError(
            "The installed oceanum version no longer exposes the staging "
            "internals this server relies on; install a version matching "
            "the pyproject.toml pin."
        ) from exc
    finally:
        session.close()


def _query_echo(query: Query) -> dict[str, Any]:
    """Canonical JSON form of a query, for echoing in responses."""
    return query.model_dump(mode="json", exclude_none=True, warnings=False)


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
            "time_resolution/time_resample apply only to a time_start/time_end "
            "range."
        )
    if times:
        q["timefilter"] = {"type": "series", "times": times}
    elif time_start or time_end:
        timefilter: dict[str, Any] = {
            "type": "range",
            "times": [time_start, time_end],
        }
        if time_resolution:
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
        time_resolution: Downsample a time range server-side to this resolution (pandas frequency string, e.g. "1D", "1MS"). Drastically shrinks long time series.
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
        limit: Maximum number of rows/records to return."""


# ---------------------------------------------------------------------------
# Catalog & Discovery
# ---------------------------------------------------------------------------


@mcp.tool(annotations=READ_TOOL)
def search_catalog(
    search: str | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    bbox: list[float] | None = None,
    limit: int = 20,
) -> str:
    """Search the Oceanum Datamesh catalog for datasets.

    Args:
        search: Text search string to filter datasources by name, description, or tags.
        time_start: ISO 8601 datetime for start of time range filter (e.g. "2023-01-01").
        time_end: ISO 8601 datetime for end of time range filter (e.g. "2023-12-31").
        bbox: Bounding box as [xmin, ymin, xmax, ymax] in WGS84 coordinates.
        limit: Maximum number of datasources to return (default 20, minimum 1).

    Returns:
        JSON with count and matching datasources (id, name, description, time
        range, bounds). If count equals limit, more results may exist.
    """
    if limit < 1:
        raise ToolError("limit must be at least 1.")

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

    results = [format_datasource(ds) for ds in catalog if ds is not None]
    out: dict[str, Any] = {"count": len(results), "results": results}
    if not results:
        out["message"] = "No datasources found matching the search criteria."
    elif len(results) >= limit:
        out["note"] = (
            f"Result count equals the limit ({limit}); more matches may exist. "
            "Raise limit or refine the search."
        )
    return to_json(out)


@mcp.tool(annotations=READ_TOOL)
def get_datasource_info(datasource_id: str) -> str:
    """Get full metadata for a specific datasource.

    Returns all fields including schema, coordinates, geometry, time range,
    variables, and attributes.

    Args:
        datasource_id: The unique ID of the datasource.

    Returns:
        Full datasource metadata as JSON.
    """
    conn = get_datamesh_connector()
    ds = conn.get_datasource(datasource_id)
    return to_json(format_datasource(ds))


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
    echoes the canonical query.
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
    try:
        stage = _stage(conn, query)
    except _DATAMESH_ERRORS as exc:
        return to_json({"error": str(exc), "query": _query_echo(query)})

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
    else:
        detail = (
            "query_data will return only a lazy structure summary; shrink the "
            "result with filters, aggregation, or time_resolution"
            if stage.container == Container.Dataset
            else "narrow the query with filters or aggregation"
        )
        out["recommendation"] = (
            f"Larger than the inline limit ({human_bytes(inline_limit)}): "
            f"{detail}, or use export_query to write it to a file."
        )
    if (
        stage.container in (Container.DataFrame, Container.GeoDataFrame)
        and stage.dlen >= DATAMESH_ROW_CAP
    ):
        out["warnings"] = [
            f"Datamesh caps tabular results at {DATAMESH_ROW_CAP} rows; this "
            "result would be truncated. Narrow the query."
        ]
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
    warnings: list[str] = []
    try:
        stage = _stage(conn, query)
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
                return _refusal(
                    stage,
                    f"Result is {human_bytes(stage.size)}, above the inline "
                    f"limit of {human_bytes(inline_limit)}. Narrow the query "
                    "with filters or aggregation, or use export_query to "
                    "write it to a file.",
                    query=_query_echo(query),
                )
        with _captured_warnings(warnings):
            data = conn.query(query, use_dask=use_dask)
    except _DATAMESH_ERRORS as exc:
        return to_json({"error": str(exc), "query": _query_echo(query)})

    out = summarize_data(data, warnings=warnings)
    out["staged_size_bytes"] = stage.size
    return to_json(out)


def export_query(
    datasource_id: str,
    path: str,
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
    """Run a query and write the FULL result to a local file.

    This is the data-handle path for results too large to return inline: the
    data never enters the conversation — analysis code reads the file instead.
    Gridded datasets stream lazily to NetCDF; tabular results write Parquet or
    CSV.
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

    dest = _resolve_export_path(path)
    if dest.exists():
        if dest.is_dir():
            raise ToolError(f"path is an existing directory: {dest}")
        if not overwrite:
            raise ToolError(f"File exists: {dest}. Pass overwrite=true to replace it.")

    warnings: list[str] = []
    try:
        stage = _stage(conn, query)
        if stage is None:
            return to_json(
                {
                    "status": "no_data",
                    "message": "No data matches this query; nothing written.",
                    "query": _query_echo(query),
                }
            )

        if stage.container == Container.Dataset:
            fmt = format or "netcdf"
            if fmt != "netcdf":
                raise ToolError(
                    "This query returns a gridded dataset; only format='netcdf' "
                    "is supported."
                )
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

        with _captured_warnings(warnings):
            # Datasets stream chunk-wise from lazy zarr; frames download fully.
            data = conn.query(query, use_dask=stage.container == Container.Dataset)
    except _DATAMESH_ERRORS as exc:
        return to_json({"error": str(exc), "query": _query_echo(query)})

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

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if fmt == "netcdf":
            data.to_netcdf(dest)
        elif fmt == "parquet":
            data.to_parquet(dest)
        else:
            data.to_csv(dest, index=False)
    except (*_DATAMESH_ERRORS, OSError) as exc:
        # A mid-stream failure (zarr chunk fetch, disk) leaves a partial file.
        dest.unlink(missing_ok=True)
        return to_json(
            {
                "error": f"Export failed while writing {dest}: {exc}",
                "query": _query_echo(query),
            }
        )

    summary = summarize_data(data, max_rows=0, warnings=warnings)
    return to_json(
        {
            "path": str(dest),
            "format": fmt,
            "bytes_written": dest.stat().st_size,
            "summary": summary,
        }
    )


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
        structure summary, explicit truncated/lazy flags, staged size, and any
        server warnings.
    """
export_query.__doc__ = f"""{export_query.__doc__}
    Args:
        path: Destination file path (parent directories are created; confined to OCEANUM_MCP_EXPORT_DIR when set).
        format: Output format: netcdf (datasets), parquet or csv (tabular). Defaults by container: dataset -> netcdf, tabular -> parquet.
        overwrite: Overwrite an existing file (default false).
{_QUERY_PARAM_DOCS}

    Returns:
        JSON with the written path, format, bytes_written, and a structure
        summary of the exported data (no inline values).
    """

stage_query = mcp.tool(annotations=READ_TOOL)(stage_query)
query_data = mcp.tool(annotations=READ_TOOL)(query_data)
export_query = mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)(export_query)


@mcp.tool(annotations=READ_TOOL)
def load_datasource(datasource_id: str) -> str:
    """Summarize an entire datasource.

    Gridded datasources are opened lazily (no data download). Tabular
    datasources are downloaded only if under the inline size limit; use
    query_data with filters or export_query otherwise.

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
                "filters, or export_query to write it to a file.",
                datasource_id=datasource_id,
            )
        with _captured_warnings(warnings):
            data = conn.load_datasource(datasource_id)
    except _DATAMESH_ERRORS as exc:
        return to_json({"error": str(exc), "datasource_id": datasource_id})
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

if is_network_transport():
    # A hosted server has no meaningful local filesystem for clients:
    # export_query writes to the server's disk, not the caller's.
    mcp.disable(names={"export_query"})
