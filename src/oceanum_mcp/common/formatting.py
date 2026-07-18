"""Shared data formatting and summarization for Oceanum MCP servers.

All summaries are plain dicts so tools can return one consistent JSON shape.
Values shown inline are always coordinate-attributed (records, not bare
arrays), and truncation or lazy loading is flagged explicitly so the model
never mistakes a preview for the full result.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pandas as pd
import xarray as xr

from oceanum_mcp.common.config import is_network_transport, max_inline_rows


def to_json(obj: Any) -> str:
    """Serialize a tool result dict to the JSON string returned to the client."""
    return json.dumps(obj, indent=2, default=str)


def export_clause() -> str:
    """Trailing clause pointing at export_query, only where it is available.

    export_query writes to the server's local disk and is disabled on network
    transports, so a hosted deployment must never name it. This is the single
    source of truth for that fact; all result-size guidance (here and in the
    datamesh server's messages) derives its export wording from this function,
    evaluated at call time so it always matches the running transport.
    """
    if is_network_transport():
        return ""
    return ", or use export_query to write the full result to a file"


def _narrow_hint() -> str:
    """How to get more than the inline preview, appropriate to the transport."""
    return (
        "Narrow the query with filters, aggregation, or time_resolution "
        "downsampling" + export_clause() + "."
    )


def human_bytes(n: int | float) -> str:
    """Format a byte count for display (decimal units)."""
    n = float(n)
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1000
    raise AssertionError("unreachable")


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _frame_summary(df: pd.DataFrame, max_rows: int) -> dict[str, Any]:
    # geopandas overrides DataFrame.to_json with a GeoJSON serializer, so geo
    # frames must be converted to plain pandas (geometry as WKT) before
    # serializing records. sys.modules is enough: a GeoDataFrame can only
    # exist if geopandas is already imported.
    gpd = sys.modules.get("geopandas")
    is_geo = gpd is not None and isinstance(df, gpd.GeoDataFrame)
    out: dict[str, Any] = {
        "container": "geodataframe" if is_geo else "dataframe",
        "rows": int(df.shape[0]),
        "columns": [{"name": str(c), "dtype": str(t)} for c, t in df.dtypes.items()],
    }
    if max_rows <= 0:
        # Structure-only summary (e.g. after an export): no preview, and no
        # truncation flags that could suggest the result itself is partial.
        return out
    shown = df.head(max_rows)
    if is_geo:
        plain = pd.DataFrame(shown).copy()
        for col in shown.columns:
            if isinstance(shown[col].dtype, gpd.array.GeometryDtype):
                plain[col] = shown[col].to_wkt()
        shown = plain
    out["data"] = _records(shown)
    out["truncated"] = df.shape[0] > max_rows
    if out["truncated"]:
        out["note"] = (
            f"Showing first {max_rows} of {df.shape[0]} rows. " + _narrow_hint()
        )
    return out


def _coord_summary(coord: xr.DataArray) -> dict[str, Any]:
    out: dict[str, Any] = {"dtype": str(coord.dtype), "size": int(coord.size)}
    # Chunked (dask-backed) coordinates would be downloaded in full just to
    # show first/last — skip values for those; dimension coords are in-memory.
    if coord.chunks is None and coord.size:
        out["first"] = str(coord.values.flat[0])
        out["last"] = str(coord.values.flat[-1])
    return out


def _dataset_summary(ds: xr.Dataset, max_rows: int) -> dict[str, Any]:
    lazy = any(ds[v].chunks is not None for v in ds.data_vars)
    out: dict[str, Any] = {
        "container": "dataset",
        "dims": {str(k): int(v) for k, v in ds.sizes.items()},
        "coords": {
            str(name): _coord_summary(coord) for name, coord in ds.coords.items()
        },
        "variables": {
            str(name): {
                "dims": [str(d) for d in var.dims],
                "shape": [int(s) for s in var.shape],
                "dtype": str(var.dtype),
            }
            for name, var in ds.data_vars.items()
        },
        "nbytes": int(ds.nbytes),
        "size_human": human_bytes(ds.nbytes),
        "lazy": lazy,
    }
    if lazy:
        out["note"] = (
            "Dataset is lazily loaded (values not downloaded). Narrow the query "
            "with filters, aggregation, or time_resolution downsampling to see "
            "values inline" + export_clause() + "."
        )
    elif max_rows > 0:
        # Eager data is already in memory — always include a preview of
        # coordinate-attributed values.
        df = ds.to_dataframe().reset_index()
        out["data"] = _records(df.head(max_rows))
        out["truncated"] = df.shape[0] > max_rows
        if out["truncated"]:
            out["note"] = (
                f"Showing first {max_rows} of {df.shape[0]} records. "
            ) + _narrow_hint()
    return out


def summarize_data(
    data: Any, max_rows: int | None = None, warnings: list[str] | None = None
) -> dict[str, Any]:
    """Summarize a query result as a structured dict for MCP output.

    max_rows defaults to the configured inline row cap (OCEANUM_MCP_MAX_INLINE_ROWS).
    max_rows <= 0 produces a structure-only summary with no value preview and
    no truncation flags (used after exports, where the written file is
    complete regardless of preview size).
    """
    if max_rows is None:
        max_rows = max_inline_rows()
    if data is None:
        summary: dict[str, Any] = {
            "status": "no_data",
            "message": "No data returned for this query.",
        }
    elif isinstance(data, pd.DataFrame):
        summary = _frame_summary(data, max_rows)
    elif isinstance(data, xr.Dataset):
        summary = _dataset_summary(data, max_rows)
    else:
        summary = {"container": type(data).__name__, "repr": str(data)}
    if warnings:
        summary["warnings"] = warnings
    return summary


def format_datasource(ds: Any) -> dict[str, Any]:
    """Format a Datasource object into a dict for MCP output."""
    result: dict[str, Any] = {
        "id": ds.id,
        "name": ds.name,
        "description": ds.description,
    }
    if ds.geom is not None:
        result["bounds"] = list(ds.bounds)
    if ds.tstart is not None:
        result["tstart"] = ds.tstart.isoformat()
    if ds.tend is not None:
        result["tend"] = ds.tend.isoformat()
    result["tags"] = ds.tags or []
    result["labels"] = ds.labels or []
    if ds.info:
        result["info"] = ds.info
    if ds.coordinates:
        result["coordinates"] = ds.coordinates
    if ds.variables is not None:
        result["variables"] = ds.variables
    if ds.attributes is not None:
        result["attributes"] = ds.attributes
    if ds.dataschema and ds.dataschema.dims:
        result["schema"] = {
            "dims": ds.dataschema.dims,
            "coords": ds.dataschema.coords,
            "data_vars": ds.dataschema.data_vars,
            "attrs": ds.dataschema.attrs,
        }
    result["driver"] = ds.driver
    if ds.details:
        result["details"] = str(ds.details)
    if ds.modified:
        result["modified"] = ds.modified.isoformat()
    if ds.created:
        result["created"] = ds.created.isoformat()
    return result
