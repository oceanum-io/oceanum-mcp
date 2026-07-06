"""Shared data formatting and summarization for Oceanum MCP servers.

All summaries are plain dicts so tools can return one consistent JSON shape.
Values shown inline are always coordinate-attributed (records, not bare
arrays), and truncation or lazy loading is flagged explicitly so the model
never mistakes a preview for the full result.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import xarray as xr

# Datasets under this many bytes have their values included as records.
SMALL_DATASET_NBYTES = 1_000_000


def to_json(obj: Any) -> str:
    """Serialize a tool result dict to the JSON string returned to the client."""
    return json.dumps(obj, indent=2, default=str)


def human_bytes(n: int | float) -> str:
    """Format a byte count for display."""
    n = float(n)
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1000
    return f"{n:.1f} TB"


def _frame_summary(df: pd.DataFrame, max_rows: int) -> dict[str, Any]:
    is_geo = hasattr(df, "geometry") and df.__class__.__name__ == "GeoDataFrame"
    out: dict[str, Any] = {
        "container": "geodataframe" if is_geo else "dataframe",
        "rows": int(df.shape[0]),
        "columns": [{"name": str(c), "dtype": str(t)} for c, t in df.dtypes.items()],
    }
    shown = df.head(max_rows)
    if is_geo:
        shown = shown.copy()
        shown["geometry"] = shown["geometry"].astype(str)
    records = json.loads(shown.to_json(orient="records", date_format="iso"))
    out["data"] = records
    out["truncated"] = df.shape[0] > max_rows
    if out["truncated"]:
        out["note"] = (
            f"Showing first {max_rows} of {df.shape[0]} rows. Narrow the query "
            "with filters or aggregation, or use export_query for the full result."
        )
    return out


def _dataset_summary(ds: xr.Dataset, max_rows: int) -> dict[str, Any]:
    lazy = any(ds[v].chunks is not None for v in ds.data_vars)
    out: dict[str, Any] = {
        "container": "dataset",
        "dims": {str(k): int(v) for k, v in ds.sizes.items()},
        "coords": {
            str(name): {
                "dtype": str(coord.dtype),
                "size": int(coord.size),
                "first": str(coord.values.flat[0]) if coord.size else None,
                "last": str(coord.values.flat[-1]) if coord.size else None,
            }
            for name, coord in ds.coords.items()
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
            "Dataset is lazily loaded (values not downloaded). Use query_data "
            "with narrower filters or aggregation to see values inline, or "
            "export_query to write the data to a file."
        )
    elif ds.nbytes <= SMALL_DATASET_NBYTES:
        # Small and eager: include coordinate-attributed values as records.
        df = ds.to_dataframe().reset_index()
        out["data"] = json.loads(
            df.head(max_rows).to_json(orient="records", date_format="iso")
        )
        out["truncated"] = df.shape[0] > max_rows
        if out["truncated"]:
            out["note"] = (
                f"Showing first {max_rows} of {df.shape[0]} records. Use "
                "export_query for the full result."
            )
    else:
        out["note"] = (
            "Values omitted (dataset larger than inline limit). Narrow the "
            "query or use export_query."
        )
    return out


def summarize_data(
    data: Any, max_rows: int = 20, warnings: list[str] | None = None
) -> dict[str, Any]:
    """Summarize a query result as a structured dict for MCP output."""
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
