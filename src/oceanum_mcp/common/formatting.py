"""Shared data formatting and summarization for Oceanum MCP servers."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import xarray as xr


def summarize_data(data: Any, max_rows: int = 20) -> str:
    """Summarize a data result for MCP output."""
    if data is None:
        return "No data returned."

    if isinstance(data, pd.DataFrame):
        lines = [
            f"DataFrame: {data.shape[0]} rows x {data.shape[1]} columns",
            f"Columns: {list(data.columns)}",
            f"Dtypes:\n{data.dtypes.to_string()}",
        ]
        if data.shape[0] <= max_rows:
            lines.append(f"\nData:\n{data.to_string()}")
        else:
            lines.append(
                f"\nFirst {max_rows} rows:\n{data.head(max_rows).to_string()}"
            )
            lines.append(
                f"\n... ({data.shape[0] - max_rows} more rows. "
                "Narrow your query with time/geo/variable filters to get smaller results.)"
            )
        return "\n".join(lines)

    if isinstance(data, xr.Dataset):
        lines = [f"xarray Dataset:\n{data}"]
        if data.nbytes < 1_000_000:
            # Small enough to show values
            for var in list(data.data_vars)[:5]:
                arr = data[var]
                lines.append(f"\n{var} sample values:\n{arr.values.flat[:20]}")
        else:
            lines.append(
                f"\nDataset size: {data.nbytes / 1e6:.1f} MB. "
                "Use query_data with filters to retrieve a subset."
            )
        return "\n".join(lines)

    return str(data)


def format_datasource(ds: Any) -> str:
    """Format a Datasource object into a JSON string for MCP output."""
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
    return json.dumps(result, indent=2, default=str)
