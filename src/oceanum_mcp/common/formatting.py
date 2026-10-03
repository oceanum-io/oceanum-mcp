"""Shared data formatting and summarization for Oceanum MCP servers.

All summaries are plain dicts so tools can return one consistent JSON shape.
Values shown inline are always coordinate-attributed (records, not bare
arrays), and truncation or lazy loading is flagged explicitly so the model
never mistakes a preview for the full result.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any

import pandas as pd
import xarray as xr

from oceanum_mcp.common.config import is_network_transport, max_inline_rows


def to_json(obj: Any) -> str:
    """Serialize a tool result dict to the JSON string returned to the client."""
    return json.dumps(obj, indent=2, default=str)


def export_clause() -> str:
    """Trailing clause pointing at export_query, in the current transport's terms.

    export_query behaves differently by transport — on hosted it returns a
    signed download link, on stdio it writes a local file. This is the single
    source of truth for that wording; all result-size guidance (here and in the
    datamesh server's messages) derives its export phrasing from this function,
    evaluated at call time so it always matches the running transport.
    """
    if is_network_transport():
        return ", or use export_query to get a download link for the full result"
    return ", or use export_query to write the full result to a file"


def _preview_note(lead: str) -> str:
    """Note for a partial inline result: never analyse it, get stats server-side."""
    return (
        lead + " PREVIEW ONLY: this is not the full result, so it is NOT "
        "suitable for statistics or aggregation (means, extremes, counts, "
        "trends). Compute statistics over the full result server-side with "
        "aggregate_operations or time_resolution downsampling; narrow the "
        "query with filters to see more values inline" + export_clause() + "."
    )


def _preview_fields(returned: int, total: int) -> dict[str, Any]:
    """Machine-readable preview flags shared by every inline result shape.

    total is the record count of the full result; preview is true whenever
    fewer records were returned than the result holds.
    """
    return {"preview": returned < total, "returned": returned, "total": total}


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
    out.update(_preview_fields(int(shown.shape[0]), int(df.shape[0])))
    if out["truncated"]:
        out["note"] = _preview_note(f"Showing first {max_rows} of {df.shape[0]} rows.")
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
    if lazy and max_rows > 0:
        # No values are returned; the record count (the full dim product, as
        # to_dataframe would yield) is known without downloading anything.
        out.update(_preview_fields(0, math.prod(out["dims"].values())))
        out["note"] = _preview_note("Dataset is lazily loaded (values not downloaded).")
    elif lazy:
        out["note"] = (
            "Dataset is lazily loaded (values not downloaded). Narrow the query "
            "with filters, aggregation, or time_resolution downsampling to see "
            "values inline" + export_clause() + "."
        )
    elif max_rows > 0:
        # Eager data is already in memory — always include a preview of
        # coordinate-attributed values.
        df = ds.to_dataframe().reset_index()
        shown = df.head(max_rows)
        out["data"] = _records(shown)
        out["truncated"] = df.shape[0] > max_rows
        out.update(_preview_fields(int(shown.shape[0]), int(df.shape[0])))
        if out["truncated"]:
            out["note"] = _preview_note(
                f"Showing first {max_rows} of {df.shape[0]} records."
            )
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


# ---------------------------------------------------------------------------
# Compact datasource views (OCE-309)
#
# format_datasource above is the complete record (detail="full"). Real records
# carry GRIB coefficient arrays, hundreds of model-parameter attributes and
# 100+ variables, and schema.data_vars/attrs repeat variables/attributes, so
# the full record can run to hundreds of kB per datasource.
# ---------------------------------------------------------------------------

# search_catalog summaries.
SUMMARY_DESCRIPTION_CHARS = 200
SUMMARY_MAX_VARIABLES = 20
SUMMARY_TAG_CHARS = 120

# get_datasource_info bounded view.
INFO_MAX_ATTRS = 8  # attributes per variable / coordinate
INFO_MAX_GLOBAL_ATTRS = 25  # global attributes, and info entries
INFO_VALUE_CHARS = 200  # longest attribute value shown before clipping
# Above this serialized size the view is rebuilt lean: every variable and
# coordinate keeps only dims/shape/dtype and its priority attributes. Every
# variable stays listed either way: its exact name is what a query needs.
INFO_BUDGET_CHARS = 60_000

# Kept first, so a capped attribute list still carries what code needs.
_PRIORITY_ATTRS = ("units", "long_name", "standard_name")

_BOUNDED_NOTE = (
    "Bounded view: long attribute values are clipped (ending in ...), attribute "
    "lists are capped (see *_omitted counts), and schema.data_vars/attrs are "
    'omitted as duplicates of variables/attributes. Pass detail="full" for '
    "the complete record."
)
_LEAN_NOTE = (
    "Bounded view of a large record: each variable and coordinate keeps only "
    "dims, shape, dtype and units/long_name/standard_name (see attrs_omitted "
    "counts), global attribute lists are capped, long values are clipped "
    "(ending in ...), and schema.data_vars/attrs are omitted as duplicates of "
    'variables/attributes. Pass detail="full" for the complete record.'
)


def _clip(text: str, limit: int) -> str:
    """text cut to limit characters, with an ellipsis when cut."""
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _clip_value(value: Any, limit: int) -> Any:
    """An attribute value, replaced by a clipped JSON string if it is long."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return value if len(text) <= limit else _clip(text, limit)


def _bounded_attrs(attrs: dict[str, Any], max_items: int) -> tuple[dict, int]:
    """Priority attributes plus others up to max_items in all, values clipped.

    Priority attributes are always kept. Returns the kept attributes and how
    many were omitted.
    """
    keys = [k for k in _PRIORITY_ATTRS if k in attrs]
    others = [k for k in attrs if k not in _PRIORITY_ATTRS]
    keys += others[: max(0, max_items - len(keys))]
    kept = {str(k): _clip_value(attrs[k], INFO_VALUE_CHARS) for k in keys}
    return kept, len(attrs) - len(kept)


def _bounded_entry(entry: Any, max_attrs: int) -> Any:
    """A schema variable/coordinate entry with its attributes bounded.

    dims, shape and dtype pass through unchanged: they are small, and clients
    index them structurally.
    """
    if not isinstance(entry, dict):
        return entry
    out: dict[str, Any] = {}
    for key, value in entry.items():
        if key == "attrs" and isinstance(value, dict):
            out["attrs"], omitted = _bounded_attrs(value, max_attrs)
            if omitted:
                out["attrs_omitted"] = omitted
        else:
            out[key] = value
    return out


def _bounded_entries(entries: dict[str, Any], max_attrs: int) -> dict[str, Any]:
    return {str(n): _bounded_entry(e, max_attrs) for n, e in entries.items()}


def format_datasource_summary(ds: Any) -> dict[str, Any]:
    """A compact search_catalog hit: identity, coverage and variable names.

    Catalog hits from the gateway usually carry no schema, so variable names
    appear only when the record includes them.
    """
    description = ds.description
    if isinstance(description, str):
        description = _clip(" ".join(description.split()), SUMMARY_DESCRIPTION_CHARS)
    result: dict[str, Any] = {"id": ds.id, "name": ds.name, "description": description}
    if ds.geom is not None:
        result["bounds"] = list(ds.bounds)
    if ds.tstart is not None:
        result["tstart"] = ds.tstart.isoformat()
    if ds.tend is not None:
        result["tend"] = ds.tend.isoformat()

    data_vars = getattr(ds.dataschema, "data_vars", None)
    if isinstance(data_vars, dict) and data_vars:
        names = [str(n) for n in data_vars]
        result["variables"] = names[:SUMMARY_MAX_VARIABLES]
        if len(names) > SUMMARY_MAX_VARIABLES:
            result["variables_total"] = len(names)

    tags = [str(t) for t in ds.tags or []]
    shown: list[str] = []
    for tag in tags:
        if sum(map(len, shown)) + len(tag) > SUMMARY_TAG_CHARS:
            break
        shown.append(tag)
    if shown:
        result["tags"] = shown
    if len(shown) < len(tags):
        result["tags_total"] = len(tags)
    return result


def _bounded_view(full: dict[str, Any], max_attrs: int) -> dict[str, Any]:
    out = dict(full)
    for key in ("attributes", "info"):
        if isinstance(full.get(key), dict):
            out[key], omitted = _bounded_attrs(full[key], INFO_MAX_GLOBAL_ATTRS)
            if omitted:
                out[f"{key}_omitted"] = omitted
    if isinstance(full.get("variables"), dict):
        out["variables"] = _bounded_entries(full["variables"], max_attrs)
    if isinstance(full.get("schema"), dict):
        schema = full["schema"]
        bounded: dict[str, Any] = {
            "dims": schema.get("dims"),
            "coords": _bounded_entries(schema.get("coords") or {}, max_attrs),
        }
        # data_vars/attrs duplicate variables/attributes only on a detailed
        # record; keep them (bounded) when the record has no such fields.
        if "variables" not in full and schema.get("data_vars"):
            bounded["data_vars"] = _bounded_entries(schema["data_vars"], max_attrs)
        if "attributes" not in full and schema.get("attrs"):
            bounded["attrs"], _ = _bounded_attrs(schema["attrs"], INFO_MAX_GLOBAL_ATTRS)
        out["schema"] = bounded
    return out


def format_datasource_bounded(ds: Any) -> dict[str, Any]:
    """The full record's fields and shape, with every large section bounded.

    Every variable and coordinate is kept with its dims, shape, dtype and
    priority attributes (units, long_name, standard_name); up to
    INFO_MAX_ATTRS attributes in all, with long values clipped. If that still
    serializes past INFO_BUDGET_CHARS, entries keep only the priority
    attributes. schema.data_vars and schema.attrs are dropped when they
    duplicate variables/attributes.
    """
    full = format_datasource(ds)
    out = _bounded_view(full, INFO_MAX_ATTRS)
    out["note"] = _BOUNDED_NOTE
    if len(to_json(out)) > INFO_BUDGET_CHARS:
        out = _bounded_view(full, 0)
        out["note"] = _LEAN_NOTE
    return out
