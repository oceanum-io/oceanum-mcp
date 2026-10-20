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
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from oceanum_mcp.common.config import is_network_transport, max_inline_rows


def to_json(obj: Any) -> str:
    """Serialize a tool result dict to the JSON string returned to the client."""
    return json.dumps(obj, indent=2, default=str)


def to_compact_json(obj: Any) -> str:
    """to_json without whitespace, for catalog metadata: it is mostly nested
    lists and dicts, which indentation roughly doubles (OCE-360)."""
    return json.dumps(obj, separators=(",", ":"), default=str)


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


def _float_value(value: float | np.floating) -> float | None:
    """A JSON-safe float at full precision; NaN/inf -> None.

    float64 converts exactly (json.dumps then writes its shortest round-trip
    repr). Narrower floats (float32/float16) take the shortest repr that
    round-trips in their own precision: widening 0.1f to float64 directly
    would print as 0.10000000149011612, digits the source data never had.
    """
    if not math.isfinite(value):
        return None
    if isinstance(value, np.floating) and value.dtype.itemsize < 8:
        return float(np.format_float_scientific(value, unique=True))
    return float(value)


# Marks object-column cells that are not floats: they keep to_json's value.
_KEEP = object()


def _exact_floats(col: pd.Series) -> list[Any] | None:
    """Full-precision values for a column's float cells, or None if it has none.

    Float dtypes (numpy, nullable, sparse, Arrow) convert every cell. Object
    columns convert their float cells and mark the rest _KEEP.
    """
    dtype = col.dtype
    if isinstance(dtype, np.dtype) and dtype.kind == "O":
        return [
            _float_value(v) if isinstance(v, (float, np.floating)) else _KEEP
            for v in col
        ]
    if not pd.api.types.is_float_dtype(dtype):
        return None
    if not isinstance(dtype, np.dtype):
        # Nullable/Arrow dtypes expose numpy_dtype; SparseDtype has subtype.
        dtype = getattr(dtype, "numpy_dtype", None) or dtype.subtype
    return [_float_value(v) for v in col.to_numpy(dtype=dtype, na_value=np.nan)]


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    # to_json rounds floats to 10 decimal places (5e-12 -> 0.0), and its
    # maximum double_precision=15 is still decimal places, not significant
    # digits (OCE-325). Keep to_json for every other dtype (ISO datetimes and
    # durations, NaT -> null) and replace float cells with exact values.
    # Rows are serialized as positional lists and keyed by str(label), as in
    # the summary's column list. Labels that repeat as keys (duplicates, or
    # 1 and "1") get a ".N" suffix so no column is dropped.
    keys: list[str] = []
    for label in df.columns:
        key = base = str(label)
        n = 0
        while key in keys:
            n += 1
            key = f"{base}.{n}"
        keys.append(key)
    rows = json.loads(df.to_json(orient="values", date_format="iso"))
    for i, (_, col) in enumerate(df.items()):
        floats = _exact_floats(col)
        if floats is None:
            continue
        for row, value in zip(rows, floats, strict=True):
            if value is not _KEEP:
                row[i] = value
    return [dict(zip(keys, row, strict=True)) for row in rows]


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


def _scalar_value(var: xr.DataArray) -> Any:
    """JSON-safe value of a 0-d variable, at full precision.

    Datetimes/timedeltas use the records conversion (ISO strings, NaT ->
    null) to match other records; numbers skip it because DataFrame.to_json
    rounds floats to 10 decimals, which would erase small aggregates.
    """
    value = var.values[()]
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return _records(pd.DataFrame({"v": [value]}))[0]["v"]
    if isinstance(value, np.floating):
        return _float_value(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _scalar_records(ds: xr.Dataset) -> list[dict[str, Any]]:
    """One record per variable of a 0-d Dataset: name, value, units/long_name."""
    out = []
    for name, var in ds.data_vars.items():
        rec: dict[str, Any] = {"name": str(name), "value": _scalar_value(var)}
        for key in ("units", "long_name"):
            if var.attrs.get(key) not in (None, ""):
                rec[key] = str(var.attrs[key])
        out.append(rec)
    return out


def _dataset_summary(ds: xr.Dataset, max_rows: int) -> dict[str, Any]:
    if not ds.sizes and max_rows > 0:
        # A 0-d result is one value per variable: compute it even if lazy
        # (use_dask is chosen from the staged size, not the result's).
        ds = ds.compute()
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
    elif max_rows > 0 and not ds.sizes:
        # A 0-d result (e.g. aggregate_operations over space and time) has no
        # index for to_dataframe (OCE-320): it is one record of scalars,
        # returned as one scalar entry per variable. Datasets with any dim
        # still go through to_dataframe, which broadcasts 0-d variables.
        out["data"] = _scalar_records(ds)
        out["truncated"] = False
        # One record, as counted elsewhere (prod of no dims = 1).
        out.update(_preview_fields(1, 1))
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
SUMMARY_MAX_TAGS = 5
SUMMARY_TAG_CHARS = 60  # per tag
# Decimal places kept in summary bounds, rounded outward so coverage never
# shrinks: ~10 m, and drops float32 noise (e.g. -38.599998474121094).
SUMMARY_BOUNDS_DECIMALS = 4

# get_datasource_info summary view (OCE-360).
INFO_DESCRIPTION_CHARS = 1000
INFO_MAX_GLOBAL_ATTRS = 25  # global attributes, and info entries
INFO_VALUE_CHARS = 200  # longest attribute value shown before clipping
# Target for the serialized summary. Past it, attributes and per-variable
# details are shed in stages (_FIT_STAGES); every variable name always stays,
# as a query needs it, so a record with very many variables can still exceed it.
INFO_BUDGET_CHARS = 30_000

# The only attributes a variable or coordinate keeps in the summary view.
_PRIORITY_ATTRS = ("units", "long_name", "standard_name")

_SUMMARY_NOTE = (
    "Summary view: each variable and coordinate keeps its dims, shape, dtype "
    "and units/long_name/standard_name only (see attrs_omitted counts), the "
    "description and long attribute values are clipped (ending in ...), "
    "global attribute lists are capped (see *_omitted counts), and "
    "schema.data_vars/attrs are omitted as duplicates of "
    'variables/attributes. Pass detail="full" for the complete record, with '
    "every attribute."
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


def _summary_entry(entry: Any) -> Any:
    """A schema variable/coordinate entry with only _PRIORITY_ATTRS kept.

    dims, shape and dtype pass through unchanged: they are small, and clients
    index them structurally.
    """
    if not isinstance(entry, dict):
        return entry
    out: dict[str, Any] = {}
    for key, value in entry.items():
        if key == "attrs" and isinstance(value, dict):
            out["attrs"] = {
                k: _clip_value(value[k], INFO_VALUE_CHARS)
                for k in _PRIORITY_ATTRS
                if k in value
            }
            if len(value) > len(out["attrs"]):
                out["attrs_omitted"] = len(value) - len(out["attrs"])
        else:
            out[key] = value
    return out


def _summary_entries(entries: dict[str, Any]) -> dict[str, Any]:
    return {str(n): _summary_entry(e) for n, e in entries.items()}


def _round_out(value: float, up: bool) -> float:
    """value rounded to SUMMARY_BOUNDS_DECIMALS, away from the box's inside.
    Raises on a non-finite or overflowing value."""
    scale = 10**SUMMARY_BOUNDS_DECIMALS
    # round() first: a float product such as 1650000.0000000002 must not
    # step to the next unit.
    step = math.ceil if up else math.floor
    return step(round(value * scale, 6)) / scale + 0.0  # + 0.0: no -0.0


def _summary_bounds(ds: Any) -> list[float] | None:
    """ds's bounds rounded outward, or None if they cannot be (empty or
    non-finite bounds, overflow): one odd record must not fail a search."""
    try:
        xmin, ymin, xmax, ymax = ds.bounds
        return [
            _round_out(xmin, up=False),
            _round_out(ymin, up=False),
            _round_out(xmax, up=True),
            _round_out(ymax, up=True),
        ]
    except Exception:  # noqa: BLE001 - any failure means "no usable bounds"
        return None


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
        result["bounds"] = _summary_bounds(ds)
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
    if tags:
        result["tags"] = [_clip(t, SUMMARY_TAG_CHARS) for t in tags[:SUMMARY_MAX_TAGS]]
    if len(tags) > SUMMARY_MAX_TAGS:
        result["tags_total"] = len(tags)
    return result


def format_datasource_bounded(ds: Any) -> dict[str, Any]:
    """The full record's fields, with every large section reduced.

    Every variable and coordinate keeps its dims, shape, dtype and
    units/long_name/standard_name only. The description and long attribute
    values are clipped and global attribute lists capped. schema.data_vars
    and schema.attrs are dropped when they duplicate variables/attributes.
    Past INFO_BUDGET_CHARS, attributes and then per-variable details are
    shed (see _fit_budget); every variable name stays.
    """
    full = format_datasource(ds)
    out = dict(full)
    if isinstance(full.get("description"), str):
        out["description"] = _clip(full["description"], INFO_DESCRIPTION_CHARS)
    for key in ("attributes", "info"):
        if isinstance(full.get(key), dict):
            out[key], omitted = _bounded_attrs(full[key], INFO_MAX_GLOBAL_ATTRS)
            if omitted:
                out[f"{key}_omitted"] = omitted
    if isinstance(full.get("variables"), dict):
        out["variables"] = _summary_entries(full["variables"])
    if isinstance(full.get("schema"), dict):
        schema = full["schema"]
        bounded: dict[str, Any] = {
            "dims": schema.get("dims"),
            "coords": _summary_entries(schema.get("coords") or {}),
        }
        # data_vars/attrs duplicate variables/attributes only on a detailed
        # record; keep them (reduced) when the record has no such fields.
        if "variables" not in full and schema.get("data_vars"):
            bounded["data_vars"] = _summary_entries(schema["data_vars"])
        if "attributes" not in full and schema.get("attrs"):
            bounded["attrs"], _ = _bounded_attrs(schema["attrs"], INFO_MAX_GLOBAL_ATTRS)
        out["schema"] = bounded
    out["note"] = _SUMMARY_NOTE
    _fit_budget(out)
    return out


def _shed_attrs(entries: Any, keep: tuple[str, ...]) -> None:
    """Drop the attributes of each schema entry not in keep, in place,
    adding them to its attrs_omitted count."""
    if not isinstance(entries, dict):
        return
    for entry in entries.values():
        attrs = entry.get("attrs") if isinstance(entry, dict) else None
        if not isinstance(attrs, dict):
            continue
        dropped = [k for k in attrs if k not in keep]
        for k in dropped:
            del attrs[k]
        if dropped:
            entry["attrs_omitted"] = entry.get("attrs_omitted", 0) + len(dropped)


def _shed_global_attrs(out: dict[str, Any]) -> None:
    for parent, key, count_key in (
        (out, "attributes", "attributes_omitted"),
        (out, "info", "info_omitted"),
        (out.get("schema") or {}, "attrs", "attributes_omitted"),
    ):
        if isinstance(parent.get(key), dict) and parent[key]:
            out[count_key] = out.get(count_key, 0) + len(parent[key])
            parent[key] = {}


def _variable_listings(out: dict[str, Any]) -> list[Any]:
    return [out.get("variables"), (out.get("schema") or {}).get("data_vars")]


def _shed_variable_attrs(out: dict[str, Any]) -> None:
    for entries in _variable_listings(out):
        _shed_attrs(entries, ("units",))


def _shed_variable_fields(out: dict[str, Any], keep: tuple[str, ...]) -> None:
    """Reduce each variable entry to the keys in keep, in place."""
    for entries in _variable_listings(out):
        if not isinstance(entries, dict):
            continue
        for name, entry in entries.items():
            if isinstance(entry, dict):
                entries[name] = {k: v for k, v in entry.items() if k in keep}


# What _fit_budget sheds, in order, until the summary fits: cheapest loss
# first, and never a variable name.
_FIT_STAGES: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
    ("global attributes", _shed_global_attrs),
    (
        "coordinate attributes",
        lambda out: _shed_attrs((out.get("schema") or {}).get("coords"), ()),
    ),
    ("variable attributes other than units", _shed_variable_attrs),
    (
        "variable shapes and dtypes",
        lambda out: _shed_variable_fields(out, ("dims", "attrs", "attrs_omitted")),
    ),
    ("variable details", lambda out: _shed_variable_fields(out, ())),
)


def _fit_budget(out: dict[str, Any]) -> None:
    """Shed _FIT_STAGES from out, in order, until it serializes within
    INFO_BUDGET_CHARS, naming in its note the stages that removed anything.
    Sets budget_exceeded if every variable name alone does not fit."""
    base_note = out["note"]
    shed: list[str] = []
    size = len(to_compact_json(out))
    for label, stage in _FIT_STAGES:
        if size <= INFO_BUDGET_CHARS:
            return
        stage(out)
        smaller = len(to_compact_json(out))
        if smaller < size:
            shed.append(label)
            out["note"] = (
                f"{base_note} This record is large, so its {', '.join(shed)} "
                "are also left out (see the *_omitted counts); every variable "
                'is still listed. Pass detail="full" for them.'
            )
        size = len(to_compact_json(out))
    if size > INFO_BUDGET_CHARS:
        out["budget_exceeded"] = True
