"""Oceanum Datamesh MCP server.

Exposes the Oceanum Datamesh API as MCP tools for AI assistants.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from oceanum_mcp.common.client import get_datamesh_connector
from oceanum_mcp.common.formatting import summarize_data, format_datasource

mcp = FastMCP(
    "Oceanum Datamesh",
    instructions=(
        "Access the Oceanum Datamesh platform for ocean and environmental data. "
        "Use search_catalog to discover datasets, get_datasource_info for metadata, "
        "and query_data or load_datasource to retrieve data."
    ),
)


# ---------------------------------------------------------------------------
# Catalog & Discovery
# ---------------------------------------------------------------------------


@mcp.tool()
def search_catalog(
    search: str | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    bbox: list[float] | None = None,
    limit: int | None = None,
) -> str:
    """Search the Oceanum Datamesh catalog for datasets.

    Args:
        search: Text search string to filter datasources by name, description, or tags.
        time_start: ISO 8601 datetime for start of time range filter (e.g. "2023-01-01").
        time_end: ISO 8601 datetime for end of time range filter (e.g. "2023-12-31").
        bbox: Bounding box as [xmin, ymin, xmax, ymax] in WGS84 coordinates.
        limit: Maximum number of datasources to return.

    Returns:
        List of matching datasources with id, name, description, time range, and bounds.
    """
    conn = get_datamesh_connector()

    timefilter = None
    if time_start or time_end:
        timefilter = [time_start or "1900-01-01", time_end or "2100-01-01"]

    geofilter = None
    if bbox:
        from oceanum.datamesh.query import GeoFilter

        geofilter = GeoFilter(type="bbox", geom=bbox)

    catalog = conn.get_catalog(
        search=search,
        timefilter=timefilter,
        geofilter=geofilter,
        limit=limit,
    )

    if len(catalog) == 0:
        return "No datasources found matching the search criteria."

    results = []
    for ds in catalog:
        if ds is None:
            continue
        results.append(json.loads(format_datasource(ds)))

    return json.dumps(results, indent=2, default=str)


@mcp.tool()
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
    return format_datasource(ds)


# ---------------------------------------------------------------------------
# Data Access
# ---------------------------------------------------------------------------


@mcp.tool()
def query_data(
    datasource_id: str,
    variables: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    bbox: list[float] | None = None,
    geofilter_geojson: str | None = None,
    level_min: float | None = None,
    level_max: float | None = None,
    coord_filters: str | None = None,
    aggregate_operations: list[str] | None = None,
    aggregate_spatial: bool = True,
    aggregate_temporal: bool = True,
    limit: int | None = None,
) -> str:
    """Query a datasource with filters and return data.

    For small results, returns the actual data. For large results, returns
    a summary with shape, columns/variables, and a preview of values.

    Args:
        datasource_id: The datasource to query.
        variables: List of variable names to select (e.g. ["temperature", "salinity"]).
        time_start: ISO 8601 start time (e.g. "2023-01-01T00:00:00Z").
        time_end: ISO 8601 end time (e.g. "2023-12-31T23:59:59Z").
        bbox: Bounding box as [xmin, ymin, xmax, ymax] in WGS84.
        geofilter_geojson: GeoJSON Feature string for spatial filtering (alternative to bbox).
        level_min: Minimum vertical level.
        level_max: Maximum vertical level.
        coord_filters: JSON string of additional coordinate filters, e.g. '[{"coord": "station", "values": ["A1", "B2"]}]'.
        aggregate_operations: List of aggregation operations (e.g. ["mean", "max"]). Options: mean, min, max, std, sum.
        aggregate_spatial: Whether to aggregate over spatial dimensions (default true).
        aggregate_temporal: Whether to aggregate over temporal dimensions (default true).
        limit: Maximum number of rows/records to return.

    Returns:
        Data summary or full data for small results.
    """
    conn = get_datamesh_connector()

    query_dict: dict[str, Any] = {"datasource": datasource_id}

    if variables:
        query_dict["variables"] = variables

    if time_start or time_end:
        query_dict["timefilter"] = {
            "type": "range",
            "times": [time_start or "1900-01-01", time_end or "2100-01-01"],
        }

    if bbox:
        query_dict["geofilter"] = {"type": "bbox", "geom": bbox}
    elif geofilter_geojson:
        feature = json.loads(geofilter_geojson)
        query_dict["geofilter"] = {"type": "feature", "geom": feature}

    if level_min is not None or level_max is not None:
        query_dict["levelfilter"] = {
            "type": "range",
            "levels": [level_min, level_max],
        }

    if coord_filters:
        query_dict["coordfilter"] = json.loads(coord_filters)

    if aggregate_operations:
        query_dict["aggregate"] = {
            "operations": aggregate_operations,
            "spatial": aggregate_spatial,
            "temporal": aggregate_temporal,
        }

    if limit is not None:
        query_dict["limit"] = limit

    data = conn.query(query_dict)
    return summarize_data(data)


@mcp.tool()
def load_datasource(datasource_id: str) -> str:
    """Load an entire datasource into memory.

    Best for small datasets. For large datasets, use query_data with filters instead.

    Args:
        datasource_id: The datasource to load.

    Returns:
        Data summary with shape, columns/variables, and preview of values.
    """
    conn = get_datamesh_connector()
    data = conn.load_datasource(datasource_id)
    return summarize_data(data)


# ---------------------------------------------------------------------------
# Data Management
# ---------------------------------------------------------------------------


@mcp.tool()
def update_metadata(
    datasource_id: str,
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    labels: list[str] | None = None,
    info: str | None = None,
    details: str | None = None,
) -> str:
    """Update metadata properties on an existing datasource.

    Only the provided fields will be updated; others remain unchanged.

    Args:
        datasource_id: The datasource to update.
        name: New human-readable name (max 128 chars).
        description: New description (max 1500 chars).
        tags: New list of keyword tags.
        labels: New list of metadata labels.
        info: JSON string of additional metadata dict.
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
        props["info"] = json.loads(info)
    if details is not None:
        props["details"] = details

    ds = conn.update_metadata(datasource_id, **props)
    return format_datasource(ds)
