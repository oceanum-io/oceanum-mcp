# Oceanum MCP

An [MCP](https://modelcontextprotocol.io/) server package that provides AI assistants with access to the [Oceanum](https://oceanum.io) platform for ocean/environmental data and cloud storage.

## Servers

This package contains multiple MCP servers, selectable at runtime:

| Server     | Description                                                   |
| ---------- | ------------------------------------------------------------- |
| `datamesh` | Search, query, and manage ocean/environmental datasets        |
| `storage`  | List, read, write, and delete files in Oceanum cloud storage  |
| `combined` | All tools from both servers under a single endpoint (default) |

## Prerequisites

Get an API token from [oceanum.io](https://oceanum.io). Set it as the `DATAMESH_TOKEN` environment variable.

## Installation

```bash
pip install oceanum-mcp
```

Or run directly with `uvx`:

```bash
uvx oceanum-mcp              # combined server (default)
uvx oceanum-mcp datamesh     # datamesh only
uvx oceanum-mcp storage      # storage only
uvx oceanum-mcp --list       # show available servers
```

## Configuration

### Claude Desktop

Add to your `claude_desktop_config.json`:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

**Combined server (all tools):**

```json
{
  "mcpServers": {
    "oceanum": {
      "command": "uvx",
      "args": ["oceanum-mcp"],
      "env": {
        "DATAMESH_TOKEN": "your-token-here"
      }
    }
  }
}
```

**Individual server (datamesh only):**

```json
{
  "mcpServers": {
    "oceanum-datamesh": {
      "command": "uvx",
      "args": ["oceanum-mcp", "datamesh"],
      "env": {
        "DATAMESH_TOKEN": "your-token-here"
      }
    }
  }
}
```

### Claude Code

```bash
# Combined server
claude mcp add --transport stdio oceanum -- uvx oceanum-mcp

# Individual server
claude mcp add --transport stdio oceanum-datamesh -- uvx oceanum-mcp datamesh
```

Set the token in your environment:

```bash
export DATAMESH_TOKEN=your-token-here
```

### VS Code / Cline / Continue

Use stdio transport with the same command:

```json
{
  "command": "uvx",
  "args": ["oceanum-mcp"],
  "env": {
    "DATAMESH_TOKEN": "your-token-here"
  }
}
```

## Environment Variables

| Variable                      | Required | Description                                                                     |
| ----------------------------- | -------- | ------------------------------------------------------------------------------- |
| `DATAMESH_TOKEN`              | Yes      | Oceanum API token (shared by all servers)                                       |
| `DATAMESH_SERVICE`            | No       | Custom datamesh service URL (default: `https://datamesh.oceanum.io`)            |
| `STORAGE_SERVICE`             | No       | Custom storage service URL (default: `https://storage.oceanum.io`)              |
| `OCEANUM_DOMAIN`              | No       | Override the base domain for all services (default: `oceanum.io`)               |
| `OCEANUM_MCP_READ_ONLY`       | No       | Set to `1`/`true` to disable write tools (`update_metadata`, storage `write_file`/`delete_file`) |
| `OCEANUM_MCP_MAX_INLINE_BYTES`| No       | Max staged result size returned inline by `query_data` (default 50,000,000)     |
| `OCEANUM_MCP_EXPORT_DIR`      | No       | If set, `export_query` may only write inside this directory                     |
| `OCEANUM_MCP_AUTH`            | No       | Auth scheme for `--transport http`: `datamesh` (default), `auth0`, or `none`    |
| `OCEANUM_MCP_AUTH0_DOMAIN`    | No       | Auth0 tenant domain for `auth0` mode (default: `auth.oceanum.io`)               |
| `OCEANUM_MCP_AUTH0_AUDIENCE`  | No       | Auth0 API audience for `auth0` mode (default: `https://api.oceanum.io`)         |

`DATAMESH_TOKEN` is required for the stdio transport (and for `--transport http`
with `OCEANUM_MCP_AUTH=none`); in authenticated http mode each request carries
its own credential and no server-side token is needed.

## Hosted mode (HTTP transport)

Run any server as a shared, multi-tenant HTTP service:

```bash
oceanum-mcp datamesh --transport http --host 0.0.0.0 --port 8000
```

This serves the MCP streamable-HTTP endpoint at `http://<host>:<port>/<server>`
(`/datamesh` here; override with `--path`). Each server owning its own path
lets several MCP servers share one domain behind an ingress — e.g.
`https://mcp.oceanum.io/datamesh` and `https://mcp.oceanum.io/storage`.
Every request must present a bearer credential, and all Datamesh/Storage calls
are made **as that request's user** — connections are cached per credential and
never shared between tokens.

Auth schemes (`OCEANUM_MCP_AUTH`):

- `datamesh` (default) — clients send their Datamesh token as the bearer:
  `Authorization: Bearer <datamesh-token>`. The server validates it against
  the gateway. Add to Claude Code with:

  ```bash
  claude mcp add --transport http oceanum-datamesh https://mcp.oceanum.io/datamesh \
    --header "Authorization: Bearer <datamesh-token>"
  ```

- `auth0` — clients send an Auth0-issued JWT, validated against the tenant
  JWKS and forwarded to the Datamesh gateway as-is.
- `none` — no authentication; every request uses the server's own
  `DATAMESH_TOKEN`. Only for trusted-network deployments.

Notes:

- `export_query` is disabled over HTTP — it writes to the server's local
  filesystem, which is meaningless for remote clients. Use `query_data` with
  filters, or Oceanum Storage.
- Combine with `OCEANUM_MCP_READ_ONLY=1` to run a read-only public service.
- Pass `--stateless` when running behind a load balancer or on autoscaled
  platforms (Cloud Run, etc.): sessions are otherwise held in instance
  memory, and consecutive requests routed to different instances would fail.
- **Breaking change for `--transport sse`** (deprecated): sse is a network
  transport and now behaves like http — authenticated by default and no
  `export_query`. Set `OCEANUM_MCP_AUTH=none` to restore the old
  unauthenticated behavior.

### Serving under an external ASGI server

To run under uvicorn/gunicorn (multiple workers, serverless platforms), use
the packaged app factory — it applies the same auth and tool policy as the
CLI; serving `mcp.http_app()` directly would bypass both:

```bash
uvicorn --factory oceanum_mcp.app:create_http_app --host 0.0.0.0 --port 8000
```

Requests on a network transport never fall back to the server's
`DATAMESH_TOKEN`: an unauthenticated request fails unless
`OCEANUM_MCP_AUTH=none` was set explicitly.

## Datamesh Tools

The intended workflow is: `search_catalog` → `get_datasource_info` → `stage_query`
(dry run: learn the result size without downloading) → `query_data` for small
results inline, or `export_query` to write large results to a file that analysis
code reads directly.

### `search_catalog`

Search the Datamesh catalog with optional text search, time range, and bounding box filters.
Returns a JSON object with `count` and `results`; if `count` equals `limit`, more matches may exist.

| Parameter    | Type        | Description                                      |
| ------------ | ----------- | ------------------------------------------------ |
| `search`     | string      | Text search for name, description, or tags       |
| `time_start` | string      | ISO 8601 start time                              |
| `time_end`   | string      | ISO 8601 end time                                |
| `bbox`       | list[float] | Bounding box `[xmin, ymin, xmax, ymax]` in WGS84 |
| `limit`      | int         | Max results to return (default 20)               |

### `get_datasource_info`

Get full metadata for a datasource including schema, variables, coordinates, and attributes.

| Parameter       | Type   | Description   |
| --------------- | ------ | ------------- |
| `datasource_id` | string | Datasource ID |

### `stage_query`

Dry-run a query on the Datamesh gateway: reports the result size, container
type, and domain length **without downloading any data**, echoes the canonical
query, and recommends the next step (inline query vs export vs narrowing).
Accepts the same query parameters as `query_data`.

### `query_data`

Query a datasource with filters and return small results inline as
coordinate-attributed JSON records with explicit `truncated`/`lazy` flags.
The query is staged first: gridded results above the inline limit are
summarized lazily (structure only); tabular results above the limit are
refused with the staged size and alternatives. Library warnings (e.g. the
2,000,000-row cap on tabular queries) are included in the response.

| Parameter              | Type         | Description                                                        |
| ---------------------- | ------------ | ------------------------------------------------------------------ |
| `datasource_id`        | string       | Datasource to query                                                |
| `variables`            | list[string] | Variables to select                                                |
| `time_start`           | string       | ISO 8601 start of a time range (open-ended if omitted)             |
| `time_end`             | string       | ISO 8601 end of a time range (open-ended if omitted)               |
| `times`                | list[string] | Discrete times (series selection); excludes `time_start`/`time_end`|
| `time_resolution`      | string       | Server-side temporal downsampling (pandas frequency, e.g. `1D`)    |
| `time_resample`        | string       | Resampling method for `time_resolution`: mean, nearest, linear     |
| `bbox`                 | list[float]  | Bounding box `[xmin, ymin, xmax, ymax]`                            |
| `geofilter_feature`    | object       | GeoJSON Feature (Point, MultiPoint, or Polygon) for selection      |
| `geofilter_interp`     | string       | Interpolation for feature selection: nearest or linear             |
| `geofilter_resolution` | float        | Max spatial resolution for downsampling, in CRS units              |
| `level_min`            | float        | Minimum vertical level                                             |
| `level_max`            | float        | Maximum vertical level                                             |
| `levels`               | list[float]  | Discrete vertical levels (series selection)                        |
| `level_interp`         | string       | Interpolation for level series: nearest or linear                  |
| `coord_filters`        | list[object] | Coordinate selections: `[{"coord": "name", "values": [...]}]`      |
| `crs`                  | string/int   | CRS for filter coordinates and returned data                       |
| `aggregate_operations` | list[string] | Aggregation ops: mean, min, max, std, sum                          |
| `aggregate_spatial`    | bool         | Aggregate over spatial dims (default true)                         |
| `aggregate_temporal`   | bool         | Aggregate over temporal dims (default true)                        |
| `limit`                | int          | Max rows to return                                                 |

### `export_query`

Run a query and write the **full** result to a local file — the data-handle
path for results too large to return inline. Gridded datasets stream lazily to
NetCDF; tabular results write Parquet or CSV. Accepts the same query
parameters as `query_data` plus:

| Parameter   | Type   | Description                                                              |
| ----------- | ------ | ------------------------------------------------------------------------ |
| `path`      | string | Destination file path (parent directories are created)                   |
| `format`    | string | `netcdf` (datasets), `parquet` or `csv` (tabular); sensible default      |
| `overwrite` | bool   | Overwrite an existing file (default false)                               |

### `load_datasource`

Summarize an entire datasource. Gridded datasources are opened lazily (no data
download); tabular datasources are downloaded only if under the inline size limit.

| Parameter       | Type   | Description        |
| --------------- | ------ | ------------------ |
| `datasource_id` | string | Datasource to load |

### `update_metadata`

Update metadata on an existing datasource. Only provided fields are changed.
Disabled when the server runs with `OCEANUM_MCP_READ_ONLY` set.

| Parameter       | Type         | Description                  |
| --------------- | ------------ | ---------------------------- |
| `datasource_id` | string       | Datasource to update         |
| `name`          | string       | New name                     |
| `description`   | string       | New description              |
| `tags`          | list[string] | New tags                     |
| `labels`        | list[string] | New labels                   |
| `info`          | object       | Additional metadata object   |
| `details`       | string       | URL for datasource details   |

## Storage Tools

### `list_files`

List files and directories in Oceanum cloud storage.

| Parameter   | Type   | Description                           |
| ----------- | ------ | ------------------------------------- |
| `path`      | string | Directory path to list (default: "/") |
| `recursive` | bool   | List subdirectories recursively       |

### `file_exists`

Check if a file or directory exists in storage.

| Parameter | Type   | Description   |
| --------- | ------ | ------------- |
| `path`    | string | Path to check |

### `read_file`

Read the contents of a text file from storage.

| Parameter | Type   | Description      |
| --------- | ------ | ---------------- |
| `path`    | string | Path to the file |

### `write_file`

Write text content to a file in storage.

| Parameter | Type   | Description           |
| --------- | ------ | --------------------- |
| `path`    | string | Destination path      |
| `content` | string | Text content to write |

### `delete_file`

Delete a file or directory from storage.

| Parameter   | Type   | Description                           |
| ----------- | ------ | ------------------------------------- |
| `path`      | string | Path to delete                        |
| `recursive` | bool   | Delete directory contents recursively |

### `file_info`

Get metadata about a file or directory.

| Parameter | Type   | Description     |
| --------- | ------ | --------------- |
| `path`    | string | Path to inspect |

## Example Workflows

**Discover wave data in the Pacific:**

1. `search_catalog(search="wave", bbox=[120, -50, 180, 10])`
2. `get_datasource_info(datasource_id="some-wave-dataset")`
3. `stage_query(datasource_id="some-wave-dataset", variables=["Hs", "Tp"], time_start="2024-01-01", time_end="2024-01-31")` to check the result size
4. `query_data(...)` with the same parameters if small, or `export_query(..., path="waves.nc")` if large

**Shrink a 40-year hourly time series to something inline-sized:**

1. `stage_query(datasource_id="hindcast", variables=["Hs"], time_start="1984-01-01", time_end="2024-01-01")` — too large
2. `query_data(..., time_resolution="1MS", time_resample="mean")` — monthly means, small enough to return inline

**Browse and read files in cloud storage:**

1. `list_files(path="/")` to see top-level contents
2. `list_files(path="/my-project", recursive=True)` to drill down
3. `read_file(path="/my-project/config.json")` to read a file

**Get a quick summary of a dataset:**

1. `get_datasource_info(datasource_id="my-dataset")` to see variables and time range
2. `query_data(datasource_id="my-dataset", limit=10)` to preview the data
