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

| Variable           | Required | Description                                                          |
| ------------------ | -------- | -------------------------------------------------------------------- |
| `DATAMESH_TOKEN`   | Yes      | Oceanum API token (shared by all servers)                            |
| `DATAMESH_SERVICE` | No       | Custom datamesh service URL (default: `https://datamesh.oceanum.io`) |
| `STORAGE_SERVICE`  | No       | Custom storage service URL (default: `https://storage.oceanum.io`)   |
| `OCEANUM_DOMAIN`   | No       | Override the base domain for all services (default: `oceanum.io`)    |

## Datamesh Tools

### `search_catalog`

Search the Datamesh catalog with optional text search, time range, and bounding box filters.

| Parameter    | Type        | Description                                      |
| ------------ | ----------- | ------------------------------------------------ |
| `search`     | string      | Text search for name, description, or tags       |
| `time_start` | string      | ISO 8601 start time                              |
| `time_end`   | string      | ISO 8601 end time                                |
| `bbox`       | list[float] | Bounding box `[xmin, ymin, xmax, ymax]` in WGS84 |
| `limit`      | int         | Max results to return                            |

### `get_datasource_info`

Get full metadata for a datasource including schema, variables, coordinates, and attributes.

| Parameter       | Type   | Description   |
| --------------- | ------ | ------------- |
| `datasource_id` | string | Datasource ID |

### `query_data`

Query a datasource with filters. Returns data summary for large results, full data for small ones.

| Parameter              | Type         | Description                                        |
| ---------------------- | ------------ | -------------------------------------------------- |
| `datasource_id`        | string       | Datasource to query                                |
| `variables`            | list[string] | Variables to select                                |
| `time_start`           | string       | ISO 8601 start time                                |
| `time_end`             | string       | ISO 8601 end time                                  |
| `bbox`                 | list[float]  | Bounding box `[xmin, ymin, xmax, ymax]`            |
| `geofilter_geojson`    | string       | GeoJSON Feature for spatial filtering              |
| `level_min`            | float        | Minimum vertical level                             |
| `level_max`            | float        | Maximum vertical level                             |
| `coord_filters`        | string       | JSON array of `{"coord": "name", "values": [...]}` |
| `aggregate_operations` | list[string] | Aggregation ops: mean, min, max, std, sum          |
| `aggregate_spatial`    | bool         | Aggregate over spatial dims (default true)         |
| `aggregate_temporal`   | bool         | Aggregate over temporal dims (default true)        |
| `limit`                | int          | Max rows to return                                 |

### `load_datasource`

Load an entire datasource. Best for small datasets.

| Parameter       | Type   | Description        |
| --------------- | ------ | ------------------ |
| `datasource_id` | string | Datasource to load |

### `update_metadata`

Update metadata on an existing datasource. Only provided fields are changed.

| Parameter       | Type         | Description                  |
| --------------- | ------------ | ---------------------------- |
| `datasource_id` | string       | Datasource to update         |
| `name`          | string       | New name                     |
| `description`   | string       | New description              |
| `tags`          | list[string] | New tags                     |
| `labels`        | list[string] | New labels                   |
| `info`          | string       | JSON string of metadata dict |
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
3. `query_data(datasource_id="some-wave-dataset", variables=["Hs", "Tp"], time_start="2024-01-01", time_end="2024-01-31")`

**Browse and read files in cloud storage:**

1. `list_files(path="/")` to see top-level contents
2. `list_files(path="/my-project", recursive=True)` to drill down
3. `read_file(path="/my-project/config.json")` to read a file

**Get a quick summary of a dataset:**

1. `get_datasource_info(datasource_id="my-dataset")` to see variables and time range
2. `query_data(datasource_id="my-dataset", limit=10)` to preview the data
