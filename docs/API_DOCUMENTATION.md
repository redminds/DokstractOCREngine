# API Documentation

## Internal API

### `GET /api/v1/internal/releases/current`

Returns the current active engine release.

### `POST /api/v1/internal/projects/{project_id}/dependencies/request`

Requests engine capability support for a project.

Example:

```json
{
  "capability": "ocr",
  "api_version": "v1"
}
```

## Admin API

### `GET /api/v1/admin/registry`

Returns the registry snapshot.

### `POST /api/v1/admin/releases`

Registers an immutable release tag, image digest, supported API versions, and capabilities.

### `POST /api/v1/admin/releases/{release_tag}/validate`

Records dev/CI validation status for a release.

### `POST /api/v1/admin/releases/{release_tag}/promote`

Atomically switches the active release and reassigns compatible projects after validation succeeds.

### `POST /api/v1/admin/releases/rollback`

Restores the previous known-good release and its project mappings.

### `GET /api/v1/admin/audit-log`

Returns recorded audit and release actions.
