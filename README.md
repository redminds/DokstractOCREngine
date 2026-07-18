# Dokstract OCR Engine Service

This service manages immutable, versioned OCR engine releases and the dependency registry for Dokstract projects.

## What it does

- stores OCR engine releases as tagged Docker image releases
- tracks which project depends on which engine release tag
- blocks unsafe promotions when active-release compatibility would be broken
- supports admin-approved promotion and rollback
- records validation, audit, and rollback history
- exposes internal and admin APIs for orchestration

## Runtime model

The service is container-based and keeps its registry in SQLite under the configured data directory.
The container includes PaddleOCR and PaddlePaddle at build time from a pinned lockfile.
Runtime package installation is intentionally not used.

## API summary

- `GET /health`
- `GET /api/v1/internal/releases/current`
- `POST /api/v1/internal/projects/{project_id}/dependencies/request`
- `GET /api/v1/admin/registry`
- `POST /api/v1/admin/releases`
- `POST /api/v1/admin/releases/{release_tag}/validate`
- `POST /api/v1/admin/releases/{release_tag}/promote`
- `POST /api/v1/admin/releases/rollback`
- `GET /api/v1/admin/audit-log`

## Local run

```bash
make up
```

## Notes

- Internal routes use `X-Service-Name` and `X-Service-Token` for service auth, and `X-Engine-Admin-Token` for admin routes.
- Admin routes use `X-Engine-Admin-Token`.
- The registry records manual intervention requests, audit events, and release history.
