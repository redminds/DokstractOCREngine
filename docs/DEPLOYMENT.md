# Deployment

## Prerequisites

- Docker Engine 24+, Docker Compose v2
- **DokstractDependencies** deployed first (for the environment-specific shared network)

## Required Environment File

- `docker/.env`
- Deployed environments use an externally managed env file passed through `ENV_FILE`.

## Canonical Jenkins Script

- `scripts/jenkins-deploy.sh`

## Build Command

```bash
make build
```

## Validation Command

```bash
make config
```

or

```bash
docker compose --env-file docker/.env -f docker/docker-compose.yml config
```

## Deploy Command

```bash
make deploy
```

## Health Check Command

```bash
make health
```

The deployment gate waits for the container healthcheck, which targets `GET /health/ready`.

## Log Command

```bash
make logs
```

## Status Command

```bash
make ps
```

## Shutdown Command

```bash
make down
```

## Deployment Order

```
1. DokstractDependencies
2. DokstractPlatformAPI
3. DokstractOCREngine  ← this project
4. DokstractSchemaAPI
5. DokstractAdminUI
```

## Important Environment Variables

| Variable | Purpose |
|----------|---------|
| `APP_ENV` | Runtime environment label |
| `ENGINE_SERVICE_NAME` | Stable engine service identity |
| `ENGINE_HOST` | In-container bind host |
| `ENGINE_PORT` | In-container API port |
| `ENGINE_HOST_PORT` | Host-published port |
| `PLATFORM_API_URL` | Platform API base URL for OCR execution reporting |
| `OCR_ENGINE_OCR_API_TOKEN` | OCR API caller token for OCR Engine routes |
| `OCR_ENGINE_SCHEMA_API_TOKEN` | Schema API caller token for OCR Engine routes |
| `PLATFORM_API_OCR_ENGINE_TOKEN` | Platform caller token for OCR execution reporting |
| `ENGINE_ADMIN_TOKEN` | Admin auth token |
| `ENGINE_CORS_ORIGINS` | Allowed browser origins |
| `ENGINE_REGISTRY_DB_PATH` | Path to engine registry DB |
| `ENGINE_DEFAULT_RELEASE_TAG` | Active release tag |
| `ENGINE_DEFAULT_IMAGE_DIGEST` | Default immutable release digest |
| `ENGINE_DEFAULT_SUPPORTED_API_VERSIONS` | Supported API versions |
| `ENGINE_DEFAULT_CAPABILITIES` | Comma-separated capabilities |
| `ENGINE_MAX_INFLIGHT_REQUESTS` | Concurrency limiter |
| `OCR_*` variables | OCR runtime tuning |

## Common Failures

| Failure | Cause | Fix |
|---------|-------|-----|
| `ENGINE_HOST_PORT is required` | Missing env var | Set in `docker/.env` |
| Port 8010 already in use | Port collision | Stop old container or change port |
| Shared network not found | Dependencies not deployed | Deploy the matching environment's Dokstract shared network |
| Health check never becomes ready | Service booted but did not pass readiness | Review `make logs` output and the `/health/ready` response |

## Values That Change by Environment

| Variable | dev | prod |
|----------|-----|------|
| `OCR_ENGINE_OCR_API_TOKEN` | (per env) | **Strong secret** |
| `OCR_ENGINE_SCHEMA_API_TOKEN` | (per env) | **Strong secret** |
| `PLATFORM_API_OCR_ENGINE_TOKEN` | (per env) | **Strong secret** |
| `ENGINE_ADMIN_TOKEN` | (per env) | **Strong secret** |
| `ENGINE_HOST_PORT` | 8010 | 8011 |
