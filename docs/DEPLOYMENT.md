# Deployment

## Container

Build and run with Docker Compose:

```bash
docker compose --env-file docker/.env -f docker/docker-compose.yml up --build
```

Image builds use `requirements.lock.txt` for pinned dependencies.

## Important environment variables

- `ENGINE_INTERNAL_TOKEN`
- `ENGINE_ADMIN_TOKEN`
- `ENGINE_REGISTRY_DB_PATH`
- `ENGINE_DEFAULT_RELEASE_TAG`
- `ENGINE_DEFAULT_IMAGE_DIGEST`
- `ENGINE_DEFAULT_SUPPORTED_API_VERSIONS`
- `ENGINE_DEFAULT_CAPABILITIES`

## Port

Default HTTP port is `8010`.
