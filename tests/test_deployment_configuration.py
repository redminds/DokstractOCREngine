from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


def test_repository_has_one_environment_file():
    env_files = [
        path
        for path in REPOSITORY_ROOT.rglob("*")
        if path.is_file()
        and (path.name == ".env" or ".env." in path.name or path.name.endswith(".env"))
        and ".git" not in path.parts
        and "workspace_tmp_tests" not in path.parts
    ]

    assert [path.relative_to(REPOSITORY_ROOT).as_posix() for path in env_files] == ["docker/.env"]


def test_compose_scopes_data_mount_and_shared_network():
    compose = _read("docker/docker-compose.yml")
    env = _read("docker/.env")

    assert "../.data/${DEPLOY_ENV:?DEPLOY_ENV is required}:/app/.data" in compose
    assert "ports:" not in compose
    assert "ENGINE_HOST_PORT" not in env
    assert "name: ${DOKSTRACT_SHARED_NETWORK:?DOKSTRACT_SHARED_NETWORK is required}" in compose
    assert "APP_ENV=${DEPLOY_ENV}" in env
    assert "COMPOSE_PROJECT_NAME=${ENGINE_SERVICE_NAME}-${DEPLOY_ENV}" in env
    assert "DOKSTRACT_SHARED_NETWORK=dokstract-shared-${DEPLOY_ENV}" in env
    assert "PLATFORM_API_URL=http://platform-api:8000" in env
    assert "ENGINE_CORS_ORIGINS=*" in env
    assert "OCR_ENGINE_ENABLE_SECONDARY_NETWORK=false" in env
    assert "OCR_ENGINE_SECONDARY_NETWORK=" in env
    assert "ENGINE_REGISTRY_DB_PATH=/app/.data/ocr-engine-registry.db" in env
    assert "PLATFORM_MYSQL_VOLUME_NAME" not in env
    assert "ENGINE_HEALTH_WAIT_TIMEOUT_SECONDS" not in env
    assert "ENGINE_HEALTH_START_PERIOD_SECONDS" not in env
    assert "ENGINE_HEALTH_INTERVAL_SECONDS" not in env
    assert "ENGINE_HEALTH_TIMEOUT_SECONDS" not in env
    assert "ENGINE_HEALTH_RETRIES" not in env


def test_deployment_script_requires_explicit_environment():
    script = _read("scripts/jenkins-deploy.sh")

    assert 'DEPLOY_ENV="${DEPLOY_ENV:-}"' in script
    assert 'DEPLOY_ROOT="${DEPLOY_ROOT:-/opt/dokstract/$DEPLOY_ENV/$APP_NAME}"' in script
    assert 'CONFIG_ROOT="${CONFIG_ROOT:-/etc/dokstract/$DEPLOY_ENV/$APP_NAME}"' in script
    assert "DEPLOY_ENV must be set explicitly" in script
