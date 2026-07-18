#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$REPO_ROOT/docker"
ENV_FILE="$DOCKER_DIR/.env"
COMPOSE_FILE="$DOCKER_DIR/docker-compose.yml"
COMPOSE_PROJECT_NAME="dokstract-ocr-engine"
COMPOSE_SERVICE="ocr-engine"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-180}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-5}"

fail() {
  printf '%s\n' "ERROR: $*" >&2
  exit 1
}

log() {
  printf '%s\n' "[jenkins] $*"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

require_file() {
  [[ -f "$1" ]] || fail "Missing required file: $1"
}

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "Missing required environment variable: $name"
}

diagnose_failure() {
  local exit_code="$1"
  local line_no="$2"

  trap - ERR
  printf '%s\n' "[jenkins] deployment failed at line ${line_no} (exit ${exit_code})." >&2
  if command -v docker >/dev/null 2>&1; then
    printf '%s\n' '[jenkins] docker compose ps:' >&2
    docker compose --env-file "$ENV_FILE" -p "$COMPOSE_PROJECT_NAME" -f "$COMPOSE_FILE" ps >&2 || true
    local container_id
    container_id="$(docker compose --env-file "$ENV_FILE" -p "$COMPOSE_PROJECT_NAME" -f "$COMPOSE_FILE" ps -q "$COMPOSE_SERVICE" 2>/dev/null || true)"
    if [[ -n "$container_id" ]]; then
      printf '%s\n' '[jenkins] container health:' >&2
      docker inspect -f 'name={{.Name}} status={{.State.Status}} health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$container_id" >&2 || true
    fi
    printf '%s\n' '[jenkins] recent service logs:' >&2
    docker compose --env-file "$ENV_FILE" -p "$COMPOSE_PROJECT_NAME" -f "$COMPOSE_FILE" logs --tail 100 "$COMPOSE_SERVICE" >&2 || true
  fi
  exit "$exit_code"
}

trap 'diagnose_failure $? $LINENO' ERR

log "Resolving engine deployment paths"
log "Repo root: $REPO_ROOT"
log "Env file: $ENV_FILE"
log "Compose file: $COMPOSE_FILE"

require_command docker
require_file "$ENV_FILE"
require_file "$COMPOSE_FILE"
require_file "$REPO_ROOT/Makefile"
require_file "$DOCKER_DIR/Makefile"
require_file "$DOCKER_DIR/Dockerfile"
[[ -d "$REPO_ROOT/app" ]] || fail "Missing required directory: $REPO_ROOT/app"

set -a
source "$ENV_FILE"
set +a

for name in \
  APP_ENV \
  ENGINE_SERVICE_NAME \
  ENGINE_HOST \
  ENGINE_PORT \
  ENGINE_HOST_PORT \
  ENGINE_INTERNAL_TOKEN \
  ENGINE_ADMIN_TOKEN \
  ENGINE_REGISTRY_DB_PATH \
  ENGINE_DEFAULT_RELEASE_TAG \
  ENGINE_DEFAULT_IMAGE_DIGEST \
  ENGINE_DEFAULT_SUPPORTED_API_VERSIONS \
  ENGINE_DEFAULT_CAPABILITIES \
  ENGINE_CORS_ORIGINS \
  OCR_LANG \
  OCR_CPU_THREADS \
  OCR_DET_LIMIT_SIDE_LEN \
  OCR_TEXT_BATCH_SIZE \
  ENGINE_MAX_INFLIGHT_REQUESTS
do
  require_env "$name"
done

log "Validating compose configuration"
docker compose --env-file "$ENV_FILE" -p "$COMPOSE_PROJECT_NAME" -f "$COMPOSE_FILE" config >/dev/null

log "Building OCR Engine image"
docker compose --env-file "$ENV_FILE" -p "$COMPOSE_PROJECT_NAME" -f "$COMPOSE_FILE" build "$COMPOSE_SERVICE"

log "Deploying OCR Engine service"
docker compose --env-file "$ENV_FILE" -p "$COMPOSE_PROJECT_NAME" -f "$COMPOSE_FILE" up -d --remove-orphans --no-build "$COMPOSE_SERVICE"

log "Waiting for OCR Engine health"
deadline=$(( $(date +%s) + WAIT_TIMEOUT_SECONDS ))
while :; do
  container_id="$(docker compose --env-file "$ENV_FILE" -p "$COMPOSE_PROJECT_NAME" -f "$COMPOSE_FILE" ps -q "$COMPOSE_SERVICE")"
  [[ -n "$container_id" ]] || fail "OCR Engine container is not running."

  health_status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$container_id" 2>/dev/null || true)"
  if [[ "$health_status" == "healthy" ]]; then
    break
  fi

  if [[ "$health_status" == "unhealthy" ]]; then
    fail "OCR Engine container reported unhealthy."
  fi

  if [[ $(date +%s) -ge "$deadline" ]]; then
    fail "Timed out waiting for OCR Engine to become healthy."
  fi

  sleep "$WAIT_INTERVAL_SECONDS"
done

log "Deployment status"
docker compose --env-file "$ENV_FILE" -p "$COMPOSE_PROJECT_NAME" -f "$COMPOSE_FILE" ps

log "Deployment completed successfully"
