
#!/bin/sh
set -eu

APP_NAME="DokstractOCREngine"
DEPLOY_ENV="${DEPLOY_ENV:-}"

# Jenkins GitHub checkout directory.
SOURCE_ROOT="$(pwd)"

# Stable application and configuration locations.
DEPLOY_ROOT="${DEPLOY_ROOT:-/opt/dokstract/$DEPLOY_ENV/$APP_NAME}"
CONFIG_ROOT="${CONFIG_ROOT:-/etc/dokstract/$DEPLOY_ENV/$APP_NAME}"
ENV_FILE="${ENV_FILE:-$CONFIG_ROOT/.env}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

log() {
  echo "[deploy] $*"
}

[ -n "$DEPLOY_ENV" ] \
  || fail "DEPLOY_ENV must be set explicitly (for example: dev, test, or prod)"

log "Source Root Path: $SOURCE_ROOT"


[ -d "$SOURCE_ROOT" ] \
  || fail "Jenkins source directory not found: $SOURCE_ROOT"

[ "$SOURCE_ROOT" != "$DEPLOY_ROOT" ] \
  || fail "Source and deployment directories must be different: $SOURCE_ROOT"

[ -d "$CONFIG_ROOT" ] \
  || fail "Configuration directory not found: $CONFIG_ROOT"

[ -f "$ENV_FILE" ] \
  || fail "Runtime environment file not found: $ENV_FILE"

for command_name in rsync docker make; do
  command -v "$command_name" >/dev/null 2>&1 \
    || fail "Required command not found: $command_name"
done

docker compose version >/dev/null 2>&1 \
  || fail "Docker Compose v2 is required"

required_source_files="
docker/Makefile
docker/Dockerfile
docker/docker-compose.yml
app/main.py
app/core/config.py
"

for rel in $required_source_files; do
  [ -f "$SOURCE_ROOT/$rel" ] \
    || fail "Missing required source file: $SOURCE_ROOT/$rel"
done

mkdir -p "$DEPLOY_ROOT"

log "Syncing $APP_NAME source"

rsync -a --delete \
  --exclude '.git/' \
  --exclude '.gitignore' \
  --exclude '.github/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  --exclude '.env' \
  --exclude '.env.*' \
  --exclude 'docker/.env' \
  --exclude 'docker/.env.*' \
  --exclude '.venv/' \
  --exclude '.jenkins-venv/' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '.cache/' \
  --exclude '.data/' \
  --exclude 'logs/' \
  --exclude 'tmp/' \
  --exclude '*.bak' \
  --exclude '*.bak.*' \
  --exclude '*.old' \
  --exclude '*~' \
  "$SOURCE_ROOT/" "$DEPLOY_ROOT/"

for rel in $required_source_files; do
  [ -f "$DEPLOY_ROOT/$rel" ] \
    || fail "Missing deployed file after sync: $DEPLOY_ROOT/$rel"
done

chmod 600 "$ENV_FILE"

log "Deploying $APP_NAME"

make -C "$DEPLOY_ROOT/docker" ENV_FILE="$ENV_FILE" deploy

log "$APP_NAME deployment completed"
