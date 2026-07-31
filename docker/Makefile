.PHONY: \
	help \
	preflight \
	config \
	build \
	up \
	down \
	restart \
	logs \
	ps \
	health \
	test-deploy \
	deploy

SHELL := /bin/bash
.SHELLFLAGS := -Eeuo pipefail -c

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

# Absolute directory containing this Makefile:
#   <repository-root>/docker
MAKEFILE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# Absolute repository root:
#   <repository-root>
REPO_ROOT := $(abspath $(MAKEFILE_DIR)/..)

# Defaults for local usage.
# Jenkins can override ENV_FILE with an absolute server-managed path:
#
#   make -C docker \
#     ENV_FILE=/etc/dokstract/dev/DokstractOCREngine/.env \
#     deploy
#
ENV_FILE ?= .env
COMPOSE_FILE ?= docker-compose.yml
SECONDARY_COMPOSE_FILE ?= docker-compose.secondary-network.yml

# Preserve absolute paths.
# Resolve relative paths against the directory containing this Makefile.
resolve_path = $(if $(filter /%,$(1)),$(1),$(abspath $(MAKEFILE_DIR)/$(1)))

RESOLVED_ENV_FILE := $(call resolve_path,$(ENV_FILE))
RESOLVED_COMPOSE_FILE := $(call resolve_path,$(COMPOSE_FILE))
RESOLVED_SECONDARY_COMPOSE_FILE := $(call resolve_path,$(SECONDARY_COMPOSE_FILE))

# Exporting this variable so the Compose file can use it for service
# environment configuration if required.
SERVICE_ENV_FILE ?= $(RESOLVED_ENV_FILE)
export SERVICE_ENV_FILE

# Ensure ${ENV_FILE} inside docker-compose.yml resolves to the authoritative
# environment file path.
export ENV_FILE := $(RESOLVED_ENV_FILE)

# -----------------------------------------------------------------------------
# Optional secondary network
# -----------------------------------------------------------------------------

# Read the secondary-network flag from the selected environment file.
# Missing or blank means disabled.
SECONDARY_NETWORK_ENABLED := $(strip $(shell \
	if [ -r "$(RESOLVED_ENV_FILE)" ]; then \
		grep -E '^OCR_ENGINE_ENABLE_SECONDARY_NETWORK=' \
			"$(RESOLVED_ENV_FILE)" 2>/dev/null \
			| tail -n 1 \
			| cut -d= -f2- \
			| tr '[:upper:]' '[:lower:]' \
			| xargs; \
	fi \
))

ifeq ($(SECONDARY_NETWORK_ENABLED),true)
COMPOSE_FILE_ARGS := \
	-f "$(RESOLVED_COMPOSE_FILE)" \
	-f "$(RESOLVED_SECONDARY_COMPOSE_FILE)"
else
COMPOSE_FILE_ARGS := \
	-f "$(RESOLVED_COMPOSE_FILE)"
endif

# -----------------------------------------------------------------------------
# Docker Compose configuration
# -----------------------------------------------------------------------------

# COMPOSE_PROJECT_NAME is loaded by Docker Compose from RESOLVED_ENV_FILE.
COMPOSE_SERVICE ?= ocr-engine

WAIT_TIMEOUT_SECONDS ?= 600
WAIT_INTERVAL_SECONDS ?= 5
LOG_TAIL_LINES ?= 100

DOCKER_COMPOSE = docker compose \
	--env-file "$(RESOLVED_ENV_FILE)" \
	$(COMPOSE_FILE_ARGS)

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------

help:
	@printf '%s\n' \
		'Dokstract OCR Engine Docker targets' \
		'' \
		'Usage:' \
		'  make -C docker <target>' \
		'' \
		'Jenkins usage:' \
		'  make -C docker ENV_FILE=/etc/dokstract/dev/DokstractOCREngine/.env deploy' \
		'' \
		'Targets:' \
		'  help         Show this help message' \
		'  preflight    Validate required commands and files' \
		'  config       Validate and render the Compose configuration' \
		'  build        Build the OCR Engine image' \
		'  up           Start the OCR Engine service' \
		'  down         Stop the OCR Engine Compose project' \
		'  restart      Restart the OCR Engine service' \
		'  logs         Follow OCR Engine logs' \
		'  ps           Show Compose service status' \
		'  health       Wait until the OCR Engine container is healthy' \
		'  deploy       Validate, build, start, and verify the service' \
		'  test-deploy  Alias for deploy' \
		'' \
		'Optional secondary network:' \
		'  OCR_ENGINE_ENABLE_SECONDARY_NETWORK=true' \
		'  OCR_ENGINE_SECONDARY_NETWORK=dokstract-shared-test' \
		'' \
		'Resolved defaults:' \
		'  Environment file:          $(RESOLVED_ENV_FILE)' \
		'  Primary Compose file:      $(RESOLVED_COMPOSE_FILE)' \
		'  Secondary network enabled: $(if $(SECONDARY_NETWORK_ENABLED),$(SECONDARY_NETWORK_ENABLED),false)' \
		'  Compose project:           loaded from $(RESOLVED_ENV_FILE)' \
		'  Compose service:           $(COMPOSE_SERVICE)'

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

preflight:
	@command -v docker >/dev/null 2>&1 || { \
		printf '%s\n' \
			'[preflight] Docker is not installed or is unavailable in PATH.' >&2; \
		exit 1; \
	}
	@docker compose version >/dev/null 2>&1 || { \
		printf '%s\n' \
			'[preflight] Docker Compose v2 is unavailable.' >&2; \
		exit 1; \
	}
	@test -f "$(RESOLVED_ENV_FILE)" || { \
		printf '%s\n' \
			'[preflight] Environment file not found:' \
			'  $(RESOLVED_ENV_FILE)' >&2; \
		exit 1; \
	}
	@test -r "$(RESOLVED_ENV_FILE)" || { \
		printf '%s\n' \
			'[preflight] Environment file is not readable:' \
			'  $(RESOLVED_ENV_FILE)' >&2; \
		exit 1; \
	}
	@test -f "$(RESOLVED_COMPOSE_FILE)" || { \
		printf '%s\n' \
			'[preflight] Compose file not found:' \
			'  $(RESOLVED_COMPOSE_FILE)' >&2; \
		exit 1; \
	}
	@test -r "$(RESOLVED_COMPOSE_FILE)" || { \
		printf '%s\n' \
			'[preflight] Compose file is not readable:' \
			'  $(RESOLVED_COMPOSE_FILE)' >&2; \
		exit 1; \
	}
	@case "$(SECONDARY_NETWORK_ENABLED)" in \
		''|false) \
			;; \
		true) \
			test -f "$(RESOLVED_SECONDARY_COMPOSE_FILE)" || { \
				printf '%s\n' \
					'[preflight] Secondary Compose file not found:' \
					'  $(RESOLVED_SECONDARY_COMPOSE_FILE)' >&2; \
				exit 1; \
			}; \
			test -r "$(RESOLVED_SECONDARY_COMPOSE_FILE)" || { \
				printf '%s\n' \
					'[preflight] Secondary Compose file is not readable:' \
					'  $(RESOLVED_SECONDARY_COMPOSE_FILE)' >&2; \
				exit 1; \
			}; \
			grep -qE '^OCR_ENGINE_SECONDARY_NETWORK=.+$$' \
				"$(RESOLVED_ENV_FILE)" || { \
				printf '%s\n' \
					'[preflight] OCR_ENGINE_SECONDARY_NETWORK is required when' \
					'            OCR_ENGINE_ENABLE_SECONDARY_NETWORK=true.' >&2; \
				exit 1; \
			} \
			;; \
		*) \
			printf '%s\n' \
				'[preflight] OCR_ENGINE_ENABLE_SECONDARY_NETWORK must be true or false.' \
				"            Current value: $(SECONDARY_NETWORK_ENABLED)" >&2; \
			exit 1 \
			;; \
	esac
	@printf '%s\n' \
		"[preflight] Environment file: $(RESOLVED_ENV_FILE)" \
		"[preflight] Primary Compose file: $(RESOLVED_COMPOSE_FILE)" \
		"[preflight] Secondary network enabled: $(if $(SECONDARY_NETWORK_ENABLED),$(SECONDARY_NETWORK_ENABLED),false)" \
		"[preflight] Compose project: loaded from $(RESOLVED_ENV_FILE)" \
		"[preflight] Compose service: $(COMPOSE_SERVICE)"

# -----------------------------------------------------------------------------
# Compose operations
# -----------------------------------------------------------------------------

config: preflight
	$(DOCKER_COMPOSE) config

build: preflight
	$(DOCKER_COMPOSE) build "$(COMPOSE_SERVICE)"

up: preflight
	$(DOCKER_COMPOSE) up \
		-d \
		--remove-orphans \
		--no-build \
		"$(COMPOSE_SERVICE)"

down: preflight
	$(DOCKER_COMPOSE) down --remove-orphans

restart: preflight
	$(DOCKER_COMPOSE) down --remove-orphans
	$(DOCKER_COMPOSE) up \
		-d \
		--remove-orphans \
		--no-build \
		"$(COMPOSE_SERVICE)"

logs: preflight
	$(DOCKER_COMPOSE) logs \
		-f \
		--tail "$(LOG_TAIL_LINES)" \
		"$(COMPOSE_SERVICE)"

ps: preflight
	$(DOCKER_COMPOSE) ps

# -----------------------------------------------------------------------------
# Health verification
# -----------------------------------------------------------------------------

health: preflight
	@container_id="$$( $(DOCKER_COMPOSE) ps -q "$(COMPOSE_SERVICE)" )"; \
	if [[ -z "$$container_id" ]]; then \
		printf '%s\n' \
			'[health] OCR Engine container is not running.' >&2; \
		$(DOCKER_COMPOSE) ps >&2 || true; \
		exit 1; \
	fi; \
	printf '%s\n' \
		"[health] Container ID: $$container_id" \
		"[health] Waiting up to $(WAIT_TIMEOUT_SECONDS) seconds for OCR Engine health..."; \
	deadline=$$(( $$(date +%s) + $(WAIT_TIMEOUT_SECONDS) )); \
	while true; do \
		running="$$( \
			docker inspect \
				-f '{{.State.Running}}' \
				"$$container_id" \
				2>/dev/null || true \
		)"; \
		status="$$( \
			docker inspect \
				-f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' \
				"$$container_id" \
				2>/dev/null || true \
		)"; \
		if [[ "$$running" != 'true' ]]; then \
			printf '%s\n' \
				'[health] OCR Engine container stopped before becoming healthy.' >&2; \
			$(DOCKER_COMPOSE) ps >&2 || true; \
			$(DOCKER_COMPOSE) logs \
				--tail "$(LOG_TAIL_LINES)" \
				"$(COMPOSE_SERVICE)" >&2 || true; \
			exit 1; \
		fi; \
		case "$$status" in \
			healthy) \
				printf '%s\n' \
					'[health] OCR Engine container is healthy.'; \
				exit 0 \
				;; \
			unhealthy) \
				printf '%s\n' \
					'[health] OCR Engine container reported unhealthy.' >&2; \
				$(DOCKER_COMPOSE) ps >&2 || true; \
				$(DOCKER_COMPOSE) logs \
					--tail "$(LOG_TAIL_LINES)" \
					"$(COMPOSE_SERVICE)" >&2 || true; \
				exit 1 \
				;; \
			none) \
				printf '%s\n' \
					'[health] Container has no Docker health status yet.' \
				;; \
			*) \
				printf '%s\n' \
					"[health] Current status: $$status" \
				;; \
		esac; \
		if (( $$(date +%s) >= deadline )); then \
			printf '%s\n' \
				'[health] Timed out waiting for OCR Engine to become healthy.' >&2; \
			$(DOCKER_COMPOSE) ps >&2 || true; \
			$(DOCKER_COMPOSE) logs \
				--tail "$(LOG_TAIL_LINES)" \
				"$(COMPOSE_SERVICE)" >&2 || true; \
			exit 1; \
		fi; \
		sleep "$(WAIT_INTERVAL_SECONDS)"; \
	done

# -----------------------------------------------------------------------------
# Deployment
# -----------------------------------------------------------------------------

deploy: preflight
	@printf '%s\n' '[deploy] Validating Docker Compose configuration'
	$(DOCKER_COMPOSE) config >/dev/null
	@printf '%s\n' '[deploy] Building OCR Engine image'
	$(DOCKER_COMPOSE) build "$(COMPOSE_SERVICE)"
	@printf '%s\n' '[deploy] Starting OCR Engine service'
	$(DOCKER_COMPOSE) up \
		-d \
		--remove-orphans \
		--no-build \
		"$(COMPOSE_SERVICE)"
	@printf '%s\n' '[deploy] Waiting for OCR Engine health'
	$(MAKE) health \
		ENV_FILE="$(RESOLVED_ENV_FILE)" \
		COMPOSE_FILE="$(RESOLVED_COMPOSE_FILE)" \
		SECONDARY_COMPOSE_FILE="$(RESOLVED_SECONDARY_COMPOSE_FILE)" \
		COMPOSE_SERVICE="$(COMPOSE_SERVICE)" \
		WAIT_TIMEOUT_SECONDS="$(WAIT_TIMEOUT_SECONDS)" \
		WAIT_INTERVAL_SECONDS="$(WAIT_INTERVAL_SECONDS)" \
		LOG_TAIL_LINES="$(LOG_TAIL_LINES)"
	@printf '%s\n' '[deploy] OCR Engine deployment completed successfully'
	$(DOCKER_COMPOSE) ps

test-deploy: deploy