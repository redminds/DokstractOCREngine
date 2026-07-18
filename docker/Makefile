.PHONY: help config build up down restart logs ps health test-deploy deploy

SHELL := /bin/bash
.SHELLFLAGS := -Eeuo pipefail -c

REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
COMPOSE_FILE ?= docker-compose.yml
ENV_FILE ?= .env
COMPOSE_PROJECT_NAME ?= dokstract-ocr-engine
COMPOSE_SERVICE ?= ocr-engine
DOCKER_COMPOSE = docker compose --env-file "$(REPO_ROOT)/$(ENV_FILE)" -p "$(COMPOSE_PROJECT_NAME)" -f "$(REPO_ROOT)/$(COMPOSE_FILE)"
WAIT_TIMEOUT_SECONDS ?= 180
WAIT_INTERVAL_SECONDS ?= 5

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make -C docker config      - validate the compose file and env file' \
		'  make -C docker build       - build the OCR Engine image' \
		'  make -C docker up          - start the OCR Engine compose service' \
		'  make -C docker down        - stop the OCR Engine compose service' \
		'  make -C docker restart     - restart the OCR Engine compose service' \
		'  make -C docker logs        - follow OCR Engine logs' \
		'  make -C docker ps          - show OCR Engine status' \
		'  make -C docker health      - wait for OCR Engine health' \
		'  make -C docker deploy      - validate, build, deploy, and wait for health' \
		'  make -C docker test-deploy - alias for deploy'

config:
	$(DOCKER_COMPOSE) config

build:
	$(DOCKER_COMPOSE) build $(COMPOSE_SERVICE)

up:
	$(DOCKER_COMPOSE) up -d --remove-orphans --no-build $(COMPOSE_SERVICE)

down:
	$(DOCKER_COMPOSE) down --remove-orphans

restart:
	$(DOCKER_COMPOSE) down --remove-orphans
	$(DOCKER_COMPOSE) up -d --remove-orphans --no-build $(COMPOSE_SERVICE)

logs:
	$(DOCKER_COMPOSE) logs -f --tail 100 $(COMPOSE_SERVICE)

ps:
	$(DOCKER_COMPOSE) ps

health:
	@container_id="$$( $(DOCKER_COMPOSE) ps -q $(COMPOSE_SERVICE) )"; \
	if [[ -z "$$container_id" ]]; then \
		printf '%s\n' '[health] OCR Engine container is not running.' >&2; \
		exit 1; \
	fi; \
	deadline=$$(( $$(date +%s) + $(WAIT_TIMEOUT_SECONDS) )); \
	while :; do \
		status="$$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$$container_id" 2>/dev/null || true)"; \
		case "$$status" in \
			healthy) \
				printf '%s\n' "[health] OCR Engine container is healthy."; \
				exit 0 ;; \
			unhealthy) \
				printf '%s\n' '[health] OCR Engine container reported unhealthy.' >&2; \
				$(DOCKER_COMPOSE) ps >&2 || true; \
				$(DOCKER_COMPOSE) logs --tail 100 $(COMPOSE_SERVICE) >&2 || true; \
				exit 1 ;; \
		esac; \
		if [[ $$(date +%s) -ge $$deadline ]]; then \
			printf '%s\n' '[health] Timed out waiting for OCR Engine to become healthy.' >&2; \
			$(DOCKER_COMPOSE) ps >&2 || true; \
			$(DOCKER_COMPOSE) logs --tail 100 $(COMPOSE_SERVICE) >&2 || true; \
			exit 1; \
		fi; \
		sleep $(WAIT_INTERVAL_SECONDS); \
	done

test-deploy: deploy

deploy:
	$(DOCKER_COMPOSE) config >/dev/null
	$(DOCKER_COMPOSE) build $(COMPOSE_SERVICE)
	$(DOCKER_COMPOSE) up -d --remove-orphans --no-build $(COMPOSE_SERVICE)
	$(MAKE) health
	$(DOCKER_COMPOSE) ps
