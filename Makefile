.PHONY: help config build up down restart logs ps health test-deploy deploy test compile validate benchmark-fast benchmark-recog benchmark-clean

SHELL := /bin/bash
.SHELLFLAGS := -Eeuo pipefail -c

REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
DOCKER_DIR := $(REPO_ROOT)/docker
COMPOSE_FILE ?= docker-compose.yml
ENV_FILE ?= .env
COMPOSE_PROJECT_NAME ?= dokstract-ocr-engine
COMPOSE_SERVICE ?= ocr-engine
DOCKER_COMPOSE = docker compose --env-file "$(DOCKER_DIR)/$(ENV_FILE)" -p "$(COMPOSE_PROJECT_NAME)" -f "$(DOCKER_DIR)/$(COMPOSE_FILE)"
PYTHON ?= py -3
PYCACHE ?= /tmp/dokstract-ocr-engine-pycache
WAIT_TIMEOUT_SECONDS ?= 180
WAIT_INTERVAL_SECONDS ?= 5
BENCHMARK_CASE ?=

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make config           - validate the compose file and env file' \
		'  make build            - build the OCR Engine image' \
		'  make up               - start the OCR Engine compose service' \
		'  make down             - stop the OCR Engine compose service' \
		'  make restart          - restart the OCR Engine compose service' \
		'  make logs             - follow OCR Engine logs' \
		'  make ps               - show OCR Engine status' \
		'  make health           - wait for OCR Engine health' \
		'  make deploy           - validate, build, deploy, and wait for health' \
		'  make test-deploy      - alias for deploy' \
		'  make test             - run the unit test suite' \
		'  make compile          - syntax-check the Python sources' \
		'  make validate         - run compile and tests' \
		'  make benchmark-fast   - fast production OCR validation [CASE=case-id]' \
		'  make benchmark-recog  - recognition investigation with crop variants [CASE=case-id]' \
		'  make benchmark-clean  - remove benchmark artifacts'

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

test:
	$(PYTHON) -m pytest

compile:
	PYTHONPYCACHEPREFIX=$(PYCACHE) $(PYTHON) -m py_compile \
		app/main.py \
		app/core/config.py \
		app/core/registry.py \
		app/core/service.py \
		app/api/v1/routes/engine.py \
		app/api/v1/routes/admin.py \
		tests/test_registry.py \
		tests/test_api.py

validate: compile test

benchmark-fast:
	$(PYTHON) tools/ocr_benchmark.py fast $(if $(BENCHMARK_CASE),--case $(BENCHMARK_CASE),)

benchmark-recog:
	$(PYTHON) tools/ocr_benchmark.py recog $(if $(BENCHMARK_CASE),--case $(BENCHMARK_CASE),)

benchmark-clean:
	rm -rf artifacts/ocr-benchmarks/*
