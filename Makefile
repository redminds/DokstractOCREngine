.PHONY: help config build up down restart logs ps health test-deploy deploy test compile validate benchmark-fast benchmark-recog benchmark-clean

SHELL := /bin/bash
.SHELLFLAGS := -Eeuo pipefail -c

# Keep root commands aligned with the deployment-aware Docker Makefile.
ENV_FILE ?= .env
COMPOSE_FILE ?= docker-compose.yml
SECONDARY_COMPOSE_FILE ?= docker-compose.secondary-network.yml
COMPOSE_SERVICE ?= ocr-engine
PYTHON ?= py -3
PYCACHE ?= /tmp/dokstract-ocr-engine-pycache
BENCHMARK_CASE ?=

help:
	@printf '%s\n' \
		'Available targets:' \
		'  config             - validate and render Docker Compose' \
		'  build              - build the OCR Engine image' \
		'  up                 - start the OCR Engine compose service' \
		'  down               - stop the OCR Engine compose service' \
		'  restart            - restart the OCR Engine compose service' \
		'  logs               - follow OCR Engine logs' \
		'  ps                 - show OCR Engine status' \
		'  health             - wait for OCR Engine health' \
		'  deploy             - validate, build, deploy, and wait for health' \
		'  test               - run the unit test suite' \
		'  compile            - syntax-check the Python sources' \
		'  validate           - run compile and tests' \
		'  benchmark-fast     - fast production OCR validation' \
		'  benchmark-recog    - recognition investigation with crop variants' \
		'  benchmark-clean    - remove benchmark artifacts'

config build up down restart logs ps health test-deploy deploy:
	$(MAKE) -C docker \
		ENV_FILE="$(ENV_FILE)" \
		COMPOSE_FILE="$(COMPOSE_FILE)" \
		SECONDARY_COMPOSE_FILE="$(SECONDARY_COMPOSE_FILE)" \
		COMPOSE_SERVICE="$(COMPOSE_SERVICE)" \
		$@

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
