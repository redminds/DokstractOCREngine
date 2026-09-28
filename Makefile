.PHONY: help config build up down restart logs ps health test-deploy deploy test compile validate benchmark-fast benchmark-recog benchmark-clean

SHELL := bash
.SHELLFLAGS := -Eeuo pipefail -c

# Keep root commands aligned with the deployment-aware Docker Makefile.
ENV_FILE ?= .env
COMPOSE_FILE ?= docker-compose.yml
SECONDARY_COMPOSE_FILE ?= docker-compose.secondary-network.yml
COMPOSE_SERVICE ?= ocr-engine
BENCHMARK_CASE ?=
# Benchmarks are optional developer tooling and are not part of lifecycle.
HOST_PYTHON ?= python3

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
	$(MAKE) -C docker test \
		ENV_FILE="$(ENV_FILE)" \
		COMPOSE_FILE="$(COMPOSE_FILE)" \
		SECONDARY_COMPOSE_FILE="$(SECONDARY_COMPOSE_FILE)" \
		TEST_COMPOSE_FILE="docker-compose.test.yml"

compile:
	$(MAKE) -C docker compile \
		ENV_FILE="$(ENV_FILE)" \
		COMPOSE_FILE="$(COMPOSE_FILE)" \
		SECONDARY_COMPOSE_FILE="$(SECONDARY_COMPOSE_FILE)" \
		TEST_COMPOSE_FILE="docker-compose.test.yml"

validate: compile test

benchmark-fast:
	$(HOST_PYTHON) tools/ocr_benchmark.py fast $(if $(BENCHMARK_CASE),--case $(BENCHMARK_CASE),)

benchmark-recog:
	$(HOST_PYTHON) tools/ocr_benchmark.py recog $(if $(BENCHMARK_CASE),--case $(BENCHMARK_CASE),)

benchmark-clean:
	rm -rf artifacts/ocr-benchmarks/*
