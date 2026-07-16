.PHONY: help up down build test compile validate

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

PYTHON ?= py -3
PYCACHE ?= /tmp/dokstract-ocr-engine-pycache

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make up        - start the OCR Engine Service compose stack' \
		'  make down      - stop the OCR Engine Service compose stack' \
		'  make build     - build the OCR Engine container' \
		'  make test      - run the unit test suite' \
		'  make compile   - syntax-check the Python sources' \
		'  make validate  - run compile and tests'

up:
	docker compose --env-file docker/.env -f docker/docker-compose.yml up -d --build

down:
	docker compose --env-file docker/.env -f docker/docker-compose.yml down

build:
	docker compose --env-file docker/.env -f docker/docker-compose.yml build

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
