# This is the build file for the docker. Note this should be run from the
# parent directory for the necessary files to be available

.PHONY: clean build run run-no-checks _docker-run

DIR := ${CURDIR}

build-mocap:
	docker compose build mocap

build-annotate:
	docker compose build annotate

# Start acquisition with full system validation (recommended)
run:
	@./scripts/acquisition/start_acquisition.sh

# Quick start without system validation checks
run-no-checks:
	@./scripts/acquisition/start_acquisition.sh --skip-checks

# Internal target - called by start_acquisition.sh, don't use directly
_docker-run:
	docker compose run --rm mocap

test:
	docker compose run --rm test

reset:
	docker compose run --rm reset

annotate:
	docker compose run --rm annotate
