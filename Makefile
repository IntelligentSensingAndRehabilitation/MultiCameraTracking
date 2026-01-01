# This is the build file for the docker. Note this should be run from the
# parent directory for the necessary files to be available

.PHONY: clean build run start-acquisition

DIR := ${CURDIR}

build-mocap:
	docker compose build mocap

build-annotate:
	docker compose build annotate

run:
	docker compose run --rm mocap

start-acquisition:
	@./scripts/acquisition/start_acquisition.sh

test:
	docker compose run --rm test

reset:
	docker compose run --rm reset

annotate:
	docker compose run --rm annotate
