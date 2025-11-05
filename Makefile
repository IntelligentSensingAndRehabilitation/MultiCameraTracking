# This is the build file for the docker. Note this should be run from the
# parent directory for the necessary files to be available

.PHONY: clean build run

DIR := ${CURDIR}

build-mocap:
	docker compose build mocap

build-annotate:
	docker compose build annotate

run:
	docker compose run --rm mocap

test:
	docker compose run --rm test

reset:
	docker compose run --rm reset

annotate:
	docker compose run --rm annotate

backup-sync:
	docker compose run --rm backup --session $(ARGS)

backup-batch:
	docker compose run --rm backup --start-date $(ARGS)

backup-status:
	docker compose run --rm backup --status $(ARGS)

backup-status-range:
	docker compose run --rm backup --status-range $(ARGS)

backup-verify:
	docker compose run --rm backup --verify $(ARGS)

backup-delete:
	docker compose run --rm -it backup --safe-delete $(ARGS)
