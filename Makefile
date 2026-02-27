# This is the build file for the docker. Note this should be run from the
# parent directory for the necessary files to be available

.PHONY: clean build run run-no-checks _docker-run run-mocap-test test test-matrix test-diagnostics validate-sync diag-recording diag-analyze

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

# Start acquisition with isolated test data (TEST_DATA_VOLUME, default /data-test).
# Cannot run simultaneously with 'make run' — both bind host ports 8000 and 3000.
run-mocap-test:
	docker compose run --rm mocap-test

# Run all acquisition tests (cameras required for test matrix)
test:
	docker compose run --rm test

# Camera test matrix only (cameras required, long)
test-matrix:
	docker compose run --rm --entrypoint pytest test \
		-s tests/acquisition/test_acquisition.py

# Diagnostics unit tests only (no cameras needed)
test-diagnostics:
	docker compose run --rm --entrypoint pytest test \
		-s tests/acquisition/test_sync_diagnostics.py tests/acquisition/test_system_monitor.py

reset:
	docker compose run --rm reset

annotate:
	docker compose run --rm annotate

# --- Diagnostics targets (cameras required unless noted) ---

# Validate sync before recording (cameras required). Usage:
#   make validate-sync CONFIG=/configs/your_config.yaml
validate-sync:
	@test -n "$(CONFIG)" || { echo "CONFIG is required. Usage: make validate-sync CONFIG=/configs/your_config.yaml"; exit 1; }
	docker compose run --rm --entrypoint python3 test \
		/Mocap/tests/acquisition/validate_sync.py \
		--config $(CONFIG)

# Short recording with full diagnostics (cameras required). Usage:
#   make diag-recording CONFIG=/configs/your_config.yaml
#   make diag-recording CONFIG=/configs/your_config.yaml FRAMES=300
diag-recording:
	@test -n "$(CONFIG)" || { echo "CONFIG is required. Usage: make diag-recording CONFIG=/configs/your_config.yaml"; exit 1; }
	docker compose run --rm --entrypoint python3 test \
		/Mocap/tests/acquisition/diag_recording.py \
		--config $(CONFIG) \
		$(if $(DATA),--output-dir $(DATA)) \
		$(if $(FRAMES),--frames $(FRAMES))

# Analyze recording JSON output (no cameras needed). Usage:
#   make diag-analyze DATA=/data
diag-analyze:
	docker compose run --rm --entrypoint python3 test \
		-m multi_camera.acquisition.diagnostics.json_parser \
		$(or $(DATA),/data) --no-plots
