# Container Reference

This page describes every Docker container defined by MultiCameraTracking — what each one does and how to start, stop, and reset it. For installing the Docker engine itself, see [Docker Setup](../acquisition/docker_setup.md).

## Overview

MultiCameraTracking builds two Docker images and pulls one stock image:

| Image | Source | Used by |
|---|---|---|
| `isr/mct_acquisition` | `docker/Dockerfile` | `mocap`, `mocap-test`, `mocap-test-dj`, `test`, `reset` |
| `isr/mct_annotation` | `docker/Dockerfile.annotation` | `annotate` |
| `mysql:8.0` | Docker Hub | `datajoint-test` |

All services run with `network_mode: host` except `datajoint-test`, which binds only to `127.0.0.1:3307`. Most operations are driven through the `Makefile`; the underlying compose file is `docker-compose.yml`.

## Quick reference

| Service | Image | Purpose | Host ports | Primary Make target |
|---|---|---|---|---|
| `mocap` | `mct_acquisition` | Production acquisition GUI + FastAPI | 8000, 3000 | `make run` |
| `mocap-test` | `mct_acquisition` | Acquisition against isolated `/data-test` | 8000, 3000 | `make run-mocap-test` |
| `mocap-test-dj` | `mct_acquisition` | Acquisition against local test DataJoint | 8000, 3000 | `make run-mocap-test-dj` |
| `test` | `mct_acquisition` | Pytest + diagnostics runner | — | `make test`, `make health`, etc. |
| `reset` | `mct_acquisition` | One-shot FLIR camera reset | — | `make reset` |
| `annotate` | `mct_annotation` | Annotation web tool | 3005 | `make annotate` |
| `datajoint-test` | `mysql:8.0` | Local DataJoint MySQL for testing | 127.0.0.1:3307 | `make run-mocap-test-dj` (auto-starts) |

**Port-conflict rule:** `mocap`, `mocap-test`, and `mocap-test-dj` all bind host ports 8000 (FastAPI) and 3000 (React). Only one can run at a time.

## Building images

```bash
make build-mocap       # builds isr/mct_acquisition from docker/Dockerfile
make build-annotate    # builds isr/mct_annotation from docker/Dockerfile.annotation
```

Rebuild after changes to `Dockerfile`, `Dockerfile.annotation`, `multi_camera/`, `frontend/`, `pyproject.toml`, or `docker/entrypoints/`. The first build is slow because the acquisition image installs the FLIR Spinnaker SDK; subsequent rebuilds are cached.

## Per-container detail

### `mocap` — production acquisition

The primary acquisition container. Runs the Python FastAPI backend and the React acquisition UI against attached FLIR cameras.

| | |
|---|---|
| **Entrypoint** | `/Mocap/entrypoints/start_acquisition_gui.sh` |
| **Processes** | FastAPI (`python3 -m multi_camera.backend.fastapi`, port 8000) and React serve (port 3000) |
| **Volumes** | `${DATA_VOLUME:-/data}` → `/data`, `${CAMERA_CONFIGS:-/camera_configs}` → `/configs`, `${DATAJOINT_EXTERNAL:-/mnt/datajoint_external}` → `/datajoint_external` |
| **Network** | `host` |
| **Lifecycle** | Foreground; runs until stopped |

**Start:**
```bash
make run             # with system validation (DHCP, disk, env) — recommended
make run-no-checks   # skip validation
```

`make run` invokes `scripts/acquisition/start_acquisition.sh`, which validates the host before calling `docker compose run --rm mocap`.

**Stop:** Ctrl-C in the foreground terminal. The `--rm` flag removes the container on exit. If launched detached, use `docker compose stop mocap`.

**Reset:** No persistent state in the container itself; recording data lives on the host under `${DATA_VOLUME}`. To reset FLIR cameras after a hang, see [`reset`](#reset--flir-camera-reset).

### `mocap-test` — acquisition with isolated data

Same image and behavior as `mocap`, but recordings land in `${TEST_DATA_VOLUME:-/data-test}` so production data is untouched. Also sets `REACT_APP_TEST_MODE=true` so the UI rebuilds with a test banner.

| | |
|---|---|
| **Start** | `make run-mocap-test` |
| **Stop** | Ctrl-C |
| **Volumes** | `${TEST_DATA_VOLUME:-/data-test}` → `/data` (all others same as `mocap`) |

Cannot run simultaneously with `mocap` or `mocap-test-dj` (port conflict).

### `mocap-test-dj` — acquisition against local DataJoint

Same image as `mocap`, configured to upload to a local MySQL container (`datajoint-test`) instead of production DataJoint. Use this when developing or testing DataJoint inserts without touching shared infrastructure.

| | |
|---|---|
| **Start** | `make run-mocap-test-dj` (auto-starts `datajoint-test` and waits for its health check) |
| **Stop** | Ctrl-C (stops mocap-test-dj; `datajoint-test` keeps running) |
| **DataJoint config** | `docker/datajoint_config.test.json` mounted read-only at `/root/.datajoint_config.json` |
| **DataJoint external storage** | `${TEST_DJ_EXTERNAL:-/tmp/datajoint_external_test}` → `/datajoint_external` |

Requires one-time setup with `make init-dj-test`. See [DataJoint Test Environment](datajoint_test_environment.md) for the full workflow.

### `test` — pytest + diagnostics

A non-interactive container used to run pytest and most diagnostic tools. Mounts `./tests` and `./tests/testdata` from the repo so test files don't need to be baked into the image.

| | |
|---|---|
| **Entrypoint** | `/Mocap/entrypoints/run_tests.sh` (default) or overridden via `--entrypoint` |
| **Lifecycle** | One-shot; auto-removes on exit |

**Test targets:**
```bash
make test                # all acquisition tests (cameras required for matrix tests)
make test-matrix         # camera test matrix only (cameras required, long)
make test-diagnostics    # sync diagnostics unit tests (no cameras required)
```

**Diagnostics targets** (also use the `test` container via `--entrypoint` overrides):

| Target | Cameras? | Purpose |
|---|---|---|
| `make validate-sync CONFIG=/configs/your_config.yaml` | yes | Pre-recording sync validation |
| `make diag-recording CONFIG=/configs/your_config.yaml [FRAMES=N] [DATA=/path]` | yes | Short recording with full diagnostics |
| `make diag-analyze [DATA=/data]` | no | Post-hoc analysis of recording JSON |
| `make health` | no | Host network/DHCP/MTU/camera reachability check |
| `make health-fix` | no | Same as `health` but auto-remediates (requires passwordless sudo for `ip link`, `sysctl`, `systemctl`) |

### `reset` — FLIR camera reset

One-shot container that hardware-resets all FLIR cameras (clears IEEE1588 state, releases stuck buffers). Useful after a sync failure or when cameras stop responding.

| | |
|---|---|
| **Command** | `python3 -m multi_camera.acquisition.cameras_reset -av` |
| **Lifecycle** | Runs once, exits, auto-removes |

```bash
make reset
```

### `annotate` — annotation web tool

Web tool for reviewing and correcting pose estimates. Independent from `mocap` — different image, different port, no camera access needed.

| | |
|---|---|
| **Entrypoint** | `/Mocap/entrypoints/run_annotate.sh` |
| **Processes** | FastAPI annotation backend (`multi_camera.backend.fastapi_annotation`) and React annotation UI |
| **Port** | 3005 |
| **Volumes** | `${DATAJOINT_EXTERNAL:-/mnt/datajoint_external}` → `/datajoint_external` |

```bash
make build-annotate   # first time only (or after Dockerfile.annotation changes)
make annotate
```

Stop with Ctrl-C.

### `datajoint-test` — local DataJoint MySQL

Stock MySQL 8.0 container providing a local DataJoint server for testing. Bound to `127.0.0.1:3307` only (not exposed externally). Data persists in the `datajoint_test_db` Docker volume.

| | |
|---|---|
| **Image** | `mysql:8.0` |
| **Port** | `127.0.0.1:3307` → `3306` |
| **Volume** | `datajoint_test_db` (named, persistent across restarts) |
| **Root password** | `djtest` by default (override with `DJ_TEST_PASS`) |
| **Health check** | `mysqladmin ping` every 10 s, 5 retries, 30 s grace |

Normally you do not start this container directly — `make run-mocap-test-dj` brings it up automatically. For full lifecycle (init, reset, stop, troubleshooting), see [DataJoint Test Environment](datajoint_test_environment.md).

## Cross-cutting concerns

### Host networking

All acquisition containers (`mocap`, `mocap-test`, `mocap-test-dj`, `test`, `reset`) and the `annotate` container use Docker's `host` network mode. This is required for FLIR camera discovery (which relies on direct GigE access) and for IEEE1588 PTP synchronization, neither of which works through Docker's default bridge networking. Consequently, container processes share the host's port space — see the port-conflict rule above.

### Environment variables

The acquisition stack reads configuration from a `.env` file at the repo root. The template is at `.env.template`; run `make setup-env` to validate your `.env` against the template (prompts for any missing values). Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `DJ_USER`, `DJ_PASS`, `DJ_HOST`, `DJ_PORT` | — | DataJoint credentials |
| `NETWORK_INTERFACE` | `enp5s0` | NIC bound to camera subnet |
| `DEPLOYMENT_MODE` | `network` | `laptop` or `network` |
| `DATA_VOLUME` | `/data` | Recording storage |
| `TEST_DATA_VOLUME` | `/data-test` | Test-mode recording storage |
| `CAMERA_CONFIGS` | `/camera_configs` | YAML camera configs |
| `DATAJOINT_EXTERNAL` | `/mnt/datajoint_external` | DataJoint external blob storage |
| `TEST_DJ_EXTERNAL` | `/tmp/datajoint_external_test` | DataJoint external storage for test |
| `DJ_TEST_PASS` | `djtest` | Local DataJoint MySQL root password |
| `DISK_SPACE_WARNING_THRESHOLD_GB` | `50` | Min free space at startup |

### Data flow

The acquisition containers write recordings to `/data` inside the container, which maps to the host's `${DATA_VOLUME}`. Each session produces per-camera `{serial}.mp4` files plus a `.json` metadata sidecar (consumed by `make diag-analyze`). Pushes to DataJoint copy large binary blobs into `${DATAJOINT_EXTERNAL}` and insert rows into MySQL.

## See also

- [Docker Setup](../acquisition/docker_setup.md) — installing the Docker engine
- [DataJoint Test Environment](datajoint_test_environment.md) — full local DJ workflow
- [Unified Startup Script](../acquisition/startup_script.md) — what `make run` validates before launching
- [Backend Architecture](../development/backend_architecture.md) — what the code inside these containers actually does
