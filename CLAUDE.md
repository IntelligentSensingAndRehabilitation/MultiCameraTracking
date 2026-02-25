# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MultiCameraTracking is a system for multi-camera video acquisition, pose estimation, 3D reconstruction, and SMPL body model fitting for clinical movement analysis and biomechanics research. It uses FLIR cameras with IEEE1588 synchronization and builds on DataJoint for pipeline orchestration.

## Build & Development Commands

```bash
# Install (analysis pipeline only, no acquisition system)
pip install -e .
pip install -e ".[opencv]"          # with OpenCV GUI
pip install -e ".[opencv-headless]" # headless variant for servers

# Testing
pytest tests/                       # all tests
pytest tests/test_camera.py         # camera math tests
pytest tests/test_camera.py::test_project  # single test
# Note: tests/test_data_integrity.py requires live DataJoint + SQLite connections
# Note: tests/acquisition/ tests require real FLIR cameras attached

# Docker (acquisition system — requires .env file, see .env.template)
make build-mocap                    # build acquisition container
make build-annotate                 # build annotation container
make run                            # start acquisition with system validation
make run-no-checks                  # skip validation
make test                           # run tests in Docker
make reset                          # hardware-reset all FLIR cameras
make annotate                       # start annotation container
```

## Architecture

### Dual-Database Pattern

The system uses two databases simultaneously:

- **DataJoint** (MySQL): Pipeline orchestration, computed tables, analysis results. Configured via `datajoint_config.json` (copied to `/root/.datajoint_config.json` in Docker). Requires `enable_python_native_blobs: true` for numpy array storage.
- **SQLite** (`recording_db.py` → `data/recordings.db`): Local acquisition database tracking participants, sessions, and recordings. The `Imported` table tracks which sessions have been pushed to DataJoint. `synchronize_to_datajoint()` reconciles both databases at startup.

### Two FastAPI Applications

- **Acquisition API** (`multi_camera/backend/fastapi.py`, port 8000): Camera recording, session management, live preview streaming via WebSocket (`/video_ws`), calibration, pushing recordings to DataJoint. Uses a `GlobalState` dataclass singleton holding the `FlirRecorder` instance, current session, and frame queue. Acquisition callbacks run on a separate thread.
- **Annotation API** (`multi_camera/backend/fastapi_annotation.py`, port 8005): Unannotated recording discovery, SMPL mesh delivery, and annotation posting.

### Frontend (React 18, CRA)

Two separate React apps in `frontend/`:
- `frontend/acquisition/` — Components: `CameraStatusTable`, `Config`, `Participant`, `RecordingControl`, `SmplBrowser`, `Video`, etc. Uses Three.js + `@react-three/fiber` for 3D mesh visualization.
- `frontend/annotation/` — `Annotator` component with `visualization_js/`.

Both use react-bootstrap (Bootswatch theme) and axios. npm start uses `--openssl-legacy-provider` flag.

### DataJoint Pipeline (core orchestration)

Three main schemas:

- **`multicamera_tracking`** (`multi_camera/datajoint/multi_camera_dj.py` + related files): Recording and reconstruction tables. Key tables: `MultiCameraRecording`, `SingleCameraVideo`, `CalibratedRecording`, `Calibration` (in `calibrate_cameras.py`), `PersonKeypointReconstruction` (Computed), `PersonKeypointReconstructionMethodLookup` (Lookup with 13 methods), `SMPLReconstruction`/`SMPLXReconstruction` (in `smpl.py`), `SynchronizationQuality` (Computed, in `quality_metrics.py`).
- **`mocap_sessions`** (`multi_camera/datajoint/sessions.py`): Session/subject organization. `Subject`, `Session`, `Recording`. Links to `MultiCameraRecording` via foreign key. `SessionCalibration` (Computed, in `session_calibrations.py`) validates one calibration per recording.
- **`multicamera_tracking_annotation`** (`multi_camera/datajoint/annotation.py`): Activity labels (`VideoActivityLookup`/`VideoActivity` — walking, TUG, FMS, CUET, etc.), `RangeAnnotation`, `EventAnnotation`.

Additional schemas: `multicamera_tracking_gaitrite` (GaitRite pressure mat comparison), NimblePhysics biomechanics.

EasyMocap integration lives in `multi_camera/datajoint/easymocap.py` with `EasymocapTracking` and `EasymocapSmpl` computed tables, plus `SkippedRecording` (standalone `dj.Manual` log table with no FKs — records failures that exhausted all tracking configs). Uses `MCTDataset` to bridge DataJoint data into EasyMocap's `MVBase` interface.

**DataJoint gotchas:**
- `camera_config_hash` (varchar) is part of primary keys — handles multiple camera setups.
- `SkippedRecording` has NO foreign keys intentionally, so it can be inserted even when FK tables lack matching entries.
- `EasymocapTracking.key_source` uses `.proj()` on the `SkippedRecording` subtraction to avoid attribute conflicts from overlapping secondary attributes (`video_project`, `video_base_filename`).
- `import_session()` wraps all inserts in explicit transactions (`dj.conn().start_transaction()`).
- Use `thorough_delete_recording()` / `thorough_delete_calibration()` from `datajoint/utils/recording_delete.py` for safe cascading deletes — required because of inverted FK dependencies with PosePipeline tables.

### Processing Pipeline Stages

The main workflow is in `scripts/session_pipeline.py`:

1. **Pre-annotation**: `preannotation_session_pipeline()` — 2D bottom-up pose detection (OpenPose/Bridging via PosePipeline) → EasyMocap tracking → SMPL fitting
2. **Annotation**: Manual correction in web interface (FastAPI + React)
3. **Post-annotation**: `postannotation_session_pipeline()` — Top-down person detection → 3D `PersonKeypointReconstruction` → refined SMPL

Calibration assignment: `assign_calibration()` matches calibrations to recordings by timestamp proximity with reprojection error threshold < 0.2.

Defaults: top-down method `"Bridging_bml_movi_87"`, reconstruction method `"Robust Triangulation"`, tracking method `"Easymocap"`, bottom-up method `"Bridging_OpenPose"` (hardcoded in `easymocap.py`).

### Module Responsibilities

- **`multi_camera/analysis/`**: Core algorithms. `camera.py` (JAX-based camera math using jaxlie SO3/SE3), `reconstruction.py` (3D triangulation via aniposelib with einops), `calibration.py` (camera calibration with ChArUco/checkerboard boards, Dash/Plotly visualization), `optimize_reconstruction.py` (implicit function optimization with Flax `KeypointTrajectory` module + Optax).
- **`multi_camera/analysis/biomechanics/`**: OpenSim fitting (`opensim_fitting.py` is the largest module at 34KB), bilevel optimization, OpenCap augmenter.
- **`multi_camera/datajoint/`**: All DataJoint table definitions and populate logic. Depends on `pose_pipeline` package for upstream tables (`Video`, `VideoInfo`, `TopDownPerson`, `BottomUpPeople`).
- **`multi_camera/acquisition/`**: FLIR camera recording (`flir_recording_api.py` — `FlirRecorder` class), diagnostics tools. Each camera gets its own writer thread with a Queue; outputs `{serial}.mp4` files with `mp4v` codec plus a `.json` metadata sidecar.
- **`multi_camera/backend/`**: Two FastAPI apps (see above), SQLite recording database, Jinja2 templates.
- **`multi_camera/utils/standard_pipelines.py`**: Reconstruction pipeline composition helpers wrapping PosePipeline utilities.
- **`multi_camera/experimental/`**: WIP code (`mvmhat.py`, `gaitrite_mtc.py`) — not part of the main pipeline.

### Camera Model Conventions

Camera parameters are stored as a dict with keys `mtx`, `dist`, `rvec`, `tvec` (all arrays indexed by camera):
- **`mtx`**: Each row is `[fx, fy, cx, cy]` in **normalized** coordinates (divide pixel values by 1000). `get_intrinsic()` multiplies by 1000 to reconstruct the standard K matrix.
- **`tvec`**: In **meters**. `get_extrinsic()` multiplies by 1000 to get mm.
- **`rvec`**: Axis-angle (Rodrigues) representation, used with jaxlie SO3.
- **`dist`**: Distortion coefficients (OpenCV convention).

`robust_triangulate_points()` returns shape `(T, J, 4)` where the 4th channel is a confidence weight. Output is in meters.

EasyMocap bridge (`_build_camera()`) divides `tvec` by 1000 — this converts from the DataJoint storage format (meters) to what EasyMocap expects.

### SVD Convergence Retry

`TRACKING_CONFIGS` in `datajoint/easymocap.py` defines a fallback chain of 5 YAML configs (`mvmp1f_default.yml` → fallbacks). Note: the ordering is default, fallback1, fallback3, fallback4, fallback2 (intentionally not sequential). When SVD doesn't converge with the primary config, subsequent configs are tried automatically. If all configs fail, the recording is inserted into `SkippedRecording`.

The default tracking config for `mvmp_association_and_tracking()` in `analysis/easymocap.py` is `mvmp1f_default.yml` (set directly in the function signature via module-level `_default_config`).

### Participant ID Normalization

`normalize_participant_id()` in `recording_db.py` strips a leading `p` or `t` when followed by digits (legacy: `p123` → `123`). Newer alphanumeric IDs (like `TF47`) are handled via a hardcoded mapping in `sessions.py`.

## Key External Dependencies

- **PosePipeline**: 2D pose detection upstream — provides `Video`, `VideoInfo`, `TopDownPerson`, `BottomUpPeople` tables
- **EasyMocap**: SMPL body model fitting — imported conditionally to avoid hard dependency for data access
- **DataJoint** (pinned < 2.0): Pipeline orchestration and data persistence
- **JAX/jaxlie/jaxopt/optax**: All numerical optimization and camera math
- **aniposelib**: Camera calibration and triangulation primitives
- **OpenCV** (>= 4.11.0.86): Required for calibration. Three optional install variants: `opencv`, `opencv-headless`, `opencv-contrib`

## Environment Variables

For the acquisition system (see `.env.template`):
- `DJ_USER`, `DJ_PASS`, `DJ_HOST`, `DJ_PORT` — DataJoint credentials
- `NETWORK_INTERFACE` — e.g. `enp5s0`
- `DEPLOYMENT_MODE` — `"laptop"` or `"network"`
- `DATA_VOLUME` (default `/data`) — recording storage
- `CAMERA_CONFIGS` (default `/camera_configs`) — YAML camera configs
- `DATAJOINT_EXTERNAL` (default `/mnt/datajoint_external`) — DataJoint external storage
- `DISK_SPACE_WARNING_THRESHOLD_GB` (default 50)
- `SMPL_CLEAN_PATH` — override default SMPL model directory (`model_data/smpl_clean/`)

## Versioning

CalVer format: `YYYY.MM.DD` (currently 2025.12.03).

## Code Style

PEP 8, ~100 character line length, type hints where appropriate. No linter/formatter currently configured in pyproject.toml.
