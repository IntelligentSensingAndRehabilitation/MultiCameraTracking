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

# Docker (acquisition system)
make build-mocap                    # build acquisition container
make build-annotate                 # build annotation container
make run                            # start acquisition with system validation
make run-no-checks                  # skip validation
make test                           # run tests in Docker
```

## Architecture

### DataJoint Pipeline (core orchestration)

Two schemas drive the system:

- **`multicamera_tracking`** (`multi_camera/datajoint/multi_camera_dj.py`): Recording and reconstruction tables. Key tables: `MultiCameraRecording`, `SingleCameraVideo`, `CalibratedRecording`, `PersonKeypointReconstruction` (Computed), `PersonKeypointReconstructionMethodLookup` (Lookup with 13 methods).
- **`mocap_sessions`** (`multi_camera/datajoint/sessions.py`): Session/subject organization. Tables: `Subject`, `Session`, `Recording`. Links to `MultiCameraRecording` via foreign key.

EasyMocap integration lives in `multi_camera/datajoint/easymocap.py` with `EasymocapTracking` and `EasymocapSmpl` computed tables, plus `SkippedRecording` (standalone `dj.Manual` log table with no FKs — records failures that exhausted all tracking configs). Uses `MCTDataset` to bridge DataJoint data into EasyMocap's `MVBase` interface.

### Processing Pipeline Stages

The main workflow is in `scripts/session_pipeline.py`:

1. **Pre-annotation**: `preannotation_session_pipeline()` — 2D bottom-up pose detection (OpenPose/Bridging via PosePipeline) → EasyMocap tracking → SMPL fitting
2. **Annotation**: Manual correction in web interface (FastAPI + React)
3. **Post-annotation**: `postannotation_session_pipeline()` — Top-down person detection → 3D `PersonKeypointReconstruction` → refined SMPL

Calibration assignment: `assign_calibration()` matches calibrations to recordings by timestamp proximity with reprojection error threshold < 0.2.

### Module Responsibilities

- **`multi_camera/analysis/`**: Core algorithms. `camera.py` (JAX-based camera math using jaxlie SO3/SE3), `reconstruction.py` (3D triangulation), `calibration.py` (camera calibration with ChArUco boards), `optimize_reconstruction.py` (implicit function optimization). Config YAMLs (`mvmp1f_*.yml`) provide EasyMocap tracking parameters with fallback chain.
- **`multi_camera/analysis/biomechanics/`**: OpenSim fitting (`opensim_fitting.py` is the largest module at 34KB), bilevel optimization, OpenCap augmenter.
- **`multi_camera/datajoint/`**: All DataJoint table definitions and populate logic. Depends on `pose_pipeline` package for upstream tables (`Video`, `VideoInfo`, `TopDownPerson`, `BottomUpPeople`).
- **`multi_camera/acquisition/`**: FLIR camera recording (`flir_recording_api.py`), diagnostics tools.
- **`multi_camera/backend/`**: FastAPI REST API (`fastapi.py`), SQLite recording database, Jinja2 templates.
- **`multi_camera/utils/standard_pipelines.py`**: Reconstruction pipeline composition helpers wrapping PosePipeline utilities.

### Camera Model Conventions

Camera parameters are stored as a dict with keys `mtx`, `dist`, `rvec`, `tvec` (all arrays indexed by camera). `camera.py` uses normalized units internally: `mtx` values are in normalized coordinates (multiply by 1000 for pixels), `tvec` is in meters (multiply by 1000 for mm). EasyMocap integration divides `tvec` by 1000 to convert to meters.

### SVD Convergence Retry

`TRACKING_CONFIGS` in `datajoint/easymocap.py` defines a fallback chain of 5 YAML configs (`mvmp1f_default.yml` → `fallback1-4.yml`). When SVD doesn't converge with the primary config, subsequent configs are tried automatically. If all configs fail, the recording is inserted into `SkippedRecording` with context from `Recording` and `MultiCameraRecording`.

`EasymocapTracking.key_source` requires a `Recording` entry to exist (ensures session context is available for `SkippedRecording`). The `SkippedRecording` subtraction uses `.proj()` to avoid DataJoint dependent-attribute join errors from overlapping secondary attributes (`video_project`, `video_base_filename`).

The default tracking config for `mvmp_association_and_tracking()` in `analysis/easymocap.py` is `mvmp1f_default.yml` (set directly in the function signature via module-level `_default_config`).

## Key External Dependencies

- **PosePipeline**: 2D pose detection upstream — provides `Video`, `VideoInfo`, `TopDownPerson`, `BottomUpPeople` tables
- **EasyMocap**: SMPL body model fitting — imported conditionally to avoid hard dependency for data access
- **DataJoint** (pinned < 2.0): Pipeline orchestration and data persistence
- **JAX/jaxlie/jaxopt/optax**: All numerical optimization and camera math
- **aniposelib**: Camera calibration and triangulation primitives

## Versioning

CalVer format: `YYYY.MM.DD` (currently 2025.12.03).

## Code Style

PEP 8, ~100 character line length, type hints where appropriate. No linter/formatter currently configured in pyproject.toml.
