# Backend Architecture

A developer-facing overview of the MultiCameraTracking codebase: the services that run, how data flows, and where to look for what. For container/lifecycle details see [Container Reference](../operations/container_reference.md). For the camera-side processing pipeline, see [Processing Pipeline](../analysis/processing_pipeline.md).

## System overview

MultiCameraTracking is the backend half of the capture system. It runs on a Linux host with attached FLIR cameras and is responsible for:

1. **Acquisition** — recording synchronized video from multiple FLIR cameras (IEEE1588 PTP)
2. **Local data management** — tracking participants, sessions, and recordings in a local SQLite database
3. **Pipeline orchestration** — pushing recordings into DataJoint and running pose estimation, 3D reconstruction, and SMPL fitting
4. **Annotation** — serving a web UI for reviewing and correcting pose estimates

The companion client is the Flutter `capture-app/`, which talks to this backend over HTTP/WebSocket. See [`capture-app/documentation/MULTICAMERA_API.md`](https://github.com/IntelligentSensingAndRehabilitation/capture-app/blob/main/documentation/MULTICAMERA_API.md) for the API surface.

## Two FastAPI applications

The backend serves two independent HTTP APIs, each backed by its own React frontend:

| App | Module | Port | Frontend | Purpose |
|---|---|---|---|---|
| Acquisition | `multi_camera.backend.fastapi` | 8000 | `frontend/acquisition` (port 3000) | Camera recording, session management, live preview via WebSocket (`/video_ws`), calibration, push to DataJoint |
| Annotation | `multi_camera.backend.fastapi_annotation` | 8005 | `frontend/annotation` (port 3005) | Unannotated-recording discovery, SMPL mesh delivery, annotation posting |

The acquisition API uses a `GlobalState` dataclass singleton that holds the `FlirRecorder`, current session, and a frame queue. Camera callbacks run on a background thread; the FastAPI event loop reads from the frame queue to stream previews.

Both apps live behind their containers' entrypoint scripts — they're started by `start_acquisition_gui.sh` / `run_annotate.sh`, which also build and serve the React bundles.

## Dual-database pattern

Two databases are in use simultaneously:

- **DataJoint** (MySQL) — pipeline orchestration, computed tables, analysis results. Configured via `datajoint_config.json` (copied to `/root/.datajoint_config.json` in the image). Requires `enable_python_native_blobs: true` for numpy storage.
- **SQLite** — local acquisition database at `data/recordings.db`, defined in `multi_camera/backend/recording_db.py` (SQLAlchemy ORM). Tracks participants, sessions, recordings, photos, and a sync-status `Imported` table.

`synchronize_to_datajoint()` reconciles the two databases at startup: any local recording marked as imported but missing from DataJoint is re-flagged so the next push can retry. This is the safety net that keeps the local and remote views consistent across crashes, network failures, and manual DataJoint edits.

For test/dev work, swap the production DataJoint for a local MySQL container — see [DataJoint Test Environment](../operations/datajoint_test_environment.md).

## Frontend apps

Two React 18 (Create React App) apps live in `frontend/`:

- `frontend/acquisition/` — main operator UI. Notable components: `CameraStatusTable`, `Config`, `Participant`, `RecordingControl`, `SmplBrowser`, `Video`. Uses Three.js + `@react-three/fiber` for 3D mesh visualization.
- `frontend/annotation/` — the `Annotator` view plus `visualization_js/` helpers.

Both use react-bootstrap (Bootswatch theme) and axios. npm scripts pass `--openssl-legacy-provider` because of an OpenSSL 3 / webpack 4 incompatibility — keep that flag when modifying the npm scripts.

The acquisition image pre-builds the React bundle at image build time with default env vars and writes a fingerprint file. `start_acquisition_gui.sh` checks the runtime env-var fingerprint and rebuilds the bundle only if it differs — for example, `mocap-test` sets `REACT_APP_TEST_MODE=true`, which triggers a rebuild on first launch.

## Module map

```
multi_camera/
├── acquisition/      FLIR camera capture, IEEE1588 sync, diagnostics
├── analysis/         3D math, calibration, reconstruction, biomechanics
├── backend/          The two FastAPI apps + SQLite recording DB
├── datajoint/        DataJoint schemas and populate logic
├── utils/            Shared helpers, standard pipeline composition
├── validation/       Input validation
├── visualization/    Plotting and figure helpers
├── wrappers/         Thin wrappers around external libs
└── experimental/     WIP code (mvmhat.py, gaitrite_mtc.py) — not in main pipeline
```

| Module | What lives there | Key entry points |
|---|---|---|
| `acquisition` | FLIR capture + sync diagnostics | `flir_recording_api.py` (`FlirRecorder`), `diagnostics/json_parser.py`, `diagnostics/system_monitor.py`, `cameras_reset.py`, `health.py` |
| `analysis` | Camera math, calibration, reconstruction | `camera.py` (JAX/jaxlie), `reconstruction.py` (aniposelib triangulation), `calibration.py` (ChArUco), `optimize_reconstruction.py` (Flax + Optax), `biomechanics/opensim_fitting.py` |
| `backend` | Web layer + local DB | `fastapi.py`, `fastapi_annotation.py`, `recording_db.py` |
| `datajoint` | All DJ tables and pipelines | `multi_camera_dj.py`, `sessions.py`, `annotation.py`, `easymocap.py`, `calibrate_cameras.py`, `smpl.py`, `quality_metrics.py`, `session_calibrations.py`, `utils/recording_delete.py` |
| `utils` | Reusable helpers | `standard_pipelines.py` |

## DataJoint schemas

Three primary schemas:

- **`multicamera_tracking`** — Recording and reconstruction tables. Key tables: `MultiCameraRecording`, `SingleCameraVideo`, `CalibratedRecording`, `Calibration`, `PersonKeypointReconstruction` (Computed), `PersonKeypointReconstructionMethodLookup` (13 methods), `SMPLReconstruction`/`SMPLXReconstruction`, `SynchronizationQuality` (Computed).
- **`mocap_sessions`** — Session/subject organization. `Subject`, `Session`, `Recording`. Links to `MultiCameraRecording` via foreign key. `SessionCalibration` validates one calibration per recording.
- **`multicamera_tracking_annotation`** — Activity labels. `VideoActivityLookup`/`VideoActivity` (walking, TUG, FMS, CUET, etc.), `RangeAnnotation`, `EventAnnotation`.

Additional schemas: `multicamera_tracking_gaitrite` (GaitRite pressure-mat comparison), plus NimblePhysics biomechanics tables.

Upstream dependencies (provided by `pose_pipeline`): `Video`, `VideoInfo`, `TopDownPerson`, `BottomUpPeople`.

Notes:
- `camera_config_hash` is part of primary keys so multiple camera setups can coexist.
- `SkippedRecording` has **no foreign keys** intentionally — it logs failures even when FK tables lack matching entries.
- Use `thorough_delete_recording()` / `thorough_delete_calibration()` in `datajoint/utils/recording_delete.py` for safe cascading deletes (handles inverted FK dependencies with PosePipeline).

See [Processing Pipeline](../analysis/processing_pipeline.md) for the populate workflow.

## Local development workflow

There are two ways to run backend code:

### In Docker (required for cameras)

Anything that touches FLIR cameras must run in the `mocap` container (FLIR Spinnaker SDK is installed there; the host doesn't need it). See [Container Reference](../operations/container_reference.md).

### Without Docker (analysis, DataJoint, unit tests)

For DataJoint pipeline work, analysis-only changes, or unit tests, install the package locally with pip:

```bash
pip install -e .
pip install -e ".[opencv]"          # with OpenCV GUI
pip install -e ".[opencv-headless]" # headless variant for servers
```

This installs the `multi_camera` package without the acquisition system. You can then run pytest, develop DataJoint table changes, or debug analysis code interactively. The `tests/acquisition/` tests still require real cameras, but most other tests run cleanly without hardware.

### Running tests

```bash
pytest tests/                            # all unit tests
pytest tests/test_camera.py              # camera math tests
pytest tests/test_camera.py::test_project # single test

# In Docker (containerized test suite):
make test                                # all (cameras required for matrix)
make test-diagnostics                    # no cameras required
make test-matrix                         # cameras required
```

`tests/test_data_integrity.py` requires live DataJoint + SQLite connections. `tests/acquisition/` requires attached FLIR cameras.

## Camera-model conventions

Camera parameters are dicts keyed by camera with arrays `mtx`, `dist`, `rvec`, `tvec`:

- `mtx` — each row is `[fx, fy, cx, cy]` in **normalized** coordinates (pixel values divided by 1000). `get_intrinsic()` multiplies by 1000 to reconstruct the standard K matrix.
- `tvec` — in **meters**. `get_extrinsic()` multiplies by 1000 to produce mm.
- `rvec` — axis-angle (Rodrigues), used with jaxlie SO3.
- `dist` — OpenCV-convention distortion coefficients.

`robust_triangulate_points()` returns shape `(T, J, 4)` where the 4th channel is a confidence weight. Output is in meters.

The EasyMocap bridge (`_build_camera()`) divides `tvec` by 1000 — that's converting from DataJoint's meter convention back to whatever EasyMocap expects internally.

## See also

- [Container Reference](../operations/container_reference.md) — containers and lifecycle
- [DataJoint Test Environment](../operations/datajoint_test_environment.md) — local DJ for development
- [Processing Pipeline](../analysis/processing_pipeline.md) — populate workflow and stage-by-stage detail
- [SMPL Model Setup](../analysis/smpl_setup.md) — model files and paths
- [`capture-app` API reference](https://github.com/IntelligentSensingAndRehabilitation/capture-app/blob/main/documentation/MULTICAMERA_API.md) — endpoints the Flutter client calls
