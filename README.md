# MultiCameraTracking

A comprehensive system for multi-camera video acquisition, pose estimation, 3D reconstruction, and SMPL body model fitting. Designed for clinical movement analysis and biomechanics research using multi-view synchronous recording.

## Project Status and Development Note

This tool is under active development and used by multiple research labs. The infrastructure is technical to set up and primarily supports internal use and collaborators. We have limited bandwidth to support special or custom use cases beyond the current core functionality.

Breaking changes may be introduced during development. Issues and pull requests are welcome.

## Overview

This library provides an end-to-end processing pipeline from multi-camera video acquisition through to 3D skeletal reconstruction and body model fitting:

- **Acquisition**: Multi-camera video recording with hardware synchronization (FLIR cameras with IEEE1588 support)
- **2D Pose Estimation**: Per-view pose detection using PosePipeline (state-of-the-art models)
- **3D Reconstruction**: Multi-view triangulation and temporal tracking of skeletal keypoints
- **SMPL Fitting**: Automatic fitting of SMPL body models to 3D keypoints with temporal smoothing
- **Data Management**: Structured database pipeline via DataJoint for organizing recordings, calibrations, and results

### Core Modules

- `multi_camera.acquisition` - Recording software
- `multi_camera.acquisition.diagnostics` - Debugging tools for the recording system
- `multi_camera.analysis` - Core 3D reconstruction and analysis algorithms
- `multi_camera.datajoint` - Database schema and computational pipelines
- `multi_camera.utils` - Miscellaneous utilities

## Getting Started (Acquisition)

To set up the acquisition system, follow the comprehensive setup **[docs](docs/README.md)**:

1. **[Hardware Setup](docs/acquisition/acquisition_hardware.md)** - Required camera equipment
2. **[System Setup](docs/acquisition/general_system_setup.md)** - Network and OS configuration
3. **[Docker Setup](docs/acquisition/docker_setup.md)** - Build the acquisition system
4. **[Acquisition Software Setup](docs/acquisition/acquisition_software_setup.md)** - Configure the recording application
5. **[Calibration Procedure](docs/calibration/calibration_procedure.md)** - Camera calibration
6. **[Annotation Setup](docs/annotation/annotation_software_setup.md)** - Annotation system configuration

## Installation (Analysis Only)

To use the analysis code without the acquisition system, install the package in development mode:

```bash
pip install -e .
```

## Data Processing Pipeline

The standard workflow for processing multi-camera recordings is:

### Overview

1. **Acquisition** - Record trials and calibration videos using the GUI
2. **Push to DataJoint** - Transfer videos to database via the GUI
3. **Calibration** - Run camera calibration and link to trials
4. **Bridging** - Run bottom-up pose detection across all views
5. **EasyMocap** - Fit SMPL body models to 3D reconstructed keypoints
6. **Post-Annotation Pipeline** - Run top-down person-specific refinement
7. **Person Keypoint Reconstruction** - Final triangulation with annotated data

### Automated Processing Script

The recommended approach is to use the `session_pipeline.py` script for automated batch processing:

```bash
python scripts/session_pipeline.py \
    --participant_id PARTICIPANT_ID \
    --session_date YYYY-MM-DD \
    --run_easymocap \
    --post_annotation
```

This script handles:
- Automatic calibration assignment (links closest valid calibration to each trial)
- Bottom-up bridging across all views
- EasyMocap SMPL fitting with temporal smoothing
- Post-annotation pipeline with person-specific top-down detection

Optional flags:
- `--project PROJECT_NAME` - Filter by video project
- `--bottom_up` - Run only the bridging step
- `--top_down_method_name` - Specify top-down detection method (default: Bridging_bml_movi_87)
- `--hand_estimation` - Include hand keypoint estimation

### Step-by-Step Manual Processing

If processing individual recordings:

#### 1. Record Videos and Calibration

Use the acquisition GUI to record:
- Trial videos with all cameras synchronized
- Calibration videos with checkerboard pattern

See [Acquisition Startup Guide](docs/acquisition/acquisition_startup.md) for detailed instructions.

#### 2. Push Data to DataJoint

Use the GUI to:
1. Import trial videos into `MultiCameraRecording`
2. Import calibration videos into `Calibration`
3. Process calibration to compute camera parameters

#### 3. Link Calibration to Trials

The script automatically assigns calibrations, but you can manually verify via:

```python
from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording, CalibratedRecording
from multi_camera.datajoint.multi_camera_dj import Calibration

# Find recordings not yet linked to a calibration
missing = (MultiCameraRecording - CalibratedRecording).proj()
calibration_offset = (Calibration * MultiCameraRecording).proj(
    calibration_offset="ABS(recording_timestamps-cal_timestamp)"
)
# Check viable candidates before assigning
```

#### 4. Run Bridging and EasyMocap

```python
from scripts.session_pipeline import preannotation_session_pipeline

# Run bottom-up bridging + EasyMocap SMPL fitting
preannotation_session_pipeline(keys=your_keys, bottom_up=True, easy_mocap=True)
```

This performs:
- Bottom-up pose detection with bridging method
- EasyMocap tracking across frames
- SMPL model fitting with temporal smoothing to 3D keypoints

#### 5. Annotate Person of Interest

After EasyMocap reconstruction, use the annotation system to select which person to track in the post-annotation pipeline. See the [Annotation Startup Guide](docs/annotation/annotation_startup.md) for instructions on running the annotation Docker container and GUI.

#### 6. Run Post-Annotation Pipeline

```python
from scripts.session_pipeline import postannotation_session_pipeline

# Run top-down person-specific detection and final triangulation
postannotation_session_pipeline(
    keys=your_annotated_keys,
    tracking_method_name="Easymocap",
    top_down_method_name="Bridging_bml_movi_87"
)
```

This performs:
- Person-specific top-down detection (typically copies 3D keypoints from bridging)
- Final triangulation with annotated tracking data

#### 7. Person Keypoint Reconstruction

```python
from multi_camera.datajoint.multi_camera_dj import PersonKeypointReconstruction

# Verify final 3D reconstructions
PersonKeypointReconstruction.populate()
```

## SMPL Model Setup

The pipeline requires SMPL body model files for fitting. This section explains what files are needed and where to place them.

### Minimal SMPL Data Required

For basic SMPL fitting, you only need these files:

```
model_data/
├── smplx/
│   ├── J_regressor_body25.npy              (OpenPose joint mapping)
│   └── smpl/
│       └── SMPL_NEUTRAL.pkl                (Body model for fitting)
└── smpl_clean/
    └── SMPL_NEUTRAL.pkl                    (Body model for annotation)
```

### Obtaining SMPL Models

1. **SMPL Model Files**: Download from the [SMPL website](https://smpl.is.tue.mpg.de/)
   - Requires free registration
   - Download the "SMPL for Python v1.1.0" package

2. **Regressor Files**: Included in the EasyMocap repository
   - `J_regressor_body25.npy` maps SMPL vertices to 25 OpenPose joints

### Directory Structure

Place the files relative to your MultiCameraTracking repository root:

```
MultiCameraTracking/
├── model_data/
│   ├── smplx/
│   │   ├── J_regressor_body25.npy
│   │   └── smpl/
│   │       └── SMPL_NEUTRAL.pkl
│   └── smpl_clean/
│       └── SMPL_NEUTRAL.pkl
```

The code looks for SMPL files in `model_data/` and `model_data/smpl_clean/` relative to the repository.

### Configuration

If you have SMPL files in a different location, you can override the paths with environment variables:

```bash
# Override SMPL location in EasyMocap
export SMPLX_PATH="/path/to/smpl_data/data/smplx"

# Override clean SMPL for annotation
export SMPL_CLEAN_PATH="/path/to/smpl_data/model_data/smpl_clean"
```

Note: Currently MultiCameraTracking uses SMPL by default. Support for SMPLx is available but not commonly used in the standard pipeline.

## Troubleshooting

See the documentation directory for detailed setup and troubleshooting:
- [General System Setup](docs/acquisition/general_system_setup.md)
- [Acquisition Software Setup](docs/acquisition/acquisition_software_setup.md)
- [Annotation Setup](docs/annotation/annotation_software_setup.md)
- [Calibration Procedure](docs/calibration/calibration_procedure.md)

## Key Dependencies

- [PosePipeline](https://github.com/IntelligentSensingAndRehabilitation/PosePipeline) - 2D pose detection
- [EasyMocap](https://github.com/zju3dv/EasyMocap/) - SMPL model fitting
- [DataJoint](https://datajoint.io/) - Database management and pipeline orchestration
- [Aniposelib](https://github.com/lambdaloop/aniposelib) - Camera calibration and triangulation

## Credits

- **Calibration & Triangulation**: Bundle adjustment code from [Aniposelib](https://github.com/lambdaloop/aniposelib)
- **SMPL Fitting**: [EasyMocap](https://github.com/zju3dv/EasyMocap/) implementation
- **2D Pose Estimation**: [PosePipeline](https://github.com/IntelligentSensingAndRehabilitation/PosePipeline) wrappers
- **Data Management**: [DataJoint](https://datajoint.io/) framework
