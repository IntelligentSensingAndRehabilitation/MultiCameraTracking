# Data Processing Pipeline

The standard workflow for processing multi-camera recordings from acquisition through final 3D reconstruction is:

## Pipeline Overview

1. **Acquisition** - Record trials and calibration videos using the GUI
2. **Push to DataJoint** - Transfer videos to database via the GUI
3. **Calibration** - Run camera calibration and link to trials
4. **Bridging** - Run bottom-up pose detection across all views
5. **EasyMocap** - Fit SMPL body models to 3D reconstructed keypoints
6. **Post-Annotation Pipeline** - Run top-down person-specific refinement
7. **Person Keypoint Reconstruction** - Final triangulation with annotated data

## Automated Processing

The recommended approach for processing batches of recordings is to use the `session_pipeline.py` script:

```bash
python scripts/session_pipeline.py \
    --participant_id PARTICIPANT_ID \
    --session_date YYYY-MM-DD \
    --run_easymocap \
    --post_annotation
```

This script automatically handles:
- Calibration assignment (links closest valid calibration to each trial)
- Bottom-up bridging across all views
- EasyMocap SMPL fitting with temporal smoothing
- Post-annotation pipeline with person-specific top-down detection

### Optional Flags

- `--project PROJECT_NAME` - Filter by video project
- `--bottom_up` - Run only the bridging step
- `--top_down_method_name METHOD` - Specify top-down detection method (default: Bridging_bml_movi_87)
- `--hand_estimation` - Include hand keypoint estimation

## Step-by-Step Manual Processing

If processing individual recordings manually:

### 1. Record Videos and Calibration

Use the acquisition GUI to record:
- Trial videos with all cameras synchronized
- Calibration videos with checkerboard pattern

See [Acquisition Startup Guide](../acquisition/acquisition_startup.md) for detailed instructions.

### 2. Push Data to DataJoint

Use the GUI to:
1. Import trial videos into `MultiCameraRecording`
2. Import calibration videos into `Calibration`
3. Process calibration to compute camera parameters

### 3. Link Calibration to Trials

The pipeline script automatically assigns calibrations. To manually verify or assign:

```python
from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording, CalibratedRecording
from multi_camera.datajoint.calibrate_cameras import Calibration

# Find recordings not yet linked to a calibration
missing = (MultiCameraRecording - CalibratedRecording).proj()
calibration_offset = (Calibration * MultiCameraRecording).proj(
    calibration_offset="ABS(recording_timestamps-cal_timestamp)"
)
# Check viable candidates before assigning
```

### 4. Run Bridging and EasyMocap

```python
from scripts.session_pipeline import preannotation_session_pipeline

# Run bottom-up bridging + EasyMocap SMPL fitting
preannotation_session_pipeline(keys=your_keys, bottom_up=True, easy_mocap=True)
```

This performs:
- Bottom-up pose detection with bridging method
- EasyMocap tracking across frames
- SMPL model fitting with temporal smoothing to 3D keypoints

### 5. Annotate Person of Interest

After EasyMocap reconstruction, use the annotation system to select which person to track in the post-annotation pipeline. See the [Annotation Startup Guide](../annotation/annotation_startup.md) for instructions on running the annotation Docker container and GUI.

### 6. Run Post-Annotation Pipeline

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

### 7. Person Keypoint Reconstruction

Verify final 3D reconstructions:

```python
from multi_camera.datajoint.multi_camera_dj import PersonKeypointReconstruction

PersonKeypointReconstruction.populate()
```
