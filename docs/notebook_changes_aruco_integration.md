# Notebook changes for ArUco-calibration integration

Apply these to `/home/vscode/workspace/notebooks/calibration_releveling_APL.ipynb` after the MCT and GA branches land.

## 1. Replace the top "Main" cell

**Before** (current top Main cell):

```python
run = True
if run:
    entry, board = run_calibration_APL(
        vid_base="/mnt/CottonLab/enigma/chair/20251008/calibration_20251008_153438",
        charuco="charuco",
        ...
    )
```

**After**:

```python
run = True
if run:
    from multi_camera.analysis.calibration import run_calibration_and_insert
    from multi_camera.datajoint.aruco import CalibrationArucoDetection
    from gait_analytics.datajoint.ten_meter_walk_test_dj import ArucoWalkwayCalibration

    cal_key = run_calibration_and_insert(
        vid_base="/mnt/CottonLab/enigma/chair/20251008/calibration_20251008_153438",

        # Calibration / releveling args (same as before)
        charuco="charuco",
        checkerboard_size=109,
        checkerboard_dim=(5, 7),
        marker_bits=6,
        releveling_type="z_up",
        z_offset_set=1.534,
        min_height=2.0,
        min_cameras=2,
    )
    print(f"Inserted calibration: {cal_key}")

    # Detection auto-fires only for calibrations whose MultiCameraCalibration.comment
    # contains "aruco" — set in the acquisition GUI at record time and propagated
    # through push-to-DataJoint. (For legacy calibrations created via this notebook
    # without a prior push, pass comment="..." to run_calibration_and_insert above.)
    CalibrationArucoDetection.populate(cal_key)

    # Walkway interpretation only fires for calibrations explicitly flagged as
    # walkway protocol via WalkwayArucoMarkers.
    WalkwayArucoMarkers.insert1(cal_key, skip_duplicates=True)
    ArucoWalkwayCalibration.populate(cal_key)
```

This single cell replaces the old `run_calibration_APL` + `insert_calibration_to_db` flow and the entire bulk-aruco section at the bottom.

## 2. Delete the `insert_calibration_to_db` cell

The function moved into `run_calibration_and_insert`. Old cell is no longer needed.

## 3. Replace the bulk-aruco section with a short retrospective-tagging snippet

**Delete:**
- The `calibration_videos = {...}` hand-maintained dict cell
- The loop that called `ArucoWalkwayCalibration.detect_and_insert(cal_key, vid_base)`

**Keep / update:**
- The per-calibration summary cell — change `ArucoWalkwayCalibration & cal_key` queries to fetch the new `goalposts` field instead of triangulating from `pixel_detections`.
- The QA cells (sample-frame overlay grid, 3D scene plot) — they still work; the only change is `pixel_detections` now lives in `CalibrationArucoDetection` rather than `ArucoWalkwayCalibration`.

**Add this markdown cell (after-the-fact tagging documentation):**

````markdown
## Tag a calibration as having ArUco markers (after the fact)

`CalibrationArucoDetection.populate()` only fires for calibrations whose
`MultiCameraCalibration.comment` contains `"aruco"`. To flag an older
calibration, update its comment to include the substring:

```python
from multi_camera.datajoint.multi_camera_dj import MultiCameraCalibration
from multi_camera.datajoint.aruco import CalibrationArucoDetection
from gait_analytics.datajoint.ten_meter_walk_test_dj import (
    WalkwayArucoMarkers, ArucoWalkwayCalibration,
)

cal_key = {"cal_timestamp": "2026-04-29 15:30:00", "camera_config_hash": "e877011a6e"}

# (MultiCameraCalibration.comment is set at push-to-DataJoint time. If the
# original comment didn't contain "aruco", update it via raw SQL or by
# deleting + re-inserting the row.)

CalibrationArucoDetection.populate(cal_key)

# To also run walkway interpretation:
WalkwayArucoMarkers.insert1(cal_key, skip_duplicates=True)
ArucoWalkwayCalibration.populate(cal_key)
```

Each populate is idempotent — running them again on a key that's already
populated is a no-op. The calibration videos must already be in `Video` /
`CalibrationVideos` (which happens automatically at push-to-DataJoint time
for new sessions, or via `run_calibration_and_insert` for older ones).
````

## 4. Update the QA cells to fetch from the new layout

In the existing 3D-scene cell:

**Before**:
```python
pixel_detections_raw = (ArucoWalkwayCalibration & qa_key).fetch1("pixel_detections")
```

**After**:
```python
pixel_detections_raw = (CalibrationArucoDetection & qa_key).fetch1("pixel_detections")
```

In the goalposts visualization:

**Before**:
```python
qa_goalposts = ArucoWalkwayCalibration().get_goalposts(qa_key)
```

**After**: same line still works — `get_goalposts` now reads from the stored
`goalposts` field on `ArucoWalkwayCalibration`.
