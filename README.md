
# Introduction

This library supports multiview recordings, pose estimation, fitting SMPL meshes, and exporting the fits into OpenSim. It uses [DataJoint](https://github.com/datajoint) for the data management.

Modules:
- `multi_camera.acquisition` - the recording software
- `multi_camera.analysis` - core analysis code
- `multi_camera.datajoint` - data management and bridges between recording software and analysis code
- `multi_camera.utils` - miscellaneous utilities.


Acquisition is designed to perform simple multi-camera video acquisition with FLIR cameras. These use network synchronization and are tested on BFS-PGE-31S4C. Requires recent firmware to support IEEE1394 synchronization.

# TODO and Warning

This project is under active development and breaking changes may be introduced.

- [ ] Develop a session management schema, ideally that is integrated into acquisition system
- [ ] Migrate existing gaitrite_comparison code and biomechanics code to a validation module
- [ ] Generalize biomechanical reconstructions to any acquisition session

# Usage

## Installation:

    pip install -r requirements
    pip install -e .

## Running Web GUI

First start FastAPI backend, which provides a REST API for the acquisition software

    python -m multi_camera.acquisition.rest_backend

Then start the web GUI

    cd react_frontend
    npm start

## Recording:

    python -m multi_camera.acquisition.record [-h] [-m MAX_FRAMES] [-n NUM_CAMS] [--preview] [-s SCALING] [-c CONFIG] vid_filename

One can either pass a number of frames or hit "Ctrl-C" to stop recording

set_mtu.sh script is used to enable jumbo packets with linux.

## Calibration

To record calibration data, analyze it, and insert it into the database run

    python -m multi_camera.acquisition.record [-n NUM_CAMS] calibration
    python -m multi_camera.datajoint.calibrate_cameras calibration_<basefile>

## Video analysis

Please follow steps from [PosePipeline](github.com/peabody124/PosePipeline)
- Annotate person of interest in videos. Note this currently requires annotating each view.
- Perform top down person keypoint detection

## Triangulation

Insert `CalibratedRecording` to indicate valid combinations of `Calibrations` and `MultiCameraRecording`. To find likely candidates can check:

    calibration_offset = (Calibration * MultiCameraRecording).proj(cal_offset = 'ABS(recording_timestamps-cal_timestamp)')
    calibration_offset = (calibration_offset & 'cal_offset < 10000')
    MultiCameraRecording * calibration_offset - CalibratedRecording

One that is set up, you can insert entries into `PersonKeypointReconstructionMethod` to indicate which `TopDownMethod` to triangulate, then run `PersonKeypointReconsturction.populate()`.

To visualize a skeleton of the raw triangulated coordinates run `PersonKeypointReconstructionVideo.populate()`.

## SMPL Reconstruction

To perform SMPL fitting on the 3D keypoints with some temporal smoothing

    SMPLReconstruction.populate(key)

And to export a TRC file that can be loaded into OpenSim

    (SMPLReconstruction & key).export_trc('outfile.trc')

## Visualization and annotation.

We have CLI support using the EasyMocap visualization. This can be used to select the person
of interest from an Easymocap reconstruction for further top down analysis and visualization
of the top down results. The smpl flag will show the SMPL reconstructions versus stick figures.

    python apps/visualize.py --smpl FILENAME

You can confirm the person you want by filtering

    python apps/visualize.py --smpl FILENAME --filter SUBJECT_ID

And if happy can even annotate accordingly

    python apps/visualize.py --smpl FILENAME --filter SUBJECT_ID --annotate

Finally, it can be used to visualize the results after annotation using the top down flag. This visualizes
the SMPLReconstruction results

    python apps/visualize.py --smpl --top_down FILENAME

# Credits

- Bundle adjustment from [Aniposelib](https://github.com/lambdaloop/aniposelib) is used for the calibration and triangulation.
- [Easymocap](https://github.com/zju3dv/EasyMocap/) is used for fitting SMPL meshes to the 3D joint locations.
- Code from [Pose2Sim](https://github.com/perfanalytics/pose2sim) is used for exporting to OpenSim and models from this repository are used for performing Inverse Kinematics on the extracted keypoints.
