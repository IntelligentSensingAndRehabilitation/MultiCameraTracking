
Script that performs simple multi-camera video acquisition with FLIR cameras.

Uses network synchronization and tested on BFS-PGE-31S4C. Requires recent firmware.

Installation:
pip install -r requirements

Running:
python -m multi_camera.acquisition.record [-h] [-m MAX_FRAMES] [-n NUM_CAMS] [--preview] [-s SCALING] vid_fil

One can either pass a number of frames or hit "Ctrl-C" to stop recording

set_mtu.sh script is used to enable jumbo packets with linux.

## Calibration

To record calibration data, analyze it, and insert it into the database run

    python -m multi_camera.acquisition.record [-n NUM_CAMS] calibration
    python -m multi_camera.datajoint.calibrate_cameras calibration_<basefile>