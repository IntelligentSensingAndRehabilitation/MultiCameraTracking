
Script that performs simple multi-camera video acquisition with FLIR cameras.

Uses network synchronization and tested on BFS-PGE-31S4C. Requires recent firmware.

Installation:
pip install -r requirements

Running:
python multi_camera.acquisition.record [-h] [-m MAX_FRAMES] [-n NUM_CAMS] [--preview] [-s SCALING] vid_fil

One can either pass a number of frames or hit "Ctrl-C" to stop recording

set_mtu.sh script is used to enable jumbo packets with linux.
