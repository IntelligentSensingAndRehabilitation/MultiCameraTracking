# Calibration

## Calibration Board
- Print out a checkerboard similar to [this one](Charuco7x5_w_margins.pdf)
  on paper/poster of your desired size
- Measure the edge length of one of the black squares. This length in mm should be passed
  as checkerboard_size when running the calibration
- For Charuco boards, the checkerboard_dim is the number of black and white squares in each dimension.
  For Checker boards, it is the number of internal corners in each dimension.

## Running the Calibration
```
python -m multi_camera.datajoint.calibrate_cameras --charuco --checkerboard_size=109 --checkerboard_dim 5,7 /path/to/calibration/base/filename
```