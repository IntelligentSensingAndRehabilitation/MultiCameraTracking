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

General Arguments:
- vid_base: path to calibration file
- charuco: either charuco or checkerboard (default: charuco)
- checkerboard_size: checkerboard square size in mm (default: 109)
- checkerboard_dim: checkerboard dimensions in mm (default: [5,7])
- marker_bits: charuco marker bits (default: 6)
- min_cameras: set minimum number of cameras necessary to fit board pose (default: 2)
- output: output file to save calibration results (default: None)

Releveling Arguments:
- releveling_type: select a type from settings [more details below] (default: no_releveling)
- z_offset_set: board height for stroller/custom releveling [meters] (default: 1.534)
- min_height: set minimum camera height for z_up [meters] (default: 2.0)
- board_ground_time: frame range for ground board (default: [0,200]), used in floor or custom releveling
- board_axis: board axis to align with global axis (default: [0, -1, 0]), used in floor or custom releveling
- global_axis: global axis to align with board axis (default: [0, 0, 1]). used in floor or custom releveling
- board_secondary: secondary axis of the board (default: [1, 0, 0]), used in floor or custom releveling
- global_secondary: secondary global axis (default: [1, 0, 0]), used in floor or custom releveling



Releveling Settings:
- no_releveling: same as original calibration method (no releveling)
- stroller: expects vertical rolling board with constant height. Optimizes calibration by tilting it to minimize the z variability of the board
- z_up: checks camera axes and rotates them to make sure they are upright (assumes y-axis points towards floor). Also shifts camera group up to set lowest camera to a set height above the floor
- ground: assumes the board is on the ground at a known point of time and aligns the global z axis with the board z axis and rotates the camera group accordingly
- custom: sets the global axes to align with the board axes during a known time and shifts group up to a specific height


How to call Calibration Pipeline:
```
entry, board = run_calibration_APL(vidbase, charuco=charuco, 'include all necessary arguments here')
```
After running, a plot of the calibration should be available to view on port 8050 or within the jupyter notebook. Type Y to accept the calibration and insert it into the database.


Example for floor re-leveling method:
```
entry, board = run_calibration_APL(

        #calibration arguments
        vid_base="/mnt/mobile_system_data/20251217/TF91/calibration_20251217_110434",  
        charuco="charuco", # charuco, else checkerboard
        checkerboard_size=109, # checkerboard square size in mm 
        checkerboard_dim=(5,7),  # checkerboard dimensions
        marker_bits=6, # charuco marker bits 

        #releveling arguments 
        releveling_type="floor", #"stroller",  "no_leveling", "z_up", "floor", "custom"
        board = None, # if not set we fall back to default board defined in run_AniposeLib_calibration
        z_offset_set = 0.0, # in meters, height of board when using the stroller or custom releveling 
        min_height=1, # min height of lowest cam in meters when using z_up
        min_cameras=2, # minimum number of cameras to use for fitting board pose
        board_ground_time= (0,90), # range in samples when the board is on the ground/known position, used in floor or custom releveling 
        board_axis=(0,0,-1), # board axis in the board frame to align with global axis, used in floor or custom releveling
        global_axis=(0,0,-1), # global axis in the global frame to align with board axis, used in floor or custom releveling
        board_secondary=(1,0,0), # secondary axis in the board frame to align with global secondary axis, used in floor or custom releveling
        global_secondary=(1,0,0) # global secondary axis in the global frame to align with board secondary axis, used in floor or custom releveling
    )
```

Running from the Command line:
```
python -m multi_camera.datajoint.calibrate_cameras --vid_base "/path/to/calibration/basefile/calibraion_date_time" --charuco charuco --any other arugments here
```