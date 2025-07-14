import numpy as np
from aniposelib.boards import CharucoBoard, Checkerboard
from aniposelib.cameras import Camera, CameraGroup
import os
import glob
import re
import json
from datetime import datetime
import cv2
import argparse
import plotly.offline as pyo
import plotly.graph_objects as go
from dash import Dash, dcc, html
import subprocess
import os
import threading
import time


def run_calibration_APL(
        vid_base, 
        charuco="charuco", 
        checkerboard_size=109, 
        checkerboard_dim=(5,7), 
        marker_bits=6,
        releveling_type="no_leveling", # "no_leveling", "z_up", "stroller", "floor", "custom"
        board = None, # if not set we fall back to default board defined in run_AniposeLib_calibration
        z_offset_set = 1.534, # in meters, height of board when using the stroller or custom releveling 
        min_height=2.0, # min height of lowest cam in meters when using z_up
        min_cameras=2, # minimum number of cameras to use for fitting board pose
        board_ground_time= (0,200), # range in samples when the board is on the ground/known position, used in floor or custom releveling 
        board_axis=(0,-1,0), # board axis in the board frame to align with global axis, used in floor or custom releveling
        global_axis=(0,0,1), # global axis in the global frame to align with board axis, used in floor or custom releveling
        board_secondary=(1,0,0), # secondary axis in the board frame to align with global secondary axis, used in floor or custom releveling
        global_secondary=(1,0,0) # global secondary axis in the global frame to align with board secondary axis, used in floor or custom releveling
        ):
    #APL calibration
    entry, board = run_AniposeLib_calibration(vid_base, charuco=charuco, checkerboard_size=checkerboard_size, checkerboard_dim=checkerboard_dim, marker_bits=marker_bits)
    
    # Releveling calibration
    params_dict_levelled, per_frame_p3ds_transformed, poses_transformed = relevel_calibration( entry= entry, releveling_type=releveling_type, board=board, 
                                                                                              z_offset_set=z_offset_set, min_height=min_height, 
                                                                                              min_cameras=min_cameras, board_ground_time=board_ground_time, 
                                                                                              board_axis=board_axis, global_axis=global_axis, board_secondary=board_secondary, 
                                                                                              global_secondary=global_secondary)
    print(f"[run_calibration_APL] Releveling completed with method: '{releveling_type}'")

    # Update entry with releveling parameters
    entry["camera_calibration"] = params_dict_levelled
    # Plotting
    plot_per_frame_p3ds_with_board_slider(per_frame_p3ds_transformed, poses_transformed, board, params_dict_levelled, entry = entry)

    return entry, board
    #db insert
    

def relevel_calibration(
        entry,
        releveling_type = "stroller", 
        z_offset_set = 1.534, # in meters
        min_height=2.0, # in meters
        min_cameras=2, 
        board=None,
        board_ground_time= (0,200), # in samples when the board is on the ground 
        board_axis=(0,-1,0), 
        global_axis=(0,0,1), 
        board_secondary=(1,0,0), 
        global_secondary=(1,0,0)
        ):
    print(f"\n[relevel_calibration] Starting releveling with method: '{releveling_type}'")

    cal_points = entry["calibration_points"]
    cgroup = recreate_cgroup_from_entry(entry)
    params_dict = entry["camera_calibration"] 

    # Triangulate 3D points and get IDs
    per_frame_p3ds, per_frame_ids = triangulate_from_calibration_points(cal_points, cgroup, min_cameras=min_cameras)

    if releveling_type == "no_leveling":
        poses = fit_charuco_board_pose_to_p3ds(per_frame_p3ds, per_frame_ids, board)
        return params_dict, per_frame_p3ds, poses

    elif releveling_type == "z_up":
        params_dict_zup, R_rot, shift = rotate_calibration_z_up_fixed(params_dict, min_height=min_height)
        poses = fit_charuco_board_pose_to_p3ds(per_frame_p3ds, per_frame_ids, board)
        per_frame_p3ds_zup_transformed = transform_and_shift_p3ds_dict(per_frame_p3ds, R_rot, shift)
        poses_zup_transformed = transform_and_shift_board_poses(poses, R_rot, shift)
        return params_dict_zup, per_frame_p3ds_zup_transformed, poses_zup_transformed

    elif releveling_type == "stroller":
        poses = fit_charuco_board_pose_to_p3ds(per_frame_p3ds, per_frame_ids, board)
        params_dict_levelled, best_frame, fit_error, R_global, shift = align_calibration_to_board_best_frame(
            poses, per_frame_ids, per_frame_p3ds, (0, len(per_frame_p3ds)-1), board, params_dict,
            board_axis=(0,-1,0), global_axis=(0,0,1), board_secondary=(1,0,0), global_secondary=(1,0,0), z_offset= z_offset_set
        )
        per_frame_p3ds_stroller_transformed = transform_and_shift_p3ds_dict(per_frame_p3ds, R_global, shift)
        poses_stroller_transformed = transform_and_shift_board_poses(poses, R_global, shift)

        params_dict_flat, poses_flat, R_flat, res = optimize_calibration_tilt_to_flatten_board_z(params_dict_levelled, poses_stroller_transformed)
        per_frame_p3ds_flat = transform_and_shift_p3ds_dict(per_frame_p3ds_stroller_transformed, R_flat)

        params_dict_flat2, per_frame_p3ds_flat2, poses_flat2, shift_vec = shift_outputs_to_board_z(
            params_dict_flat, per_frame_p3ds_flat, poses_flat, z_offset_set=z_offset_set, use_median=False)
        return params_dict_flat2, per_frame_p3ds_flat2, poses_flat2

    elif releveling_type == "floor":
        z_offset_set = 0.0
        poses = fit_charuco_board_pose_to_p3ds(per_frame_p3ds, per_frame_ids, board)
        params_dict_levelled, best_frame, fit_error, R_global, shift = align_calibration_to_board_best_frame(
            poses, per_frame_ids, per_frame_p3ds, board_ground_time, board, params_dict,
            board_axis=(0,0,1), global_axis=(0,0,1), board_secondary=(1,0,0), global_secondary=(1,0,0), z_offset= z_offset_set
        )
        per_frame_p3ds_ground_transformed = transform_and_shift_p3ds_dict(per_frame_p3ds, R_global, shift)
        poses_ground_transformed = transform_and_shift_board_poses(poses, R_global, shift)
        params_dict_ground, per_frame_p3ds_ground, poses_ground, shift_vec = shift_outputs_to_frame_z(
            params_dict_levelled, per_frame_p3ds_ground_transformed, poses_ground_transformed, best_frame=best_frame, z_offset_set=z_offset_set)
        return params_dict_ground, per_frame_p3ds_ground, poses_ground

    elif releveling_type == "custom":
        poses = fit_charuco_board_pose_to_p3ds(per_frame_p3ds, per_frame_ids, board)
        params_dict_levelled, best_frame, fit_error, R_global, shift = align_calibration_to_board_best_frame(
            poses, per_frame_ids, per_frame_p3ds, board_ground_time, board, params_dict,
            board_axis=board_axis, global_axis=global_axis, board_secondary=board_secondary, global_secondary=global_secondary, z_offset= z_offset_set
        )
        per_frame_p3ds_ground_transformed = transform_and_shift_p3ds_dict(per_frame_p3ds, R_global, shift)
        poses_ground_transformed = transform_and_shift_board_poses(poses, R_global, shift)
        params_dict_custom, per_frame_p3ds_custom, poses_custom, shift_vec = shift_outputs_to_frame_z(
            params_dict_levelled, per_frame_p3ds_ground_transformed, poses_ground_transformed, best_frame=best_frame, z_offset_set=z_offset_set)
        return params_dict_custom, per_frame_p3ds_custom, poses_custom
    
def run_AniposeLib_calibration(vid_base = None, charuco="charuco", checkerboard_size=109, checkerboard_dim=(5,7), marker_bits=6):
    
    vid_path, vid_base = os.path.split(vid_base) # vidbase full path to video without .camera.mp4 e.g. /mnt/CottonLab/mobile_system_data/20250122/q8o77/calibration_20250122_181045

    print(vid_path, vid_base)
    all_videos = glob.glob(os.path.join(vid_path, f"{vid_base}.*.mp4"))

    if not all_videos:
        print(f"No videos found for {vid_base}")
        

    # Extract camera IDs
    camera_videos = {}
    for video_path in all_videos:
        filename = os.path.basename(video_path)
        match = re.search(r'\.(\d+)\.mp4$', filename)
        if match:
            camera_id = int(match.group(1))
            camera_videos[camera_id] = [video_path]

    if not camera_videos:
        print(f"Could not extract camera IDs from videos {vid_base}")
        

    # Define camera names
    cam_names = []
    calib_videos = []
    original_cam_ids = []

    for cam_id in sorted(camera_videos.keys()):
        
        cam_names.append(str(cam_id))
        calib_videos.append(camera_videos[cam_id])
        original_cam_ids.append(str(cam_id))

    print(f"Found {len(cam_names)} cameras: {cam_names}")
    print(f"Using these videos:")
    for i, videos in enumerate(calib_videos):
        print(f"  {cam_names[i]}: {[os.path.basename(v) for v in videos]}")

    # Define the CharucoBoard parameters
    if charuco == "charuco":
        board = CharucoBoard(squaresX=checkerboard_dim[1], squaresY=checkerboard_dim[0], square_length=checkerboard_size, marker_length=0.8*checkerboard_size, 
                            marker_bits=marker_bits, dict_size=250, manually_verify=False)
        print(f"Using CharucoBoard: {checkerboard_dim}, {checkerboard_size}, marker_bits={marker_bits}")
    else:
        board = Checkerboard(squaresX=checkerboard_dim[1], squaresY=checkerboard_dim[0], square_length=checkerboard_size) #never tried
        print(f"Using Checkerboard: {checkerboard_dim}, {checkerboard_size}")


    # Initialize camera group and calibrate
    print("Starting calibration...")

    cgroup = CameraGroup.from_names(cam_names, fisheye=False)
    error, all_rows = cgroup.calibrate_videos(calib_videos, board)
    num_frames = get_num_frames_from_videos(vid_path, vid_base, cam_names)
    num_corners = len(board.objPoints) 
    calibration_points = extract_calibration_points_from_all_rows(all_rows, cam_names,num_frames=num_frames, num_corners=num_corners)

    # Get camera dictionaries from camera group
    cam_dicts = cgroup.get_dicts()

    # Format camera parameters for database as required by MultiCameraTracking
    params_dict = {
        "mtx": np.array([
            [c["matrix"][0][0], c["matrix"][1][1], c["matrix"][0][2], c["matrix"][1][2]]
            for c in cam_dicts
        ]) / 1000.0,
        "dist": np.array([c["distortions"] for c in cam_dicts]),
        "rvec": np.array([c["rotation"] for c in cam_dicts]),
        "tvec": np.array([c["translation"] for c in cam_dicts]) / 1000.0,
    }

 
    # Create simplified placeholder for calibration points and shape
    num_cameras = len(cam_names)
    
    calibration_shape = np.zeros((num_corners, 3))  # For a 4x6 grid

    # Try to get config hash from JSON file
    config_hash = None
    json_file = os.path.join(vid_path, f"{vid_base}.json")
    if os.path.exists(json_file):
        try:
            with open(json_file, "r") as f:
                json_data = json.load(f)
                if "camera_config_hash" in json_data:
                    config_hash = json_data["camera_config_hash"]
        except Exception as e:
            print(f"Warning: Could not read JSON file: {e}")

    # If no config hash from JSON, create a simple hash from camera IDs
    if not config_hash:
        import hashlib
        config_hash = hashlib.md5(''.join(original_cam_ids).encode('utf-8')).hexdigest()[:10]
        print(f"Using generated config hash: {config_hash}")

    # Convert timestamp string to datetime object
    try:
        trial_timestamp = vid_base[len("calibration_"):]
        cal_timestamp = datetime.strptime(trial_timestamp, "%Y%m%d_%H%M%S")
    except ValueError:
        print(f"Warning: Could not parse vid_base {vid_base}")
    # Create the database entry
    entry = {
        "cal_timestamp": cal_timestamp,
        "camera_config_hash": config_hash,
        "recording_base": vid_base,
        "num_cameras": num_cameras,
        "camera_names": cam_names,
        "camera_calibration": params_dict,
        "reprojection_error": error,
        "calibration_points": calibration_points,
        "calibration_shape": calibration_shape,
        "calibration_type": "charuco"
    }

    return entry, board



def extract_calibration_points_from_all_rows(all_rows, cam_names, num_frames=None, num_corners=24):
    """
    Extract calibration points into shape (num_cameras, num_frames, num_corners, 2).
    Pads missing detections with np.nan.

    Args:
        all_rows: list of per-camera detection dicts
        cam_names: list of camera names
        num_frames: int, total number of frames (if None, auto-detects max frame index + 1)
        num_corners: int, number of board markers (default 24 for 7x5 Charuco)
    Returns:
        np.ndarray of shape (num_cameras, num_frames, num_corners, 2)
    """
    import numpy as np

    num_cameras = len(cam_names)

    # Auto-detect number of frames if not provided
    if num_frames is None:
        max_frame = -1
        for cam_rows in all_rows:
            for detection in cam_rows:
                frame_idx = detection.get('prefix')
                if frame_idx is None and 'framenum' in detection:
                    frame_idx = detection['framenum'][1]
                if frame_idx is not None:
                    max_frame = max(max_frame, int(frame_idx))
        num_frames = max_frame + 1

    arr = np.full((num_cameras, num_frames, num_corners, 2), np.nan, dtype=np.float32)

    for cam_idx, cam_rows in enumerate(all_rows):
        for detection in cam_rows:
            # Get frame index
            frame_idx = detection.get('prefix')
            if frame_idx is None and 'framenum' in detection:
                frame_idx = detection['framenum'][1]
            if frame_idx is None or frame_idx >= num_frames:
                continue
            ids = detection.get('ids', [])
            corners = detection.get('corners', [])
            # Flatten ids if needed
            if hasattr(ids, 'flatten'):
                ids = ids.flatten()
            for id_, corner in zip(ids, corners):
                # Convert id_ to int
                if isinstance(id_, np.ndarray):
                    id_ = id_.item()
                id_ = int(id_)
                if 0 <= id_ < num_corners:
                    arr[cam_idx, int(frame_idx), id_] = np.array(corner).flatten()[:2]
    return arr

def rotate_calibration_z_up_fixed(camera_params, min_height=2.0):
    """
    Rotate calibration data to make Z-axis point up and ensure cameras are properly oriented
    with Y-axes pointing downward (negative Z). Applies a unified transformation to all cameras.
    
    Parameters:
        camera_params (dict): Dictionary with camera calibration parameters
        min_height (float): Minimum height (in meters) for cameras above ground
        
    Returns:
        dict: Modified camera calibration parameters
    """
    import cv2
    import numpy as np
    import copy
    
    # Create a deep copy to avoid modifying the original
    params = copy.deepcopy(camera_params)
    
    # Extract vectors
    tvec = params["tvec"]
    rvec = params["rvec"]
    
    # Convert rotation vectors to rotation matrices
    rmats = [cv2.Rodrigues(np.array(r[None, :]))[0] for r in rvec]
    
    # Calculate camera positions in current space
    camera_positions = np.array([-np.linalg.inv(R).dot(t) for R, t in zip(rmats, tvec)])
    
    # Step 1: First transformation - make Z-axis point up
    R_transform = np.array([
        [1, 0, 0],
        [0, 0, -1],  # Y becomes negative Z
        [0, 1, 0]    # Z becomes Y
    ])
    
    # Step 2: Check if we need to flip all cameras to make Y point downward
    # Apply the transformation to the first camera's rotation matrix
    first_R_transformed = rmats[0] @ R_transform.T
    
    # The second column of the rotation matrix represents the Y-axis direction in world coordinates
    first_y_axis = first_R_transformed[:, 1]
    
    # Check if the Z component (index 2) of the Y-axis is positive
    # If so, the Y-axis is pointing upward after transformation
    # We need all cameras' Y-axes to point downward (negative Z component)
    need_flip = first_y_axis[2] > 0  # Z component is positive = pointing up
    
    print(f"First camera's Y-axis after initial transform has Z component: {first_y_axis[2]:.4f}")
    
    # Always apply the flip to force cameras to be upright
    # This ensures cameras have Y-axis pointing down consistently
    # regardless of initial orientation
    flip_matrix = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    
    if need_flip:
        print("Y-axis has positive Z component (pointing up) - applying 180° flip")
        # Combine transformations
        full_transform = R_transform @ flip_matrix
    else:
        print("Y-axis already has negative Z component (pointing down)")
        # Force the flip anyway to ensure consistency
        full_transform = R_transform @ flip_matrix
        print("Applying 180° flip to ensure all cameras are consistently oriented")
    
    # Apply the unified transformation to all cameras
    new_rmats = []
    new_tvecs = []
    new_positions = []
    
    for i, (R, t) in enumerate(zip(rmats, tvec)):
        # Apply full rotation to camera orientation
        new_R = R @ full_transform.T
        
        # Calculate new camera position
        new_pos = full_transform @ camera_positions[i]
        
        # Calculate new tvec
        new_tvec = -new_R @ new_pos
        
        new_rmats.append(new_R)
        new_tvecs.append(new_tvec)
        new_positions.append(new_pos)
    
    # Convert lists to numpy arrays
    new_rvecs = np.array([cv2.Rodrigues(R)[0].flatten() for R in new_rmats])
    new_tvecs = np.array(new_tvecs)
    new_positions = np.array(new_positions)
    
    # Adjust heights to ensure minimum height above ground
    lowest_z = np.min(new_positions[:, 2])
    
    if lowest_z < min_height:
        height_adjustment = min_height - lowest_z
        
        # Apply the height adjustment to all camera positions
        for i in range(len(new_positions)):
            new_positions[i, 2] += height_adjustment
            
            # Recalculate tvec based on the new position
            new_tvecs[i] = -new_rmats[i] @ new_positions[i]
    
    # Verify the orientation of cameras after transformation
    downward_y_count = 0
    for i, R in enumerate(new_rmats):
        y_axis_z_component = R[:, 1][2]  # Z component of Y-axis
        is_pointing_down = y_axis_z_component < 0
        downward_y_count += is_pointing_down
        print(f"Camera {i} Y-axis Z component: {y_axis_z_component:.4f} ({'down' if is_pointing_down else 'up'})")
    
    # Update the calibration parameters with new values
    params["rvec"] = new_rvecs
    params["tvec"] = new_tvecs
    shift = [0, 0, height_adjustment]
    
    print(f"Calibration rotated with consistent transformation for all cameras.")
    print(f"{downward_y_count}/{len(new_rmats)} cameras have Y-axis pointing downward (negative Z).")
    print(f"All cameras are now at least {min_height}m above ground level.")
    print(f"Camera heights (Z): {new_positions[:, 2]}")
    
    return params, full_transform, shift


def rotate_calibration_arbitrary(params_dict, R_global):
    """
    Apply an arbitrary global rotation to the entire camera group,
    using the same logic as rotate_calibration_z_up_fixed.
    """
    import copy
    import cv2
    import numpy as np

    params = copy.deepcopy(params_dict)
    tvec = params["tvec"]
    rvec = params["rvec"]

    # Convert rotation vectors to rotation matrices
    rmats = [cv2.Rodrigues(np.array(r[None, :]))[0] for r in rvec]

    # Calculate camera positions in world space
    camera_positions = np.array([-np.linalg.inv(R).dot(t) for R, t in zip(rmats, tvec)])

    new_rmats = []
    new_tvecs = []
    new_positions = []

    for i, (R, t) in enumerate(zip(rmats, tvec)):
        # Apply global rotation to orientation
        new_R = R @ R_global.T
        # Apply global rotation to position
        new_pos = R_global @ camera_positions[i]
        # Recompute tvec
        new_tvec = -new_R @ new_pos

        new_rmats.append(new_R)
        new_tvecs.append(new_tvec)
        new_positions.append(new_pos)

    # Convert to rvecs and tvecs arrays
    new_rvecs = np.array([cv2.Rodrigues(R)[0].flatten() for R in new_rmats])
    new_tvecs = np.array(new_tvecs)

    params["rvec"] = new_rvecs
    params["tvec"] = new_tvecs
    return params


def plot_per_frame_p3ds_with_board_slider(per_frame_p3ds, poses, board, params_dict=None, cam_names=None,entry = None):
    """
    Like plot_per_frame_p3ds_with_slider, but overlays the fitted Charuco board pose for each frame.
    Uses aspectmode='cube' so all axes have the same scale and spacing.
    Shows the board Z height in the slider label.
    """
    import numpy as np
    import plotly.graph_objects as go
    import cv2
    cam_names = entry["camera_names"]

    def get_camera_traces_and_positions(params_dict, cam_names=None):
        tvec = params_dict["tvec"]
        rvec = params_dict["rvec"]
        rmats = [cv2.Rodrigues(np.array(r[None, :]))[0].T for r in rvec]
        pos = np.array([-R.dot(t) for R, t in zip(rmats, tvec)])
        labels = cam_names if cam_names is not None else [f"Camera {i}" for i in range(len(pos))]
        traces = [
            go.Scatter3d(
                x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                mode='markers+text',
                marker=dict(size=4, color='red'),
                text=labels,
                textposition="top center",
                name='Camera Positions'
            )
        ]
        for i, (R, t) in enumerate(zip(rmats, pos)):
            length = np.max(np.abs(pos)) * 0.2
            x_axis = R.dot(np.array([length, 0, 0])) + t
            y_axis = R.dot(np.array([0, length, 0])) + t
            z_axis = R.dot(np.array([0, 0, length])) + t
            traces.append(go.Scatter3d(
                x=[t[0], x_axis[0]], y=[t[1], x_axis[1]], z=[t[2], x_axis[2]],
                mode='lines', line=dict(color='red', width=4),
                name=f'X-axis {i}' if i==0 else '',
                showlegend=i==0
            ))
            traces.append(go.Scatter3d(
                x=[t[0], y_axis[0]], y=[t[1], y_axis[1]], z=[t[2], y_axis[2]],
                mode='lines', line=dict(color='green', width=4),
                name=f'Y-axis {i}' if i==0 else '',
                showlegend=i==0
            ))
            traces.append(go.Scatter3d(
                x=[t[0], z_axis[0]], y=[t[1], z_axis[1]], z=[t[2], z_axis[2]],
                mode='lines', line=dict(color='blue', width=4),
                name=f'Z-axis {i}' if i==0 else '',
                showlegend=i==0
            ))
        return traces, pos

    # Compute board Z for each frame
    board_z_per_frame = {}
    for frame_idx in poses:
        rvec, tvec, _ = poses[frame_idx]
        t = tvec.reshape(3)
        board_z_per_frame[frame_idx] = float(t[2])

    # Collect all points to determine axis limits
    all_pts = []
    for pts in per_frame_p3ds.values():
        arr = np.asarray(pts)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            if arr.size % 3 == 0:
                arr = arr.reshape(-1, 3)
            else:
                continue
        elif arr.shape[1] != 3:
            continue
        all_pts.append(arr)
    all_pts = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3))

    if params_dict is not None:
        camera_traces, cam_pos = get_camera_traces_and_positions(params_dict, cam_names)
    else:
        camera_traces, cam_pos = [], np.zeros((0, 3))

    # Combine all points for axis limits
    if all_pts.size and cam_pos.size:
        all_xyz = np.vstack([all_pts, cam_pos])
    elif all_pts.size:
        all_xyz = all_pts
    elif cam_pos.size:
        all_xyz = cam_pos
    else:
        all_xyz = np.zeros((1, 3))

    # Compute cube bounds for equal spacing
    mins = np.nanmin(all_xyz, axis=0)
    maxs = np.nanmax(all_xyz, axis=0)
    centers = (mins + maxs) / 2
    span = np.max(maxs - mins)
    margin = 0.05
    span *= (1 + margin)
    x_range = [centers[0] - span/2, centers[0] + span/2]
    y_range = [centers[1] - span/2, centers[1] + span/2]
    z_range = [centers[2] - span/2, centers[2] + span/2]

    # --- Add ground plane at z=0 ---
    x_plane = np.linspace(x_range[0], x_range[1], 2)
    y_plane = np.linspace(y_range[0], y_range[1], 2)
    xx, yy = np.meshgrid(x_plane, y_plane)
    zz = np.zeros_like(xx)
    ground_plane = go.Surface(
        x=xx, y=yy, z=zz,
        showscale=False,
        opacity=1,
        colorscale=[[0, 'lightgray'], [1, 'lightgray']],
        name='Ground (z=0)',
        hoverinfo='skip'
    )
    camera_traces = [ground_plane] + camera_traces

    frames = []
    frame_numbers = sorted(per_frame_p3ds.keys())
    for frame_idx in frame_numbers:
        pts = per_frame_p3ds[frame_idx]
        arr = np.asarray(pts)
        if arr.size == 0:
            scatter = go.Scatter3d(x=[], y=[], z=[], mode='markers', marker=dict(size=1, color='blue'), name=f'Frame {frame_idx}')
        else:
            if arr.ndim == 1:
                if arr.size % 3 == 0:
                    arr = arr.reshape(-1, 3)
                else:
                    arr = np.full((0, 3), np.nan)
            elif arr.shape[1] != 3:
                arr = np.full((0, 3), np.nan)
            scatter = go.Scatter3d(
                x=arr[:, 0], y=arr[:, 1], z=arr[:, 2],
                mode='markers', marker=dict(size=1, color='blue'),
                name=f'Frame {frame_idx}'
            )
        # Add board if pose is available
        board_trace = []
        if frame_idx in poses:
            rvec, tvec, _ = poses[frame_idx]
            board_trace = [get_charuco_board_trace(rvec, tvec, board, color='orange', width=3)]
            board_trace += get_board_axes_trace(rvec, tvec, length=0.2)
        frames.append(go.Frame(data=camera_traces + [scatter] + board_trace, name=str(frame_idx)))

    # --- Build slider steps with z height in label ---
    steps = []
    for i, frame in enumerate(frames):
        frame_idx = int(frame.name)
        z_val = board_z_per_frame.get(frame_idx, None)
        if z_val is not None:
            label = f"{frame_idx} (z={z_val:.3f} m)"
        else:
            label = str(frame_idx)
        step = dict(
            method="animate",
            args=[[frame.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
            label=label
        )
        steps.append(step)

    # --- Set initial currentvalue to first frame's z ---
    first_idx = int(frames[0].name)
    first_z = board_z_per_frame.get(first_idx, None)
    if first_z is not None:
        currentvalue_prefix = f"Frame (z={first_z:.3f} m): "
    else:
        currentvalue_prefix = "Frame: "

    sliders = [dict(
        active=0,
        currentvalue={"prefix": currentvalue_prefix},
        pad={"t": 50},
        steps=steps
    )]

    fig = go.Figure(
        data=frames[0].data,
        frames=frames
    )

    fig.update_layout(
        sliders=sliders,
        updatemenus=[{
            "type": "buttons",
            "buttons": [{
                "label": "Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 33, "redraw": True}, "fromcurrent": True}]
            }]
        }],
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='cube',
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(range=z_range)
        ),
        title="Triangulated 3D Keypoints Per Frame + Fitted Board (Slider)",
        width=1000,
        height=800
    )

    # Function to serve the Plotly figure using Dash
    def kill_process_on_port(port):
        """Kill any process using the specified port."""
        try:
            # Find the process using the port
            result = subprocess.run(
                ["lsof", "-t", f"-i:{port}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            process_ids = result.stdout.strip().split("\n")
            
            # Kill the process if found
            for pid in process_ids:
                if pid:
                    os.system(f"kill -9 {pid}")
                    print(f"Killed process {pid} using port {port}")
        except Exception as e:
            print(f"Could not kill process on port {port}: {e}")

    def serve_plot(fig, port=8050, delay=10):
        """Serve the Plotly figure using Dash."""
        app = Dash(__name__)
        app.layout = html.Div([
            html.H1("Plotly Figure Display"),
            dcc.Graph(figure=fig)
        ])

        url = f"http://localhost:{port}"
        print(f"Serving the figure at {url}")

        # Auto-open the URL in the host's browser
        #os.system(f"$BROWSER {url}")

        # Run the server in a separate thread
        def run_server():
            app.run(host="0.0.0.0", port=port)

        server_thread = threading.Thread(target=run_server)
        server_thread.start()

        # Wait for the specified delay, then stop the server
        print(f"Waiting for {delay} seconds before stopping the server...")
        time.sleep(delay)
        print("Stopping the server...")
        


    # Show on port 8050
    serve_plot(fig, port=8050)



def fit_charuco_board_pose_to_p3ds(per_frame_p3ds, per_frame_ids, board, min_points=4):
    """
    For each frame, fit a rigid Charuco board pose to the triangulated 3D points.
    Returns: dict {frame_idx: (rvec, tvec, used_ids)} for frames with enough points.
    The returned rvec, tvec are board->world (world-frame pose).
    """
    import numpy as np
    import cv2

    board_points = board.objPoints
    id_to_board_point = {i: pt/1000.0 for i, pt in enumerate(board_points)}
    poses = {}
    for frame_idx, p3ds in per_frame_p3ds.items():
        ids = per_frame_ids[frame_idx]
        obj_points = []
        scene_points = []
        for id_, p3d in zip(ids, p3ds):
            if id_ in id_to_board_point:
                obj_points.append(id_to_board_point[id_])
                scene_points.append(p3d)
        obj_points = np.array(obj_points, dtype=np.float32)
        scene_points = np.array(scene_points, dtype=np.float32)
        if obj_points.shape[0] < min_points or scene_points.shape[0] < min_points:
            continue
        centroid_obj = obj_points.mean(axis=0)
        centroid_scene = scene_points.mean(axis=0)
        X = scene_points - centroid_scene
        Y = obj_points - centroid_obj
        H = X.T @ Y
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = centroid_obj - R @ centroid_scene

        # --- Invert the pose to get board->world ---
        R_inv = R.T
        t_inv = -R.T @ t
        rvec_inv, _ = cv2.Rodrigues(R_inv)
        poses[frame_idx] = (rvec_inv, t_inv.reshape(3, 1), ids)
    return poses


def get_charuco_board_trace(rvec, tvec, board, color='orange', width=2, name='Charuco Board'):
    """
    Returns a Plotly Scatter3d trace for the Charuco board outline at the given pose.
    """
    import numpy as np
    import cv2
    import plotly.graph_objects as go

    # Board size in meters
    w = (board.squaresX - 1) * board.square_length / 1000.0
    h = (board.squaresY - 1) * board.square_length / 1000.0

    # Board corners in board frame
    corners = np.array([
        [0, 0, 0],
        [w, 0, 0],
        [w, h, 0],
        [0, h, 0],
        [0, 0, 0]
    ], dtype=np.float32)

    # Transform corners to world frame
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    world_corners = (R @ corners.T).T + t

    return go.Scatter3d(
        x=world_corners[:, 0], y=world_corners[:, 1], z=world_corners[:, 2],
        mode='lines',
        line=dict(color=color, width=width),
        name=name,
        showlegend=False
    )

def get_board_axes_trace(rvec, tvec, length=0.2, name='Board Axes'):
    """
    Returns Plotly Scatter3d traces for the board's coordinate axes at the given pose.
    """
    import numpy as np
    import cv2
    import plotly.graph_objects as go

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    origin = t
    axes = np.eye(3) * length  # X, Y, Z axes

    # Transform axes to world frame
    axes_world = (R @ axes.T).T + origin

    colors = ['red', 'green', 'blue']
    names = ['X', 'Y', 'Z']
    traces = []
    for i in range(3):
        traces.append(go.Scatter3d(
            x=[origin[0], axes_world[i, 0]],
            y=[origin[1], axes_world[i, 1]],
            z=[origin[2], axes_world[i, 2]],
            mode='lines',
            line=dict(color=colors[i], width=6),
            name=f'{name} {names[i]}',
            showlegend=False
        ))
    return traces


def align_calibration_to_board_best_frame(
    poses,
    per_frame_ids,
    per_frame_p3ds,
    frame_range,
    board,
    params_dict,
    
    board_axis=(0, 0, 1),
    global_axis=(0, 0, 1),
    board_secondary=(1, 0, 0),
    global_secondary=(1, 0, 0),
    z_offset=0.0
):
    """
    Aligns the global coordinate system to the Charuco board in the best frame within a range.
    First selects frames with the most detected points, then picks the one with the lowest fit error.
    Returns new calibration dict, best frame index, fit error, R_global, and shifted per_frame_p3ds and poses.
    """
    import numpy as np
    import cv2

    def get_axis(R, axis):
        idx = np.argmax(np.abs(axis))
        sign = np.sign(axis[idx])
        return R[:, idx] * sign

    # 1. Select frames in range
    start, end = frame_range
    candidate_frames = [f for f in poses if start <= f <= end]
    if not candidate_frames:
        raise ValueError("No frames in specified range with board pose.")

    # 2. Count detections per frame
    frame_counts = {f: len(per_frame_ids.get(f, [])) for f in candidate_frames}
    max_count = max(frame_counts.values())
    # Only keep frames with the maximum number of detections
    best_count_frames = [f for f, cnt in frame_counts.items() if cnt == max_count]

    # 3. Compute fit error for these frames
    fit_errors = {}
    for f in best_count_frames:
        rvec, tvec, ids = poses[f]
        p3ds = per_frame_p3ds.get(f, None)
        if p3ds is None or len(p3ds) == 0:
            fit_errors[f] = np.inf
            continue
        board_points = board.objPoints
        id_to_board_point = {i: pt/1000.0 for i, pt in enumerate(board_points)}
        obj_points = []
        scene_points = []
        for id_, p3d in zip(ids, p3ds):
            if id_ in id_to_board_point:
                obj_points.append(id_to_board_point[id_])
                scene_points.append(p3d)
        obj_points = np.array(obj_points, dtype=np.float32)
        scene_points = np.array(scene_points, dtype=np.float32)
        if obj_points.shape[0] < 3:
            fit_errors[f] = np.inf
            continue
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)
        obj_points_world = (R @ obj_points.T).T + t
        rms = np.sqrt(np.mean(np.sum((obj_points_world - scene_points)**2, axis=1)))
        fit_errors[f] = rms

    # 4. Pick best frame (lowest fit error)
    finite_errors = {f: e for f, e in fit_errors.items() if np.isfinite(e)}
    if finite_errors:
        best_frame = min(finite_errors, key=finite_errors.get)
        best_fit_error = finite_errors[best_frame]
    else:
        # fallback: just pick the first with max detections
        best_frame = best_count_frames[0]
        best_fit_error = None

    rvec, tvec, ids = poses[best_frame]
    R_board, _ = cv2.Rodrigues(rvec)
    t_board = tvec.reshape(3)

    # 5. Compute alignment rotation (world-to-board)
    board_axes = np.eye(3)
    board_axes_world = R_board @ board_axes
    board_axis_vec = np.array(board_axis, dtype=np.float32)
    global_axis_vec = np.array(global_axis, dtype=np.float32)
    board_secondary_vec = np.array(board_secondary, dtype=np.float32)
    global_secondary_vec = np.array(global_secondary, dtype=np.float32)
    src_main = get_axis(board_axes_world, board_axis_vec)
    tgt_main = global_axis_vec / np.linalg.norm(global_axis_vec)
    v = np.cross(src_main, tgt_main)
    c = np.dot(src_main, tgt_main)
    if np.linalg.norm(v) < 1e-8:
        R_align_main = np.eye(3) if c > 0 else -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R_align_main = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v)**2))
    board_axes_aligned = R_align_main @ board_axes_world
    src_sec = get_axis(board_axes_aligned, board_secondary_vec)
    tgt_sec = global_secondary_vec / np.linalg.norm(global_secondary_vec)
    src_sec_proj = src_sec - np.dot(src_sec, tgt_main) * tgt_main
    tgt_sec_proj = tgt_sec - np.dot(tgt_sec, tgt_main) * tgt_main
    if np.linalg.norm(src_sec_proj) < 1e-8 or np.linalg.norm(tgt_sec_proj) < 1e-8:
        R_align_sec = np.eye(3)
    else:
        src_sec_proj /= np.linalg.norm(src_sec_proj)
        tgt_sec_proj /= np.linalg.norm(tgt_sec_proj)
        angle = np.arccos(np.clip(np.dot(src_sec_proj, tgt_sec_proj), -1, 1))
        axis = tgt_main
        if np.abs(angle) < 1e-6:
            R_align_sec = np.eye(3)
        else:
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R_align_sec = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            if np.dot(np.cross(src_sec_proj, tgt_sec_proj), axis) < 0:
                R_align_sec = R_align_sec.T
    R_global = R_align_sec @ R_align_main

    # 6. Apply rotation to all cameras using rotate_calibration_arbitrary
    params_dict_new = rotate_calibration_arbitrary(params_dict, R_global)

    # --- Apply z_offset translation to all outputs ---
    shift = 0.0

    # Compute board origin in global frame (after alignment)
    rvec_board, tvec_board, _ = poses[best_frame]
    R_board, _ = cv2.Rodrigues(rvec_board)
    t_board = tvec_board.reshape(3)
    # Transform to new global frame:
    t_board_global = R_global @ t_board
    board_z = t_board_global[2]
        
    if z_offset != 0.0:
        # Move camera group so that the board's Z in the best frame matches z_offset
        tvec = params_dict_new["tvec"]
        rvec = params_dict_new["rvec"]
        rmats = [cv2.Rodrigues(np.array(r[None, :]))[0] for r in rvec]

        # Compute board origin in global frame (after alignment)
        rvec_board, tvec_board, _ = poses[best_frame]
        R_board, _ = cv2.Rodrigues(rvec_board)
        t_board = tvec_board.reshape(3)
        # Transform to new global frame:
        t_board_global = R_global @ t_board
        board_z = t_board_global[2]

        # Compute how much to shift so that board_z == z_offset
        shift = z_offset - board_z

        # Apply shift to all cameras (move along global Z)
        for i, (R, t) in enumerate(zip(rmats, tvec)):
            C_world = -R.T @ t
            C_world_new = C_world + np.array([0, 0, shift])
            tvec[i] = -R @ C_world_new
        params_dict_new["tvec"] = tvec

 
    # Also return R_global for transforming points/poses
    print(f"Best frame: {best_frame}, fit error: {best_fit_error:.4f}" if best_fit_error is not None else f"Best frame: {best_frame}")
    print(f"Selected from {len(best_count_frames)} frames with {max_count} detections.")
    print(f"board_z: {board_z}, z_offset: {z_offset}, shift: {shift}")
    return params_dict_new, best_frame, best_fit_error, R_global, shift


def transform_and_shift_p3ds_dict(per_frame_p3ds, R_global=None, shift=None):
    """
    Apply a global rotation and/or shift to all 3D keypoints per frame.
    Handles empty arrays safely.
    """
    out = {}
    for frame_idx, arr in per_frame_p3ds.items():
        arr = np.asarray(arr)
        if arr.size == 0:
            out[frame_idx] = arr
            continue
        if arr.ndim == 1:
            arr = arr.reshape(-1, 3)
        if R_global is not None:
            arr = (R_global @ arr.T).T
        if shift is not None:
            # Ensure shift is a 3-element vector
            shift_vec = np.array(shift)
            if shift_vec.size == 1:
                shift_vec = np.array([0, 0, float(shift_vec)])
            arr = arr + shift_vec.reshape(1, 3)
        out[frame_idx] = arr
    return out

def transform_and_shift_board_poses(poses, R_global=None, shift=None):
    """
    Apply a global rotation and/or shift to all board poses.
    Handles empty tvecs safely.
    """
    import cv2
    out = {}
    for frame_idx, (rvec, tvec, ids) in poses.items():
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)
        if R_global is not None:
            R = R_global @ R
            t = R_global @ t
        if shift is not None:
            shift_vec = np.array(shift)
            if shift_vec.size == 1:
                shift_vec = np.array([0, 0, float(shift_vec)])
            t = t + shift_vec.reshape(3)
        rvec_new, _ = cv2.Rodrigues(R)
        out[frame_idx] = (rvec_new, t.reshape(3, 1), ids)
    return out

def optimize_calibration_tilt_to_flatten_board_z(params_dict, poses, verbose=True):
    """
    Finds a global rotation (pitch, roll) that minimizes the variance of board Z values,
    applies it to both the camera calibration and board poses.
    Returns: new_params_dict, new_poses, R_opt, optimization_result
    """
    import numpy as np
    from scipy.optimize import minimize

    # Collect all board origins
    tvecs = np.array([tvec.reshape(3) for _, tvec, _ in poses.values()])

    def rotmat_from_angles(angles):
        # angles: [pitch, roll] in radians
        pitch, roll = angles
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch),  np.cos(pitch)]
        ])
        Ry = np.array([
            [np.cos(roll), 0, np.sin(roll)],
            [0, 1, 0],
            [-np.sin(roll), 0, np.cos(roll)]
        ])
        return Ry @ Rx

    def z_var_loss(angles):
        R = rotmat_from_angles(angles)
        tvecs_rot = (R @ tvecs.T).T
        z_vals = tvecs_rot[:, 2]
        return np.var(z_vals)

    # Initial guess: no rotation
    x0 = [0.0, 0.0]
    res = minimize(z_var_loss, x0, method='BFGS')

    if verbose:
        print(f"Optimal angles (rad): {res.x}")
        print(f"Optimal angles (deg): {np.degrees(res.x)}")
        print(f"Minimized Z variance: {res.fun}")

    R_opt = rotmat_from_angles(res.x)

    # Apply to camera calibration
    new_params_dict = rotate_calibration_arbitrary(params_dict, R_opt)
    # Apply to board poses
    new_poses = transform_and_shift_board_poses(poses, R_opt)

    return new_params_dict, new_poses, R_opt, res

def shift_outputs_to_board_z(params_dict, per_frame_p3ds, poses, z_offset_set, use_median=False):
    """
    Shifts all outputs so that the mean (or median) board Z matches z_offset_set.
    Args:
        params_dict: dict, camera calibration (must have "tvec")
        per_frame_p3ds: dict, frame_idx -> (N,3) array of 3D points
        poses: dict, frame_idx -> (rvec, tvec, ids)
        z_offset_set: float, desired board Z value
        use_median: bool, if True use median instead of mean
    Returns:
        params_dict_shifted, per_frame_p3ds_shifted, poses_shifted, shift_vec
    """
    import numpy as np
    import cv2
    from copy import deepcopy
    from collections import OrderedDict

    # Compute mean or median Z of board after tilt
    board_zs = [float(tvec.reshape(3)[2]) for _, tvec, _ in poses.values()]
    if use_median:
        board_z = np.median(board_zs)
    else:
        board_z = np.mean(board_zs)
    shift_z = z_offset_set - board_z
    shift_vec = np.array([0, 0, shift_z])

    # --- Correctly shift camera extrinsics ---
    params_dict_shifted = params_dict.copy()
    tvecs = params_dict["tvec"]
    rvecs = params_dict["rvec"]
    new_tvecs = []
    for r, t in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(r)
        C = -R.T @ t
        C_new = C + shift_vec
        t_new = -R @ C_new
        new_tvecs.append(t_new)
    params_dict_shifted["tvec"] = np.array(new_tvecs)

    # Shift per-frame 3D points
    per_frame_p3ds_shifted = deepcopy(per_frame_p3ds)
    for k, arr in per_frame_p3ds_shifted.items():
        if arr.size > 0:
            per_frame_p3ds_shifted[k] = arr + shift_vec.reshape(1, 3)

    # Shift board poses
    poses_shifted = OrderedDict()
    for k, (rvec, tvec, ids) in poses.items():
        tvec_new = tvec + shift_vec.reshape(3, 1)
        poses_shifted[k] = (rvec, tvec_new, ids)

    return params_dict_shifted, per_frame_p3ds_shifted, poses_shifted, shift_vec

def shift_outputs_to_frame_z(params_dict, per_frame_p3ds, poses, best_frame, z_offset_set=0.0):
    """
    Shifts all outputs so that the board Z at best_frame matches z_offset_set.
    Args:
        params_dict: dict, camera calibration (must have "tvec")
        per_frame_p3ds: dict, frame_idx -> (N,3) array of 3D points
        poses: dict, frame_idx -> (rvec, tvec, ids)
        best_frame: int, frame index to use for reference
        z_offset_set: float, desired board Z value at best_frame
    Returns:
        params_dict_shifted, per_frame_p3ds_shifted, poses_shifted, shift_vec
    """
    import numpy as np
    import cv2
    from copy import deepcopy
    from collections import OrderedDict

    # Get board Z at best_frame
    rvec, tvec, ids = poses[best_frame]
    board_z = float(tvec.reshape(3)[2])
    shift_z = z_offset_set - board_z
    shift_vec = np.array([0, 0, shift_z])

    # --- Correctly shift camera extrinsics ---
    params_dict_shifted = params_dict.copy()
    tvecs = params_dict["tvec"]
    rvecs = params_dict["rvec"]
    new_tvecs = []
    for r, t in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(r)
        C = -R.T @ t
        C_new = C + shift_vec
        t_new = -R @ C_new
        new_tvecs.append(t_new)
    params_dict_shifted["tvec"] = np.array(new_tvecs)

    # Shift per-frame 3D points
    per_frame_p3ds_shifted = deepcopy(per_frame_p3ds)
    for k, arr in per_frame_p3ds_shifted.items():
        if arr.size > 0:
            per_frame_p3ds_shifted[k] = arr + shift_vec.reshape(1, 3)

    # Shift board poses
    poses_shifted = OrderedDict()
    for k, (rvec, tvec, ids) in poses.items():
        tvec_new = tvec + shift_vec.reshape(3, 1)
        poses_shifted[k] = (rvec, tvec_new, ids)

    return params_dict_shifted, per_frame_p3ds_shifted, poses_shifted, shift_vec


def get_num_frames_from_videos(input_folder, vid_base, cam_names):
    """
    Returns the number of frames in the calibration videos.
    Assumes all videos have the same number of frames.
    """
    for cam in cam_names:
        # Try to find the video file for this camera
        # Example: calibration_20250510_100309.0.mp4
        pattern = f"{vid_base}.{cam}.mp4"
        video_path = os.path.join(input_folder, pattern)
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return num_frames
    raise FileNotFoundError("No video files found for any camera.")

def recreate_cgroup_from_entry(entry):
    params = entry["camera_calibration"]
    cam_names = entry["camera_names"]
    cameras = []
    for i, name in enumerate(cam_names):
        # Build camera matrix
        mtx = np.array([
            [params["mtx"][i,0]*1000, 0, params["mtx"][i,2]*1000],
            [0, params["mtx"][i,1]*1000, params["mtx"][i,3]*1000],
            [0, 0, 1]
        ])
        cam = Camera(
            name=name,
            matrix=mtx,
            dist=params["dist"][i],
            rvec=params["rvec"][i],      
            tvec=params["tvec"][i]*1000  
        )
        cameras.append(cam)
    return CameraGroup(cameras)

def triangulate_from_calibration_points(cal_points, cgroup, min_cameras=2):
    num_cams, num_frames, num_corners, _ = cal_points.shape
    per_frame_p3ds = {}
    per_frame_ids = {}
    for frame_idx in range(num_frames):
        id_to_2d = {}
        id_to_cams = {}
        for cam_idx in range(num_cams):
            for corner_idx in range(num_corners):
                pt = cal_points[cam_idx, frame_idx, corner_idx]
                if not np.any(np.isnan(pt)):
                    if corner_idx not in id_to_2d:
                        id_to_2d[corner_idx] = []
                        id_to_cams[corner_idx] = []
                    id_to_2d[corner_idx].append(pt)
                    id_to_cams[corner_idx].append(cam_idx)
        p3ds = []
        ids = []
        for id_, points2d in id_to_2d.items():
            if len(points2d) >= min_cameras:
                pts2d_full = np.full((num_cams, 1, 2), np.nan, dtype=np.float32)
                for cam_idx, pt in zip(id_to_cams[id_], points2d):
                    pts2d_full[cam_idx, 0, :] = pt
                pts3d = cgroup.triangulate(pts2d_full)
                p3ds.append(pts3d[0] / 1000.0)  # convert to meters
                ids.append(id_)
        per_frame_p3ds[frame_idx] = np.array(p3ds)
        per_frame_ids[frame_idx] = np.array(ids)
    return per_frame_p3ds, per_frame_ids