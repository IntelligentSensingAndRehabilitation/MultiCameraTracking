import numpy as np
import datajoint as dj
from pose_pipeline.dj_schema import TimestampedSchema

schema = TimestampedSchema("multicamera_tracking")


# keeping this class definition in this file to avoid it needing to depend
# on the pose pipeline, which is required for the rest of the class definitions


@schema
class Calibration(dj.Manual):
    definition = """
    # Calibration of multiple camera system
    cal_timestamp        : timestamp
    camera_config_hash   : varchar(10)
    ---
    recording_base       : varchar(50)
    num_cameras          : int
    camera_names         : longblob   # list of camera names
    camera_calibration   : longblob   # calibration results
    reprojection_error   : float
    calibration_points   : longblob
    calibration_shape    : longblob
    calibration_type=""  : varchar(50)
    """


if __name__ == "__main__":
    import argparse
    import os

    import datajoint as dj

    from ..analysis.calibration import insert_calibration_videos, run_calibration_APL

    parser = argparse.ArgumentParser(description="Run camera calibration and releveling pipeline.")
    parser.add_argument('--vid_base', type=str, required=True, help='Base path to calibration video set (without .cam.mp4)')
    parser.add_argument('--trial_video_project', type=str, default=None,
                        help='Trial-side video_project for this session, e.g. CLINIC_GAIT. Only '
                             'needed when the session has not been pushed to DataJoint yet; the '
                             'calibration videos are filed under "{project}_CALIBRATION".')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment for the MultiCameraCalibration row. ArUco detection only '
                             'runs on calibrations whose comment contains "aruco".')
    parser.add_argument('--charuco', type=str, default='charuco', help='Use charuco or checkerboard')
    parser.add_argument('--checkerboard_size', type=int, default=109, help='Checkerboard square size in mm')
    parser.add_argument('--checkerboard_dim', type=int, nargs=2, default=[5,7], help='Checkerboard dimensions (rows cols)')
    parser.add_argument('--marker_bits', type=int, default=6, help='Charuco marker bits')
    parser.add_argument('--releveling_type', type=str, default='no_leveling', help='Releveling method')
    parser.add_argument('--z_offset_set', type=float, default=1.534, help='Board height for stroller/custom releveling (meters)')
    parser.add_argument('--min_height', type=float, default=2.0, help='Min camera height for z_up (meters)')
    parser.add_argument('--min_cameras', type=int, default=2, help='Min cameras for fitting board pose')
    parser.add_argument('--board_ground_time', type=int, nargs=2, default=[0,200], help='Frame range for ground board')
    parser.add_argument('--board_axis', type=float, nargs=3, default=[0,-1,0], help='Board axis to align with global axis')
    parser.add_argument('--global_axis', type=float, nargs=3, default=[0,0,1], help='Global axis to align with board axis')
    parser.add_argument('--board_secondary', type=float, nargs=3, default=[1,0,0], help='Board secondary axis')
    parser.add_argument('--global_secondary', type=float, nargs=3, default=[1,0,0], help='Global secondary axis')
    parser.add_argument('--output', type=str, default=None, help='Optional: Output file to save calibration results')

    args = parser.parse_args()

    # Call your calibration pipeline
    entry, board = run_calibration_APL(
        vid_base=args.vid_base,
        charuco=args.charuco,
        checkerboard_size=args.checkerboard_size,
        checkerboard_dim=tuple(args.checkerboard_dim),
        marker_bits=args.marker_bits,
        releveling_type=args.releveling_type,
        board=None,
        z_offset_set=args.z_offset_set,
        min_height=args.min_height,
        min_cameras=args.min_cameras,
        board_ground_time=tuple(args.board_ground_time),
        board_axis=tuple(args.board_axis),
        global_axis=tuple(args.global_axis),
        board_secondary=tuple(args.board_secondary),
        global_secondary=tuple(args.global_secondary)
    )

    if np.isnan(entry["reprojection_error"]):
        raise Exception(f"Calibration failed: {entry}")

    if entry["reprojection_error"] > 0.3:
        print(entry)
        print(f'The error was {entry["reprojection_error"]}. Are you sure you would like to store this in the database? [Yes/No]')

        response = input()
        if response[0].upper() != "Y":
            print("Cancelling")

    vid_path, vid_base = os.path.split(args.vid_base)
    entry["recording_base"] = vid_base # we default to using using input vid_path with path+base then we split here

    if args.charuco == "charuco": # is "charuco" string the right type?
        entry["calibration_type"] = "charuco"


    print('Does the calibration visualization on port 8050 look correct? Are you sure you would like to store this in the database? [Yes/No]')
    response = input()
    if response[0].upper() != "Y":
        print("Cancelling")
    else:
        print("Storing calibration in database...")
        # Register the videos alongside the calibration. Without them the
        # calibration never enters CalibrationArucoDetection.key_source, so its
        # trials can never reach TenMeterWalkTest.
        with dj.conn().transaction:
            Calibration.insert1(entry)
            insert_calibration_videos(
                {k: entry[k] for k in ("cal_timestamp", "camera_config_hash")},
                args.vid_base,
                camera_names=entry["camera_names"],
                trial_video_project=args.trial_video_project,
                comment=args.comment,
            )
        print("Calibration complete.")
