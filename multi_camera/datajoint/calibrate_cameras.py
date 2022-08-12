import numpy as np
import datajoint as dj

schema = dj.schema("multicamera_tracking")


# keeping this class definition in this file to avoid it needing to depend
# on the pose pipeline, which is required for the rest of the class definitions


@schema
class Calibration(dj.Manual):
    definition = """
    # Calibration of multiple camera system
    cal_timestamp        : timestamp
    camera_config_hash   : varchar(10)
    ---
    num_cameras          : int
    camera_names         : longblob   # list of camera names
    camera_calibration   : longblob   # calibration results
    reprojection_error   : float
    """


def run_calibration(vid_base, vid_path="."):

    from ..analysis.calibration import run_calibration

    entry = run_calibration(vid_base, vid_path)

    if np.isnan(entry['reproject_error']):
        raise Exception(f'Calibration failed: {entry}')

    if entry["reprojection_error"] > 0.3:
        print(
            f'The error was {entry["reprojection_error"]}. '
            "Are you sure you would like to store this in the database? [Yes/No]"
        )

        response = input()
        if response[0].upper() != "Y":
            print("Cancelling")
            return

    Calibration.insert1(entry)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Compute calibration from specified videos and insert into database")
    parser.add_argument("vid_base", help="Base filenames to use for calibration")
    parser.add_argument("--vid_path", help="Path to files", default=".")
    args = parser.parse_args()

    run_calibration(vid_base=args.vid_base, vid_path=args.vid_path)
