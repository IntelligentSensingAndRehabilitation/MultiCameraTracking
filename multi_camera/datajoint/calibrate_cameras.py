
def run_calibration(vid_base, vid_path='.'):

    from .multi_camera_dj import Calibration
    from ..analysis.calibration import run_calibration

    entry = run_calibration(vid_base, vid_path)

    if entry['reprojection_error'] > 0.3:
        print(f'The error was {entry["reprojection_error"]}. '
               'Are you sure you would like to store this in the database? [Yes/No]')

        response = input()
        if response[0].upper() != 'Y':
            print('Cancelling')
            return

    Calibration.insert1(entry)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Compute calibration from specified videos and insert into database')
    parser.add_argument('vid_base', help='Base filenames to use for calibration')
    parser.add_argument('vid_path', help='Path to files', default='.')
    args = parser.parse_args()

    run_calibration(vid_base=args.vid_base, vid_path=args.vid_path)
