import cv2
import numpy as np
from tqdm import trange


def hash_names(x):
    ''' Hash a list of camera names '''
    import hashlib
    x = list(np.sort(x))
    x = ', '.join(x)
    return hashlib.sha256(x.encode('utf-8')).hexdigest()[:10]


class CheckerboardAccumulator:
    """
    Helper class to detect and store the checkerboards in a
    video.
    """

    def __init__(self, checkerboard_size=110.0, cherboard_dim=(4, 6), downsample=1, save_images=False):
        self.rows, self.cols = cherboard_dim

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((self.rows * self.cols, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2) * checkerboard_size  # cm

        self.frames = []
        self.corners = []
        self.images = []
        self.last_image = None

        self.shape = None

        self.save_images = save_images
        self.downsample = downsample

    def process_frame(self, idx, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        # chessboard_flags = cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_LARGER + chessboard_flags

        gray_ds = cv2.resize(gray, (img.shape[1] // self.downsample, img.shape[0] // self.downsample))
        ret, corners = cv2.findChessboardCorners(gray_ds, (self.cols, self.rows), chessboard_flags)

        if not self.shape:
            self.shape = img.shape

        if ret:
            corners = corners * self.downsample

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            self.frames.append(idx)
            self.corners.append(corners2)

            if self.save_images:
                self.images.append(img)

        if self.save_images:
            self.last_image = img

        return ret

    def calibrate_camera(self):
        N = len(self.frames)
        objpoints = [self.objp] * N
        imgpoints = self.corners

        h, w, _ = self.shape

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (h, w), cv2.CALIB_RATIONAL_MODEL, None)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        return {"mtx": mtx, "dist": dist, "rvecs": rvecs, "tvecs": tvecs, "newcameramtx": newcameramtx, "roi": roi}

    def get_points(self, idx):
        return [self.objp] * len(idx), list(np.array(self.corners)[idx])


def get_checkerboards(filenames, max_frames=None, skip=1, multithread=False, **kwargs):
    """
    Detect checkboards in a list of videos.

        Parameters:
            filenames (list) : list of pths to videos
            max_frames (int, optional) : maximum number of frames to parse
            skip (int, optional) : skip between frames to detect
            multithread (boolean, optional): where to have a per-video frame
            kwargs : optional parameters passed to the accumulator

        Returns:
            list of CheckboardAccumulators for each video
    """

    num_views = len(filenames)

    caps = [cv2.VideoCapture(f) for f in filenames]
    parsers = [CheckerboardAccumulator(**kwargs) for _ in range(num_views)]

    frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    frames = min(max_frames or frames, frames)

    if multithread == False:
        for i in trange(0, frames, skip):

            for c, p in zip(caps, parsers):

                c.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, img = c.read()

                if not ret or img is None:
                    break

                p.process_frame(i, img)

    else:
        import multiprocessing
        from multiprocessing.dummy import Pool as ThreadPool

        pool = ThreadPool(num_views)

        def process_video(params):
            cap, parser, idx = params
            if idx == 0:
                progress_fn = trange
            else:
                progress_fn = range

            for i in progress_fn(0, frames, skip):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, img = cap.read()
                if not ret or img is None:
                    break

                parser.process_frame(i, img)

            return parser

        parsers = pool.map(process_video, zip(caps, parsers, range(num_views)))

    for c in caps:
        c.release()

    return parsers


def calibrate_pair(
    p1, p2, stereocalib_flags=cv2.CALIB_USE_INTRINSIC_GUESS, rectify_scale=1.0, rectify_flags=cv2.CALIB_ZERO_DISPARITY
):
    """
    Perform stereo camera calibration with OpenCV on a pair of cameras

        Parameters:
            p1 (Accumulator) : the accumulator for first camera
            p2 (Accumulator) : the accumulator for second camera

        Returns:
            dictionary of calibration parameters
    """

    _, idx0, idx1 = np.intersect1d(p1.frames, p2.frames, return_indices=True)

    assert len(idx0) > 0, "No overlapping frames"

    objpoints, im1points = p1.get_points(idx0)
    _, im2points = p2.get_points(idx1)

    h, w, _ = p1.shape

    calibration1 = p1.calibrate_camera()
    calibration2 = p2.calibrate_camera()

    stereocalib_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6)

    res = cv2.stereoCalibrate(
        objpoints,
        im1points,
        im2points,
        calibration1["mtx"].copy(),
        calibration1["dist"].copy(),
        calibration2["mtx"].copy(),
        calibration2["dist"].copy(),
        (h, w),
        criteria=stereocalib_criteria,
        flags=stereocalib_flags,
    )
    stereocalib_retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = res

    cal = {
        "cameraMatrix1": cameraMatrix1,
        "distCoeffs1": distCoeffs1,
        "cameraMatrix2": cameraMatrix2,
        "distCoeffs2": distCoeffs2,
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "N": len(idx0),
    }

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cal["cameraMatrix1"],
        cal["distCoeffs1"],
        cal["cameraMatrix2"],
        cal["distCoeffs2"],
        (w, h),
        R=cal["R"],
        T=cal["T"],
        alpha=rectify_scale,
        newImageSize=(w, h),
        flags=rectify_flags,
    )

    cal["R1"] = R1
    cal["R2"] = R2
    cal["P1"] = P1
    cal["P2"] = P2
    cal["Q"] = Q
    cal["roi1"] = roi1
    cal["roi2"] = roi2

    return cal


def extract_origin(cgroup, imgp):
    cal3d = cgroup.triangulate(imgp[:, :24])

    x0 = cal3d[0]
    x = cal3d[5] - x0
    z = cal3d[18] - x0

    x = x / np.linalg.norm(x)
    z = -z / np.linalg.norm(z)
    y = -np.cross(x, z)

    board_rotation = np.stack([x, y, z])
    return x0, board_rotation


def shift_calibration(cgroup, offset, rotation=np.eye(3), zoffset=None):
    cgroup = cgroup.copy()

    for cam in cgroup.cameras:
        camera_offset = cv2.Rodrigues(cam.rvec)[0] @ offset
        cam.tvec = cam.tvec + camera_offset
        cam.rvec = cv2.Rodrigues(cv2.Rodrigues(cam.rvec)[0] @ rotation.transpose())[0]

    if zoffset:
        cgroup = shift_calibration(cgroup, np.array([0, 0, zoffset]))

    return cgroup


def calibrate_bundle(parsers, camera_names=None, fisheye=False, verbose=True, zero_origin=False, **kwargs):
    """
    Calibrate multiple cameras using bundle adjustment

    This uses the bundle adjustment implemented in aniposelib to perform
    the calibration.

        Parameters:
            parsers (list) : list of checkerboard video parser results
            camera_names (list, optional) : list of camera names
            fisheye (boolean, optional) : set true to enable a fisheye camear
            versbose (boolean, optional) : set true to produce more outputs
            **kwargs : option arguments that can be passed into aniposelib.bundle_adjust_iter
                (e.g. n_samp_full=200, n_samp_iter=100)

        Returns:
            reprojection error
            dictionary of configurations
    """

    from aniposelib.cameras import CameraGroup
    from aniposelib.boards import Checkerboard
    from aniposelib.utils import get_initial_extrinsics, make_M, get_rtvec, get_connections
    from aniposelib.boards import merge_rows, extract_points, extract_rtvecs

    if camera_names is None:
        camera_names = list(range(len(parsers)))

    p = parsers[0]
    square_length = p.objp[1, 0] - p.objp[0, 0]
    cols = p.cols
    rows = p.rows

    cgroup = CameraGroup.from_names(camera_names, fisheye=fisheye)
    board = Checkerboard(cols, rows, square_length=square_length)

    for cam, p in zip(cgroup.cameras, parsers):
        h, w, _ = p.last_image.shape
        cam.set_size((w, h))

    # convert parser results to the "rows" used by aniposelib
    all_rows = []
    for p in parsers:
        rows = []
        for frame_num, corner in zip(p.frames, p.corners):
            row = {"framenum": frame_num, "corners": corner, "ids": None}
            row['filled'] = board.fill_points(row['corners'], row['ids'])
            rows.append(row)
        all_rows.append(rows)

    for rows, camera, parser in zip(all_rows, cgroup.cameras, parsers):
        size = camera.get_size()

        assert size is not None, \
            "Camera with name {} has no specified frame size".format(camera.get_name())

        if fisheye:
            objp, imgp = board.get_all_calibration_points(rows)
            mixed = [(o, i) for (o, i) in zip(objp, imgp) if len(o) >= 7]
            objp, imgp = zip(*mixed)
            matrix = cv2.initCameraMatrix2D(objp, imgp, tuple(size))
            camera.set_camera_matrix(matrix)
        else:
            cal = parser.calibrate_camera()
            camera.set_camera_matrix(cal['mtx'])
            # camera.set_distortions(cal['dist'][0])  # system only expects one distortion parameters

    for i, (row, cam) in enumerate(zip(all_rows, cgroup.cameras)):
        all_rows[i] = board.estimate_pose_rows(cam, row)

    merged = merge_rows(all_rows)
    imgp, extra = extract_points(merged, board, min_cameras=2)

    rtvecs = extract_rtvecs(merged)
    rvecs, tvecs = get_initial_extrinsics(rtvecs, cgroup.get_names())
    cgroup.set_rotations(rvecs)
    cgroup.set_translations(tvecs)

    if verbose:
        print(cgroup.get_dicts())

    error = cgroup.bundle_adjust_iter(imgp, extra, verbose=verbose, **kwargs)

    if zero_origin:
        x0, board_rotation = extract_origin(cgroup, imgp)
        cgroup = shift_calibration(cgroup, x0, board_rotation, 80)

    return error, cgroup.get_dicts()


def run_calibration(vid_base, vid_path="."):
    """
    Run the calibration routine on a video recording session

    Designed to be used on data recorded with acquisition.record and assumes the
    files have a calibration_ prefix.

        Parameters:
            vid_base (str): filter to match (e.g. calibration_20220802_110011)
            vid_path (str, optional): bath to files, otherwise assumes PWD

        Returns:
            calibration dictionary
    """

    import os
    import numpy as np
    from datetime import datetime

    # search for files. expects them to be in the format vid_base.serial_number.mp4
    vids = []
    for v in os.listdir(vid_path):
        base, ext = os.path.splitext(v)
        if ext == ".mp4" and len(base.split('.')) == 2 and base.split(".")[0] == vid_base:
            vids.append(os.path.join(vid_path, v))
    vids.sort()

    cam_names = [os.path.split(v)[1].split(".")[1] for v in vids]
    camera_hash = hash_names(cam_names)
    print(f'Cam names: {cam_names} camera hash: {camera_hash}')

    print(f"Found {len(vids)} videos. Now detecting checkerboards.")
    parsers = get_checkerboards(vids, max_frames=5000, skip=10, save_images=True,
                                downsample=2, multithread=True, checkerboard_size=11.0)

    print("Now running calibration")
    error, camera_params = calibrate_bundle(parsers, cam_names, verbose=True, zero_origin=True)
    timestamp = vid_base.split("calibration_")[1]
    timestamp = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")

    entry = {
        "cal_timestamp": timestamp,
        "camera_config_hash": camera_hash,
        "num_cameras": len(cam_names),
        "camera_names": cam_names,
        "camera_calibration": camera_params,
        "reprojection_error": error,
    }

    return entry
