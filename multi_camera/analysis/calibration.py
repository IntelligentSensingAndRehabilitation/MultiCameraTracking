import cv2
import numpy as np
import jax
import jaxopt
from jax import vmap, jit
from jax import numpy as jnp
from tqdm import trange
from functools import partial

from multi_camera.analysis.camera import triangulate_point, reprojection_error, reconstruction_error, get_checkboard_3d


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

    def calibrate_camera(self, flags=None):
        N = len(self.frames)
        objpoints = [self.objp] * N
        imgpoints = self.corners

        h, w, _ = self.shape

        if flags is None:
            flags = cv2.CALIB_ZERO_DISPARITY | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2| cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_ZERO_TANGENT_DIST

        _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (h, w), None, np.zeros((5,)), flags=flags)
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
        from multiprocessing.dummy import Pool as ThreadPool

        pool = ThreadPool(num_views)

        def process_video(params):
            cap, parser, idx = params
            if idx == 0:
                progress_fn = trange
            else:
                progress_fn = range

            for i in progress_fn(0, frames, skip):
                if skip != 1:
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


def extract_origin(camera_params, checkerboard_points):
    cal3d = triangulate_point(camera_params, checkerboard_points[:, 0])

    x0 = cal3d[0]
    x = cal3d[5] - x0
    z = cal3d[18] - x0

    x = x / np.linalg.norm(x)
    z = -z / np.linalg.norm(z)
    y = -np.cross(x, z)

    board_rotation = np.stack([x, y, z])
    return x0, board_rotation


def shift_calibration(camera_params, offset, rotation=np.eye(3), zoffset=None):
    from jaxlie import SO3

    camera_params = camera_params.copy()
    offset = offset / 1000.0

    camera_rotations = vmap(lambda x: SO3.exp(x).as_matrix())(camera_params['rvec'])
    tvec = camera_params['tvec'] + camera_rotations @ offset.reshape((3,))
    rvec = vmap(lambda x: (SO3.exp(x) @ SO3.from_matrix(rotation.T)).log())(camera_params['rvec'])

    camera_params['rvec'] = rvec
    camera_params['tvec'] = tvec

    if zoffset:
        camera_params = shift_calibration(camera_params, np.array([0, 0, -zoffset]))

    return camera_params


def initialize_group_calibration(parsers):
    """
    Use detected checkerboards to initialize calibration parameters

    Parameters:
        parsers (List[CheckerboardAccumulator]) : detection of checkerboards

    Returns:
        calibration dictionary - contains intrinsic and extrinsic parameters and distortion
        checkerboard_params - initial location and rotations of checkerboards
        checkerboard_points - matrix of 2D corners (cameras X points X 2)
    """

    from jax import jit, vmap
    from jax import numpy as jnp
    from jaxlie import SO3, SE3

    frames = np.sort(np.unique(np.concatenate([p.frames for p in parsers])))
    N = parsers[0].objp.shape[0]  # number of coordinates in pattern

    checkerboard_points = np.zeros((len(parsers), len(frames), N, 2)) * np.nan
    rvecs = np.zeros((len(parsers), len(frames), 3)) * np.nan
    tvecs = np.zeros((len(parsers), len(frames), 3)) * np.nan
    cals = [p.calibrate_camera(flags=0) for p in parsers]

    # convert format from the parser to a matrix of observations with nan for missing
    for i, p in enumerate(parsers):
        for j, f in enumerate(frames):
            idx = np.where(p.frames == f)[0]
            if len(idx) == 1:
                checkerboard_points[i,j, :, :] = p.corners[idx[0]][:, 0, :]
                rvecs[i,j] = cals[i]['rvecs'][idx[0]][:,0]
                tvecs[i,j] = cals[i]['tvecs'][idx[0]][:,0]

    N, frames, _, _ = checkerboard_points.shape

    camera_rvecs = np.ones((N, 3)) * np.nan
    camera_tvecs = np.ones((N, 3)) * np.nan

    checkerboard_rvecs = np.empty((frames, 3))
    checkerboard_tvecs = np.empty((frames, 3))

    def make_M(rvec, tvec):
        return SE3.from_rotation_and_translation(SO3.exp(jnp.array(rvec)), jnp.array(tvec))

    for i in range(frames):

        if i == 0:
            checkerboard_rvecs[i, :] = 0
            checkerboard_tvecs[i, :] = 0
        else:
            for j in range(N):
                if ~np.isnan(camera_rvecs[j,0]) and ~np.isnan(rvecs[j, i, 0]):
                    T = (make_M(rvecs[j,i], tvecs[j,i]).inverse() @ make_M(camera_rvecs[j], camera_tvecs[j])).inverse()

                    checkerboard_rvecs[i] = T.rotation().log()
                    checkerboard_tvecs[i] = T.translation()
                    break
            pass

        checkerboardT = make_M(checkerboard_rvecs[i, :], checkerboard_tvecs[i, :])

        for j in range(N):
            if np.isnan(camera_rvecs[j,0]) and ~np.isnan(rvecs[j, i, 0]):
                camT = make_M(rvecs[j, i], tvecs[j, i])
                T = camT @ checkerboardT.inverse()

                camera_rvecs[j, :] = T.rotation().log()
                camera_tvecs[j, :] = T.translation()

    # Initialize params from my code
    camera_params = {'mtx': np.array([[c['mtx'][0,0], c['mtx'][1,1], c['mtx'][0,2], c['mtx'][1, 2]] for c in cals]) / 1000.0,
                     'dist': np.array([c['dist'].reshape((-1)) for c in cals]),
                     'rvec': camera_rvecs,
                     'tvec': camera_tvecs / 1000.0,
                    }

    checkerboard_params = {'rvecs': checkerboard_rvecs, 'tvecs': checkerboard_tvecs / 1000.0}

    return camera_params, checkerboard_params, checkerboard_points


def checkerboard_reprojection_loss(camera_params, checkerboard_params, checkerboard_points, objp):
    checkerboard_rvecs = checkerboard_params['rvecs']
    checkerboard_tvecs = checkerboard_params['tvecs']

    estimated_3d_points = vmap(lambda a, b: get_checkboard_3d(a, b, objp))(checkerboard_rvecs, checkerboard_tvecs)
    err = reprojection_error(camera_params, checkerboard_points, estimated_3d_points)

    norm=False
    if norm:
        #err = jnp.linalg.norm(err, axis=-1) ** 2
        return jnp.nanmean(err ** 2)
    else:
        return jnp.nanmean(jnp.abs(err))


def checkerboard_reconstruction_loss(camera_params, checkerboard_params, checkerboard_points, objp):
    checkerboard_rvecs = checkerboard_params['rvecs']
    checkerboard_tvecs = checkerboard_params['tvecs']

    estimated_3d_points = vmap(lambda a, b: get_checkboard_3d(a, b, objp))(checkerboard_rvecs, checkerboard_tvecs)
    err = reconstruction_error(camera_params, checkerboard_points, estimated_3d_points, stop_grad=True)

    norm = False
    if norm:
        #err = jnp.linalg.norm(err, axis=-1) ** 2
        return jnp.nanmean(err ** 2)
    else:
        return jnp.nanmean(jnp.abs(err))

@jit
def checkerboard_loss(checkerboard_params, camera_params, checkerboard_points, objp):
    #return checkerboard_reconstruction_loss(camera_params, checkerboard_params, checkerboard_points, objp)
    return checkerboard_reprojection_loss(camera_params, checkerboard_params, checkerboard_points, objp)

@jit
def camera_loss(camera_params, checkerboard_params, checkerboard_points, objp):
    #return checkerboard_reconstruction_loss(camera_params, checkerboard_params, checkerboard_points, False)
    return checkerboard_reprojection_loss(camera_params, checkerboard_params, checkerboard_points, objp)

@jit
def cycle_loss(camera_params, checkerboard_points):
    est_checkerboard_3d = triangulate_point(camera_params, checkerboard_points)
    # backpropgating through SVD takes a huge amount of time
    est_checkerboard_3d = jax.lax.stop_gradient(est_checkerboard_3d)
    err = reprojection_error(camera_params, checkerboard_points, est_checkerboard_3d)
    return jnp.nanmean(jnp.abs(err))

@jit
def update_checkerboard(checkerboard_params, camera_params, checkerboard_points, objp, iterations=10):
    checkerboard_solver = jaxopt.GradientDescent(fun=checkerboard_loss, maxiter=iterations, verbose=False)
    return checkerboard_solver.run(checkerboard_params, camera_params=camera_params, checkerboard_points=checkerboard_points, objp=objp)[0]

@jit
def update_camera(checkerboard_params, camera_params, checkerboard_points, objp, iterations=10):
    camera_solver = jaxopt.GradientDescent(fun=camera_loss, maxiter=iterations, verbose=False)
    return camera_solver.run(camera_params, checkerboard_params=checkerboard_params, checkerboard_points=checkerboard_points, objp=objp)[0]

#@jit
#def update_camera_checkerboard
@jit
def update_camera_cycle(camera_params, checkerboard_points, iterations=10, stepsize=0.0):
    cycle_solver = jaxopt.GradientDescent(fun=cycle_loss, maxiter=iterations, verbose=False, stepsize=stepsize)
    return cycle_solver.run(camera_params, checkerboard_points=checkerboard_points)[0]


def refine_calibration(camera_params, checkerboard_params, checkerboard_points, objp, iterations=20,
                       inner_iterations=10, verbose=True, cycle_consistency=True):

    for _ in range(iterations):

        checkerboard_params = update_checkerboard(checkerboard_params, camera_params, checkerboard_points,
                                                  objp, iterations=inner_iterations)
        camera_params = update_camera(checkerboard_params, camera_params, checkerboard_points,
                                      objp, iterations=inner_iterations)

        if cycle_consistency:
            camera_params = update_camera_cycle(camera_params, checkerboard_points, iterations=inner_iterations)

        if verbose:
            e1 = checkerboard_reprojection_loss(camera_params, checkerboard_params, checkerboard_points, objp)
            e2 = checkerboard_reconstruction_loss(camera_params, checkerboard_params, checkerboard_points, objp)
            e3 = cycle_loss(camera_params, checkerboard_points)
            print(f'Reprojection error {e1:.2f} pixels. Modeled reconstruction error {e2:.2f} mm. Cycle reprojection error {e3:0.2f} pixels.')

    return camera_params, checkerboard_params


def run_calibration(vid_base, vid_path=".", return_parsers=False, frame_skip=5, **kwargs):
    """
    Run the calibration routine on a video recording session

    Designed to be used on data recorded with acquisition.record and assumes the
    files have a calibration_ prefix.

        Parameters:
            vid_base (str) : filter to match (e.g. calibration_20220802_110011)
            vid_path (str, optional) : bath to files, otherwise assumes PWD
            return_parsers (boolean, option) : set true to get back checkboard coordinates
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
    parsers = get_checkerboards(vids, max_frames=5000, skip=frame_skip, save_images=True,
                                downsample=2, multithread=True, checkerboard_size=110.0)
    objp = parsers[0].objp

    print("Now running calibration")
    init_camera_params, init_checkerboard_params, checkerboard_points = initialize_group_calibration(parsers)
    camera_params, checkerboard_params = refine_calibration(init_camera_params, init_checkerboard_params, checkerboard_points,
                                                            objp, inner_iterations=5, iterations=150, cycle_consistency=True)
    error = float(checkerboard_loss(checkerboard_params, camera_params, checkerboard_points, objp))

    print("Zeroing coordinates")
    x0, board_rotation = extract_origin(camera_params, checkerboard_points[:, 5:])
    camera_params_zeroed = shift_calibration(camera_params, x0, board_rotation, zoffset=1245)

    # prepare for saving to database
    params_dict = jax.tree_map(np.array, camera_params_zeroed)
    params_dict['camera_names'] = cam_names

    timestamp = vid_base.split("calibration_")[1]
    timestamp = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")

    entry = {
        "cal_timestamp": timestamp,
        "camera_config_hash": camera_hash,
        "num_cameras": len(cam_names),
        "camera_names": cam_names,
        "camera_calibration": params_dict,
        "reprojection_error": error,
    }

    if return_parsers:
        return entry, parsers

    return entry
