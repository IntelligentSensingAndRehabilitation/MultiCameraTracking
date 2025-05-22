import cv2
import numpy as np
from typing import Callable, List, Tuple, Dict, Optional, Union
import jax
import jaxopt
from jax import vmap, jit
from jax import numpy as jnp
from tqdm import trange
from functools import partial

from multi_camera.analysis.camera import (
    robust_triangulate_points,
    triangulate_point,
    reprojection_error,
    reconstruction_error,
    get_checkboard_3d,
)

class ChArucoAccumulator:
    """
    Helper class to detect and store the checkerboards in a
    video.
    """

    def __init__(
        self,
        checkerboard_size=109.0,
        checkerboard_dim=(5, 7),
        downsample=1,
        save_images=False,
    ):
        self.rows, self.cols = checkerboard_dim
        self.square_size = checkerboard_size
        self.marker_size = checkerboard_size*0.8
        self.marker_bits = 6
        self.ARUCO_DICT = cv2.aruco.DICT_6X6_250
        self.DICT = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT)

        self.board = cv2.aruco.CharucoBoard((self.cols, self.rows), self.square_size, self.marker_size, self.DICT)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros(((self.cols-1) * (self.rows-1), 3), np.float32)
        self.objp[:, :2] = np.mgrid[0 : self.cols-1, 0 : self.rows-1].T.reshape(-1, 2) * checkerboard_size  # cm

        self.frames = []
        self.calibrated_frames = []
        self.corners = []
        self.images = []
        self.ids=[]
        self.last_image = None

        self.shape = None

        self.save_images = save_images
        self.downsample = downsample

    def recreate(self, checkerboard_detections, checkerboard_shape=None, height=1536, width=2048):
        if checkerboard_shape is not None:
            self.objp = checkerboard_shape

        self.shape = (height, width, 3)

        # now iterate through checkerboard detections and keep frames and corners not NaN
        self.frames = []
        self.corners = []
        self.ids=[]

        for i in range(len(checkerboard_detections)):
            if np.all(~np.isnan(checkerboard_detections[i])):
                self.frames.append(i)
                self.corners.append(checkerboard_detections[i, :, None].astype(np.float32))

        keep = self.filter_corners()
        print("Keeping {} frames".format(np.mean(keep)))
        self.frames = list(np.array(self.frames)[keep])
        self.corners = list(np.array(self.corners)[keep])
        self.ids = list(np.array(self.ids)[keep])
        return self

    def process_frame(self, idx, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_ds = cv2.resize(gray, (img.shape[1] // self.downsample, img.shape[0] // self.downsample))

        if not self.shape:
            self.shape = img.shape

        charuco_detector = cv2.aruco.CharucoDetector(self.board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray_ds)

        if charuco_corners is not None and charuco_ids is not None:
            if charuco_corners.shape[0] == (self.rows - 1) * (self.cols - 1):
                self.corners.append(charuco_corners * self.downsample)
                self.ids.append(charuco_ids)
                self.frames.append(idx)
            else:
                print(f"\rcorners2 {charuco_corners.shape}", end='')

        if self.save_images:
            self.images.append(img)
            self.last_image = img

        return len(charuco_corners) if charuco_corners is not None else 0

    def get_rvecs_tvecs(self, mtx, dist, corners=None):
        if corners is None:
            corners = self.corners

        rvecs = []
        tvecs = []
        for imgpoint in corners:
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(self.objp, imgpoint, mtx, dist, confidence=0.9, reprojectionError=30)
            rvecs.append(rvec)
            tvecs.append(tvec)

        return rvecs, tvecs

    def filter_corners(self, thresh=10.0, return_errors=False, update_self=False):
        N = len(self.frames)
        objpoints = [self.objp] * N
        imgpoints = self.corners

        h, w, _ = self.shape
        mtx = cv2.initCameraMatrix2D(objpoints, imgpoints, (w, h))
        dist = np.zeros((5,))

        corners = self.corners
        rvecs, tvecs = self.get_rvecs_tvecs(mtx, dist, self.corners)

        print(len(rvecs), len(tvecs), len(corners))
        errors = []
        for i in range(len(rvecs)):
            imgpoints2, _ = cv2.projectPoints(self.objp, rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(corners[i], imgpoints2, cv2.NORM_L2)

            errors.append(error)

        if update_self:
            keep = np.array(errors) < thresh
            print("Keeping {} frames".format(np.mean(keep)))
            self.frames = list(np.array(self.frames)[keep])
            self.corners = list(np.array(self.corners)[keep])

        if return_errors:
            return np.array(errors)
        else:
            return np.array(errors) < thresh

    def calibrate_camera(self, flags=None, max_frames=100, filter=False):
        frames = self.frames
        N = len(frames)
        objpoints = [self.objp] * N
        imgpoints = self.corners

        if filter:
            keep = self.filter_corners()
            objpoints = list(np.array(objpoints)[keep])
            imgpoints = list(np.array(imgpoints)[keep])
            frames = list(np.array(frames)[keep])
            N = len(frames)

        # if N > max_frames take an approximately evently space subset of both lists
        # to make the total max_frames. cv2 calibration starts taking excessively long
        # much beyond this
        if N > max_frames:
            idx = np.linspace(0, N - 1, max_frames).astype(int)
            # randomly sample a set of frames
            # idx = np.random.choice(N, max_frames, replace=False)
            objpoints = list(np.array(objpoints)[idx])
            imgpoints = list(np.array(imgpoints)[idx])
            self.calibrated_frames = [self.frames[i] for i in idx.tolist()]
        else:
            self.calibrated_frames = self.frames

        h, w, _ = self.shape

        if True:
            if flags is None:
                flags = (
                    cv2.CALIB_FIX_K2
                    | cv2.CALIB_FIX_K3
                    | cv2.CALIB_FIX_K4
                    | cv2.CALIB_FIX_K5
                    | cv2.CALIB_ZERO_TANGENT_DIST
                    | cv2.CALIB_FIX_PRINCIPAL_POINT
                    | cv2.CALIB_FIX_ASPECT_RATIO
                    | cv2.CALIB_RATIONAL_MODEL
                    | cv2.CALIB_USE_INTRINSIC_GUESS
                )
            initial_matrix = np.array([[1950, 0, w / 2], [0, 1950, h / 2], [0, 0, 1]])
            (ret, mtx, dist,
                rvecs, tvecs,
                stdDeviationsIntrinsics, stdDeviationsExtrinsics,
                perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=imgpoints,
                      charucoIds=self.ids,
                      board=self.board,
                      imageSize=(h, w),
                      cameraMatrix=initial_matrix,
                      distCoeffs=np.zeros((5,1)),
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

            # _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            #     objpoints,
            #     imgpoints,
            #     (w, h),
            #     initial_matrix,
            #     np.zeros((5,)),
            #     flags=flags,
            # )

            if dist[0] > 0:
                print("Warning: distortion is positive")
                flags = flags | cv2.CALIB_FIX_K1
                (ret, mtx, dist,
                rvecs, tvecs,
                stdDeviationsIntrinsics, stdDeviationsExtrinsics,
                perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=imgpoints,
                      charucoIds=self.ids,
                      board=self.board,
                      imageSize=self.shape,
                      cameraMatrix=initial_matrix,
                      distCoeffs=np.zeros((5,1)),
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))


            # print(mtx.astype(int))
            # print(dist)
            # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        else:
            mtx = cv2.initCameraMatrix2D(objpoints, imgpoints, (w, h))
            dist = np.zeros((5,))

            print(mtx.astype(int))

        # rvecs = []
        # tvecs = []
        # for imgpoint in self.corners:
        #     retval, rvec, tvec, inliers = cv2.solvePnPRansac(self.objp, imgpoint, mtx, dist, confidence=0.9, reprojectionError=30)
        #     rvecs.append(rvec)
        #     tvecs.append(tvec)

        self.calibrated_frames = self.frames
        return {
            "mtx": mtx,
            "dist": dist,
            "rvecs": rvecs,
            "tvecs": tvecs,
            "frames": self.calibrated_frames,
            "objpoints": objpoints,
            "imgpoints": imgpoints,
        }

    def get_points(self, idx):
        return [self.objp] * len(idx), list(np.array(self.corners)[idx])


class CheckerboardAccumulator:
    """
    Helper class to detect and store the checkerboards in a
    video.
    """

    def __init__(
        self,
        checkerboard_size=110.0,
        checkerboard_dim=(4, 6),
        downsample=1,
        save_images=False,
    ):
        self.rows, self.cols = checkerboard_dim

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((self.rows * self.cols, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2) * checkerboard_size  # cm

        self.frames = []
        self.calibrated_frames = []
        self.corners = []
        self.images = []
        self.last_image = None

        self.shape = None

        self.save_images = save_images
        self.downsample = downsample

    def recreate(self, checkerboard_detections, checkerboard_shape=None, height=1536, width=2048):
        if checkerboard_shape is not None:
            self.objp = checkerboard_shape

        self.shape = (height, width, 3)

        # now iterate through checkerboard detections and keep frames and corners not NaN
        self.frames = []
        self.corners = []

        for i in range(len(checkerboard_detections)):
            if np.all(~np.isnan(checkerboard_detections[i])):
                self.frames.append(i)
                self.corners.append(checkerboard_detections[i, :, None].astype(np.float32))

        keep = self.filter_corners()
        print("Keeping {} frames".format(np.mean(keep)))
        self.frames = list(np.array(self.frames)[keep])
        self.corners = list(np.array(self.corners)[keep])

        return self

    def process_frame(self, idx, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE  # + cv2.CALIB_CB_FAST_CHECK

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

    def get_rvecs_tvecs(self, mtx, dist, corners=None):
        if corners is None:
            corners = self.corners

        rvecs = []
        tvecs = []
        for imgpoint in corners:
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(self.objp, imgpoint, mtx, dist, confidence=0.9, reprojectionError=30)
            rvecs.append(rvec)
            tvecs.append(tvec)

        return rvecs, tvecs

    def filter_corners(self, thresh=10.0, return_errors=False, update_self=False):
        N = len(self.frames)
        objpoints = [self.objp] * N
        imgpoints = self.corners

        h, w, _ = self.shape
        mtx = cv2.initCameraMatrix2D(objpoints, imgpoints, (w, h))
        dist = np.zeros((5,))

        corners = self.corners
        rvecs, tvecs = self.get_rvecs_tvecs(mtx, dist, self.corners)

        print(len(rvecs), len(tvecs), len(corners))
        errors = []
        for i in range(len(rvecs)):
            imgpoints2, _ = cv2.projectPoints(self.objp, rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(corners[i], imgpoints2, cv2.NORM_L2)

            errors.append(error)

        if update_self:
            keep = np.array(errors) < thresh
            print("Keeping {} frames".format(np.mean(keep)))
            self.frames = list(np.array(self.frames)[keep])
            self.corners = list(np.array(self.corners)[keep])

        if return_errors:
            return np.array(errors)
        else:
            return np.array(errors) < thresh

    def calibrate_camera(self, flags=None, max_frames=100, filter=False):
        frames = self.frames
        N = len(frames)
        objpoints = [self.objp] * N
        imgpoints = self.corners

        if filter:
            keep = self.filter_corners()
            objpoints = list(np.array(objpoints)[keep])
            imgpoints = list(np.array(imgpoints)[keep])
            frames = list(np.array(frames)[keep])
            N = len(frames)

        # if N > max_frames take an approximately evently space subset of both lists
        # to make the total max_frames. cv2 calibration starts taking excessively long
        # much beyond this
        if N > max_frames:
            idx = np.linspace(0, N - 1, max_frames).astype(int)
            # randomly sample a set of frames
            # idx = np.random.choice(N, max_frames, replace=False)
            objpoints = list(np.array(objpoints)[idx])
            imgpoints = list(np.array(imgpoints)[idx])
            self.calibrated_frames = [self.frames[i] for i in idx.tolist()]
        else:
            self.calibrated_frames = self.frames

        h, w, _ = self.shape

        if True:
            if flags is None:
                flags = (
                    cv2.CALIB_FIX_K2
                    | cv2.CALIB_FIX_K3
                    | cv2.CALIB_FIX_K4
                    | cv2.CALIB_FIX_K5
                    | cv2.CALIB_ZERO_TANGENT_DIST
                    | cv2.CALIB_FIX_PRINCIPAL_POINT
                    | cv2.CALIB_FIX_ASPECT_RATIO
                )

            initial_matrix = np.array([[1950, 0, w / 2], [0, 1950, h / 2], [0, 0, 1]])

            _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints,
                imgpoints,
                (w, h),
                initial_matrix,
                np.zeros((5,)),
                flags=flags,
            )

            if dist[0] > 0:
                print("Warning: distortion is positive")
                flags = flags | cv2.CALIB_FIX_K1
                _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints,
                    imgpoints,
                    (w, h),
                    initial_matrix,
                    np.zeros((5,)),
                    flags=flags,
                )

            print(mtx.astype(int))
            print(dist)
            # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        else:
            mtx = cv2.initCameraMatrix2D(objpoints, imgpoints, (w, h))
            dist = np.zeros((5,))

            print(mtx.astype(int))

        rvecs = []
        tvecs = []
        for imgpoint in self.corners:
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(self.objp, imgpoint, mtx, dist, confidence=0.9, reprojectionError=30)
            rvecs.append(rvec)
            tvecs.append(tvec)

        self.calibrated_frames = self.frames
        return {
            "mtx": mtx,
            "dist": dist,
            "rvecs": rvecs,
            "tvecs": tvecs,
            "frames": self.calibrated_frames,
            "objpoints": objpoints,
            "imgpoints": imgpoints,
        }

    def get_points(self, idx):
        return [self.objp] * len(idx), list(np.array(self.corners)[idx])


def get_checkerboards(filenames, cam_names, max_frames=None, skip=1, multithread=False, filter_frames=True, charuco=True,**kwargs):
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

    if charuco:
        print("Running ChAruco Calibration")
        parsers = [ChArucoAccumulator(**kwargs) for _ in range(num_views)]
    else:
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
            cap, parser, cam_name, idx = params
            if idx == 0:
                progress_fn = trange
            else:
                progress_fn = trange

            for i in progress_fn(0, frames, skip):
                if skip != 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, img = cap.read()
                if not ret or img is None:
                    break

                parser.process_frame(i, img)
            print(f"{cam_name} detected {len(parser.frames)} frames")
            return parser

        parsers = pool.map(process_video, zip(caps, parsers, cam_names, range(num_views)))

    for c in caps:
        c.release()

    # for p in parsers:
    #    p.filter_corners(update_self=True)

    return parsers


def get_checkerboard_points(parsers: List[CheckerboardAccumulator]):
    """
    Extract the checkerboard points from a list of parsers
    """

    frames = np.sort(np.unique(np.concatenate([p.frames for p in parsers])))
    N = parsers[0].objp.shape[0]  # number of coordinates in pattern
    # N = (parsers[0].rows-1)*(parsers[0].cols-1)
    checkerboard_points = np.zeros((len(parsers), len(frames), N, 2)) * np.nan

    # convert format from the parser to a matrix of observations with nan for missing
    for i, p in enumerate(parsers):
        for j, f in enumerate(frames):
            idx = np.where(p.frames == f)[0]
            if len(idx) == 1:
                # print(f, idx,checkerboard_points[i, j, :, :].shape, p.corners[idx[0]].shape)
                checkerboard_points[i, j, :, :] = p.corners[idx[0]][:, 0, :]

    N, frames, _, _ = checkerboard_points.shape
    return checkerboard_points


def filter_keypoints(keypoints_2d, min_visible=3):
    """
    Only keep the times where keypoints are visible from multiple perspectives
    """

    visible = np.sum(~np.isnan(keypoints_2d[:, :, 0, 0]), axis=0)
    return keypoints_2d[:, visible >= min_visible]


def extract_origin(camera_params: dict, checkerboard_points, width: int = 5, checks: bool = True):
    """
    Extract the location and orientation of the charuco/checkerboard

    Our system uses a convention where the x and y axes define the plane of the
    floor and the z axis is vertical. This code defines the two dimensions of the
    checkerboard as x and z, and the y axis is the cross product of these two. This
    means it is assumes the checkerboard is vertical oriented.

    Args:
        camera_params (dict): camera parameters
        checkerboard_points (np.ndarray): 2D checkerboard points
        width (int, optional): width of the checkerboard
        checks (bool, optional): whether to check the results

    Returns:
        tuple: (x0, board_rotation)
    """
    cal3d = triangulate_point(camera_params, checkerboard_points)

    N = checkerboard_points.shape[2]
    M = N // width

    x0 = cal3d[0]
    x = cal3d[width-1] - x0
    z = cal3d[(M - 1) * width] - x0

    x = x / np.linalg.norm(x)
    z = -z / np.linalg.norm(z)
    y = -np.cross(x, z)

    board_rotation = np.stack([x, y, z])

    if checks:
        # confirm x and z are orthogonal, and that this is a valid rotation matrix
        # the typical reason this fails is if the width parameter is wrong
        assert np.abs(np.dot(x, z)) < 1e-2
        assert np.linalg.det(board_rotation) > 0.99

    return x0, board_rotation


def shift_calibration(camera_params, offset, rotation=np.eye(3), zoffset=None):
    from jaxlie import SO3

    camera_params = camera_params.copy()
    offset = offset / 1000.0

    camera_rotations = vmap(lambda x: SO3.exp(x).as_matrix())(camera_params["rvec"])
    tvec = camera_params["tvec"] + camera_rotations @ offset.reshape((3,))
    rvec = vmap(lambda x: (SO3.exp(x) @ SO3.from_matrix(rotation.T)).log())(camera_params["rvec"])

    camera_params["rvec"] = rvec
    camera_params["tvec"] = tvec

    if zoffset:
        camera_params = shift_calibration(camera_params, np.array([0, 0, -zoffset]))

    return camera_params


def initialize_group_calibration(parsers, max_cv2_frames=50):
    """
    Use detected checkerboards to initialize calibration parameters

    Parameters:
        parsers (List[CheckerboardAccumulator]) : detection of checkerboards

    Returns:
        calibration dictionary - contains intrinsic and extrinsic parameters and distortion
        checkerboard_params - initial location and rotations of checkerboards
        checkerboard_points - matrix of 2D corners (cameras X points X 2)
    """

    import itertools
    from jax import jit, vmap
    from jax import numpy as jnp
    from jaxlie import SO3, SE3

    ### First build up a mapping from all the cameras for the calibration graph

    # get the checkerboard points
    checkerboard_points = get_checkerboard_points(parsers)
    checkerboard_frames = np.sort(np.unique(np.concatenate([p.frames for p in parsers])))

    # make a map of the first pairs of frames where a checkerboard was found on both
    found = ~np.isnan(checkerboard_points[:, :, 0, 0])

    N = found.shape[0]
    matches = np.zeros((N, N), dtype=object)

    for i in range(N):
        matches[i, i] = np.array([])
        for j in range(i + 1, N):
            both = np.logical_and(found[i], found[j])
            idx = np.where(both)[0]
            if len(idx) > 0:
                # if idx is greater than 10, then take an even spaced 10
                if len(idx) > max_cv2_frames:
                    idx = idx[np.linspace(0, len(idx) - 1, max_cv2_frames).astype(int)]
                idx = checkerboard_frames[idx]
                matches[i, j] = idx
                matches[j, i] = idx

    # now use numpy map over the matrix and get counts
    get_count = lambda x: len(x) if type(x) == np.ndarray else 0
    matrix = np.vectorize(get_count)(matches)

    display(matrix)

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    matrix_csr = csr_matrix(matrix)

    # Applying reverse Cuthill-McKee algorithm
    reordering = reverse_cuthill_mckee(matrix_csr)
    reordered_matrix = matrix[reordering][:, reordering]
    display(reordered_matrix)

    # Step 1: Identify the hub node
    origin = np.argmax((matrix**2).sum(axis=0))

    # Step 2: Build the list of pairs to create a subgraph
    subgraph_edges = []
    visited = [origin]

    while len(visited) < len(matrix):
        # Find the neighbor with the highest weight that hasn't been visited yet
        max_weight = -1
        next_node = None
        for i in visited:
            for j in range(len(matrix)):
                if j not in visited and matrix[i, j] > max_weight:
                    max_weight = matrix[i, j]
                    next_node = (i, j, matrix[i, j])

        # Add the edge to the subgraph and mark the node as visited
        subgraph_edges.append(next_node)
        visited.append(next_node[1])

    assert max_weight > 0, "Missing edges found in the calibration graph"
    display(subgraph_edges)

    # find the row with the most calibrated cameras to call the origin
    print(f"Using camera {origin} as the origin")
    initialized = np.zeros((N,), dtype=bool)
    initialized[origin] = True

    Rs = np.zeros((N, 3, 3))
    Ts = np.zeros((N, 3))

    Rs[origin] = np.eye(3)  # establish the first camera as the reference

    ### Now initialize the camera calibrations
    cals = [p.calibrate_camera() for p in parsers]

    for (
        i,
        j,
        w,
    ) in subgraph_edges:  # in itertools.chain.from_iterable(itertools.repeat(np.arange(0, N), N * 2)):
        # iterate through cameras as a reference

        if ~initialized[i]:
            continue

        if initialized[j]:
            # no need to double compute
            continue

        assert type(matches[i, j]) == np.ndarray, f"No matches between {i} and {j}: {matches[i, j]}"

        p1 = parsers[i]
        p2 = parsers[j]

        frames = matches[i, j]

        print(f"Linking {i} -> {j} using {len(frames)} frames")

        objpoints = [p1.objp] * len(frames)

        idx1 = [p1.frames.index(f) for f in frames]
        im1points = [p1.corners[i] for i in idx1]

        idx2 = [p2.frames.index(f) for f in frames]
        im2points = [p2.corners[i] for i in idx2]

        h, w, _ = p1.shape

        cal1 = cals[i]
        cal2 = cals[j]

        stereocalib_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            1000,
            1e-6,
        )
        stereocalib_flags = cv2.CALIB_USE_INTRINSIC_GUESS

        res = cv2.stereoCalibrate(
            objpoints,
            im1points,
            im2points,
            cal1["mtx"].copy(),
            cal1["dist"].copy(),
            cal2["mtx"].copy(),
            cal2["dist"].copy(),
            (h, w),
            criteria=stereocalib_criteria,
            flags=stereocalib_flags,
        )

        retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = res

        # adjust rotation and translation based on the reference camera
        T = T[:, 0] + R @ Ts[i]
        R = R @ Rs[i]

        Rs[j] = R
        Ts[j] = T

        initialized[j] = True

    if np.any(~initialized):
        print("Not all cameras initialized. Calibration graph not fully connected")

    # now put it the standard structure
    camera_params = {
        "mtx": np.array([[c["mtx"][0, 0], c["mtx"][1, 1], c["mtx"][0, 2], c["mtx"][1, 2]] for c in cals]) / 1000.0,
        "dist": np.array([c["dist"].reshape((-1)) for c in cals]),
        "rvec": vmap(lambda x: SO3.from_matrix(x).log())(Rs),
        "tvec": Ts / 1000.0,
    }

    # TODO: get better initial estimates of checkerboard rvecs
    N = checkerboard_points.shape[1]
    checkerboard_rvecs = np.zeros((N, 3))
    checkerboard_tvecs = np.zeros((N, 3))

    checkerboard_params = {"rvecs": checkerboard_rvecs, "tvecs": checkerboard_tvecs}

    return camera_params, checkerboard_params, checkerboard_points


def checkerboard_reprojection_loss(camera_params, checkerboard_params, checkerboard_points, objp):
    checkerboard_rvecs = checkerboard_params["rvecs"]
    checkerboard_tvecs = checkerboard_params["tvecs"]

    estimated_3d_points = vmap(lambda a, b: get_checkboard_3d(a, b, objp))(checkerboard_rvecs, checkerboard_tvecs)
    err = reprojection_error(camera_params, checkerboard_points, estimated_3d_points)

    norm = False
    if norm:
        # err = jnp.linalg.norm(err, axis=-1) ** 2
        return jnp.nanmean(err**2)
    else:
        return jnp.nanmean(jnp.abs(err))


def checkerboard_reconstruction_loss(camera_params, checkerboard_params, checkerboard_points, objp, norm=False):
    checkerboard_rvecs = checkerboard_params["rvecs"]
    checkerboard_tvecs = checkerboard_params["tvecs"]

    estimated_3d_points = vmap(lambda a, b: get_checkboard_3d(a, b, objp))(checkerboard_rvecs, checkerboard_tvecs)
    err = reconstruction_error(camera_params, checkerboard_points, estimated_3d_points, stop_grad=True)

    if norm:
        # err = jnp.linalg.norm(err, axis=-1) ** 2
        return jnp.nanmean(err**2)
    else:
        return jnp.nanmean(jnp.abs(err))


@jit
def checkerboard_loss(checkerboard_params, camera_params, checkerboard_points, objp):
    # return checkerboard_reconstruction_loss(camera_params, checkerboard_params, checkerboard_points, objp)
    return checkerboard_reprojection_loss(camera_params, checkerboard_params, checkerboard_points, objp)


@jit
def camera_loss(camera_params, checkerboard_params, checkerboard_points, objp):
    # return checkerboard_reconstruction_loss(camera_params, checkerboard_params, checkerboard_points, False)
    return checkerboard_reprojection_loss(camera_params, checkerboard_params, checkerboard_points, objp)


@jit
def cycle_loss(camera_params, checkerboard_points):
    est_checkerboard_3d = triangulate_point(camera_params, checkerboard_points)
    # backpropgating through SVD takes a huge amount of time
    est_checkerboard_3d = jax.lax.stop_gradient(est_checkerboard_3d)
    err = reprojection_error(camera_params, checkerboard_points, est_checkerboard_3d)
    return jnp.nanmean(jnp.abs(err))


@partial(jit, static_argnums=(4, 5))
def update_checkerboard(
    checkerboard_params,
    camera_params,
    checkerboard_points,
    objp,
    stepsize=0.0,
    iterations=10,
):
    checkerboard_solver = jaxopt.GradientDescent(
        fun=checkerboard_loss,
        maxiter=iterations,
        verbose=False,
        stepsize=stepsize,
        acceleration=True,
    )
    return checkerboard_solver.run(
        checkerboard_params,
        camera_params=camera_params,
        checkerboard_points=checkerboard_points,
        objp=objp,
    )[0]


@partial(jit, static_argnums=(4, 5))
def update_camera(
    checkerboard_params,
    camera_params,
    checkerboard_points,
    objp,
    stepsize=0.0,
    iterations=10,
):
    camera_solver = jaxopt.GradientDescent(
        fun=camera_loss,
        maxiter=iterations,
        verbose=False,
        stepsize=stepsize,
        acceleration=True,
    )
    return camera_solver.run(
        camera_params,
        checkerboard_params=checkerboard_params,
        checkerboard_points=checkerboard_points,
        objp=objp,
    )[0]


@jit
def update_camera_cycle(camera_params, checkerboard_points, stepsize=0.0, iterations=10):
    cycle_solver = jaxopt.GradientDescent(fun=cycle_loss, maxiter=iterations, verbose=False, stepsize=stepsize)
    return cycle_solver.run(camera_params, checkerboard_points=checkerboard_points)[0]


# @partial(jit, static_argnums=(5,))
def update_combined(
    params,
    checkerboard_points,
    objp,
    stepsize=0.0,
    iterations=10,
    cycle=False,
    verbose=False,
):
    @jit
    def regularization(params):
        matrix_loss = jnp.sum(jax.nn.relu(-params["camera_params"]["mtx"])) * 1e5  # non-negative
        dist_loss = jnp.sum(params["camera_params"]["dist"] ** 2) * 1e4
        return matrix_loss + dist_loss

    @jit
    def loss(params):
        camera_params = params["camera_params"]
        checkerboard_params = params["checkerboard_params"]

        l = camera_loss(camera_params, checkerboard_params, checkerboard_points, objp)
        l = l + checkerboard_loss(checkerboard_params, camera_params, checkerboard_points, objp)
        if cycle:
            # this tends to be too unstable during optimization
            l = l + cycle_loss(camera_params, checkerboard_points)

        # l = l + regularization(params)
        return l

    # solver = jaxopt.ScipyMinimize(fun=loss, maxiter=iterations, verbose=True) #, stepsize=stepsize)
    solver = jaxopt.GradientDescent(
        fun=loss,
        maxiter=iterations,
        verbose=verbose,
        stepsize=stepsize,
        acceleration=True,
    )
    # solver = jaxopt.ProximalGradient(fun=loss, prox=regularization, maxiter=iterations, verbose=True, stepsize=stepsize)
    return solver.run(params)[0]


def refine_calibration(
    camera_params,
    checkerboard_params,
    checkerboard_points,
    objp,
    iterations=500,
    inner_iterations=10,
    verbose=True,
    cycle_consistency=False,
):
    height, width = 1536, 2048

    for i in range(iterations):
        checkerboard_params = update_checkerboard(
            checkerboard_params,
            camera_params,
            checkerboard_points,
            objp,
            iterations=1 if i < 50 else 100,
        )
        camera_params = update_camera(
            checkerboard_params,
            camera_params,
            checkerboard_points,
            objp,
            iterations=1 if i < 50 else inner_iterations,
        )

        if cycle_consistency:
            camera_params = update_camera_cycle(
                camera_params,
                checkerboard_points,
                iterations=1 if i < 50 else inner_iterations,
            )

        # apply some regularization to principal point and distortion
        decay = 0.9
        if i < 300:
            camera_params["dist"] = camera_params["dist"] * decay
        else:
            camera_params["dist"].at[:, 1:].set(camera_params["dist"][:, 1:] * decay)
        new_principal = camera_params["mtx"][:, 2:] + (np.array([[height / 2, width / 2]]) / 1000.0 - camera_params["mtx"][:, 2:]) * (1 - decay)
        camera_params["mtx"] = camera_params["mtx"].at[:, 2:].set(new_principal)

        if verbose:
            e1 = checkerboard_reprojection_loss(camera_params, checkerboard_params, checkerboard_points, objp)
            e2 = checkerboard_reconstruction_loss(camera_params, checkerboard_params, checkerboard_points, objp)
            e3 = cycle_loss(camera_params, checkerboard_points)
            print(f"Reprojection error {e1:.2f} pixels. Modeled reconstruction error {e2:.2f} mm. Cycle reprojection error {e3:0.2f} pixels.")

    return camera_params, checkerboard_params


def filter_calibration(checkerboard_points, checkerboard_params, min_visible=2):
    """
    Only keep the times where checkerboard is visible from multiple perspectives
    """

    visible = np.sum(~np.isnan(checkerboard_points[:, :, 0, 0]), axis=0)
    checkerboard_params = checkerboard_params.copy()
    checkerboard_params["rvecs"] = checkerboard_params["rvecs"][visible >= min_visible]
    checkerboard_params["tvecs"] = checkerboard_params["tvecs"][visible >= min_visible]
    checkerboard_points = checkerboard_points[:, visible >= min_visible]

    return checkerboard_points, checkerboard_params


def calibrate_bundle(
    parsers,
    camera_names=None,
    fisheye=False,
    verbose=True,
    zero_origin=False,
    extra_dist=False,
    both_focal=True,
    bundle_adjust=True,
    charuco = False,
    **kwargs,
):
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
    from aniposelib.boards import Checkerboard,CharucoBoard
    from aniposelib.utils import (
        get_initial_extrinsics,
        make_M,
        get_rtvec,
        get_connections,
    )
    from aniposelib.boards import merge_rows, extract_points, extract_rtvecs

    if camera_names is None:
        camera_names = list(range(len(parsers)))

    p = parsers[0]
    square_length = p.objp[1, 0] - p.objp[0, 0]
    cols = p.cols
    rows = p.rows

    cgroup = CameraGroup.from_names(camera_names, fisheye=fisheye)
    if charuco:
        board = CharucoBoard(cols, rows, square_length=p.square_size, marker_length=p.marker_size,marker_bits = p.marker_bits,dict_size=250)
    else:
        board = Checkerboard(cols, rows, square_length=square_length)

    for c in cgroup.cameras:
        c.extra_dist = extra_dist
        c.both_focal = both_focal

    if verbose:
        print(f"Extra: {cgroup.cameras[0].extra_dist}. Both: {cgroup.cameras[0].both_focal}")

    for cam, p in zip(cgroup.cameras, parsers):
        h, w, _ = p.shape
        cam.set_size((w, h))

    # convert parser results to the "rows" used by aniposelib
    all_rows = []
    for p in parsers:
        rows = []
        if charuco:
            for frame_num, corner, ids in zip(p.frames, p.corners,p.ids):
                row = {"framenum": frame_num, "corners": corner, "ids": ids}
                row["filled"] = board.fill_points(row["corners"], row["ids"])
                rows.append(row)
        else:
            for frame_num, corner in zip(p.frames, p.corners):
                row = {"framenum": frame_num, "corners": corner, "ids": None}
                row["filled"] = board.fill_points(row["corners"], row["ids"])
                rows.append(row)
            # if p.ids is not None:
            #     row["ids"] = p.ids[frame_num]
            # # iis = row["ids"].ravel()
            # for i, cxs in enumerate(row["corners"]):
            #         # print("EXCITING",i,cxs.shape)
            #         if cxs.shape[0] == 1:
            #             row["corners"][i] = np.expand_dims(cxs[0][0],axis=0)
            # # for i, cxs in zip(iis,row["corners"]):
            #         print(cxs.shape)
            # print(frame_num, iis)
            # if any(x >((p.cols-1)*(p.rows-1)) for x in iis):
            #     print("Skipping frame",frame_num,"because of ids",iis)
            #     row["filled"] = row["corners"]
            #     continue
        all_rows.append(rows)

    for rows, camera, parser in zip(all_rows, cgroup.cameras, parsers):
        size = camera.get_size()

        assert size is not None, "Camera with name {} has no specified frame size".format(camera.get_name())
        # print(board.get_all_calibration_points(rows),rows)
        if not extra_dist or fisheye:
            objp, imgp = board.get_all_calibration_points(rows)
            mixed = [(o, i) for (o, i) in zip(objp, imgp) if len(o) >= 7]
            # if len(mixed) > 0:
            objp, imgp = zip(*mixed)
            matrix = cv2.initCameraMatrix2D(objp, imgp, tuple(size))
            camera.set_camera_matrix(matrix)
        else:
            cal = parser.calibrate_camera()
            camera.set_camera_matrix(cal["mtx"])
            camera.set_distortions(cal["dist"])  # system only expects one distortion parameters

    for i, (row, cam) in enumerate(zip(all_rows, cgroup.cameras)):
        all_rows[i] = board.estimate_pose_rows(cam, row)
    if charuco:
        all_rows = [[r for r in rows if r['ids'].size >= 8] for rows in all_rows]
    
    merged = merge_rows(all_rows)
    # merged = merge_rows(all_rows)
    imgp, extra = extract_points(merged, board, min_cameras=2)

    rtvecs = extract_rtvecs(merged)
    rvecs, tvecs = get_initial_extrinsics(rtvecs, cgroup.get_names())
    cgroup.set_rotations(rvecs)
    cgroup.set_translations(tvecs)

    if verbose:
        print(cgroup.get_dicts())

    if bundle_adjust:
        error = cgroup.bundle_adjust_iter(imgp, extra, verbose=verbose, **kwargs)
    else:
        error = 100.0

    if zero_origin:
        x0, board_rotation = extract_origin(cgroup, imgp)

        # distance from my checkerboard top corner to ground
        cgroup = shift_calibration(cgroup, x0, board_rotation, 1245)

    return error, cgroup.get_dicts()


def run_calibration(
    vid_base,
    vid_path=".",
    return_parsers=False,
    frame_skip=2,
    jax_cal=False,
    checkerboard_size=110.0,
    checkerboard_dim=(4, 6),
    charuco = False,
    **kwargs,
):
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
    import json

    print(f"Running calibration on {vid_base} in {vid_path}. Checkerboard size {checkerboard_size} mm and dim {checkerboard_dim}")
    # search for files. expects them to be in the format vid_base.serial_number.mp4
    vids = []
    for v in os.listdir(vid_path):
        base, ext = os.path.splitext(v)
        if ext == ".mp4" and len(base.split(".")) == 2 and base.split(".")[0] == vid_base:
            vids.append(os.path.join(vid_path, v))
    vids.sort()

    cam_names = [os.path.split(v)[1].split(".")[1] for v in vids]

    # Loading the JSON file corresponding to the calibration to get the hash
    with open(os.path.join(vid_path, f"{vid_base}.json"), "r") as f:
        output_json = json.load(f)

    config_hash = output_json["camera_config_hash"]

    print(f"Cam names: {cam_names} camera config hash: {config_hash}")

    print(f"Found {len(vids)} videos. Now detecting checkerboards.")

    parsers = get_checkerboards(
        vids,
        cam_names,
        max_frames=5000,
        skip=frame_skip,
        save_images=False,
        downsample=1,
        multithread=True,
        checkerboard_size=checkerboard_size,
        checkerboard_dim=checkerboard_dim,
        charuco=charuco,
    )
    objp = parsers[0].objp

    print("Now running calibration")

    if jax_cal:
        if False:
            # this still doesn't work well enough
            (
                init_camera_params,
                init_checkerboard_params,
                checkerboard_points,
            ) = initialize_group_calibration(parsers)
        else:
            error, cgroup = calibrate_bundle(parsers, bundle_adjust=False, return_error=True, charuco=charuco)

            init_camera_params = {
                "mtx": np.array(
                    [
                        [
                            c["matrix"][0][0],
                            c["matrix"][1][1],
                            c["matrix"][0][2],
                            c["matrix"][1][2],
                        ]
                        for c in cgroup
                    ]
                )
                / 1000.0,
                "dist": np.array([c["distortions"] for c in cgroup]),
                "rvec": np.array([c["rotation"] for c in cgroup]),
                "tvec": np.array([c["translation"] for c in cgroup]) / 1000.0,
            }

        camera_params, error = checkerboard_bundle_calibrate(
            parsers,
            initial_params=init_camera_params,
            iterations=100,
            return_error=True,
        )

    else:
        error, cgroup = calibrate_bundle(parsers, n_samp_iter=2000, n_samp_full=2500, charuco=charuco)

        camera_params = {
            "mtx": np.array(
                [
                    [
                        c["matrix"][0][0],
                        c["matrix"][1][1],
                        c["matrix"][0][2],
                        c["matrix"][1][2],
                    ]
                    for c in cgroup
                ]
            )
            / 1000.0,
            "dist": np.array([c["distortions"] for c in cgroup]),
            "rvec": np.array([c["rotation"] for c in cgroup]),
            "tvec": np.array([c["translation"] for c in cgroup]) / 1000.0,
        }

        if np.isnan(error):
            error = 10000

    print("Frames: ", [p.frames[:5] for p in parsers])

    print("Zeroing coordinates")
    checkerboard_points = get_checkerboard_points(parsers)
    for i in range(checkerboard_points.shape[1]):
        try:
            x0, board_rotation = extract_origin(camera_params, checkerboard_points[:, i], width = 6 if charuco else 5)
        except:
            continue
        print(f'Accepted frame {i} as origin')
        break
    
    if charuco:
        # if the board is originally laying on the ground, according to our new procedure, then
        # we need to rotate the axis definitions
        board_rotation = board_rotation[[2, 0, 1]]
        # and then we need to flip the z axis to point down, and will flip a second axis to keep the handness
        # the same
        board_rotation = board_rotation * np.array([[1, -1, -1]]).T
    camera_params_zeroed = shift_calibration(camera_params, x0, board_rotation, zoffset=0 if charuco else 1245)
    params_dict = jax.tree_map(np.array, camera_params_zeroed)

    timestamp = vid_base.split("calibration_")[1]
    timestamp = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")

    entry = {
        "cal_timestamp": timestamp,
        "camera_config_hash": config_hash,
        "num_cameras": len(cam_names),
        "camera_names": cam_names,
        "camera_calibration": params_dict,
        "reprojection_error": error,
        "calibration_points": checkerboard_points,
        "calibration_shape": objp,
    }

    if return_parsers:
        return entry, parsers

    return entry


def cycle_residual_fun(camera_params, keypoints_2d, nonlinear_threshold=1000, sigma=250):
    """
    Compute the residual distance between reconstructed and reprojected keypoints

    Note: this is often unstable unless very clean keypoints and close to the right
    initial conditions. Likely because the triangulation process is going to be sensitive
    to these types of errors

    Parameters:
        camera_params: parameters of the camera (pose, intrinsic parameters etc.)
        keypoints_2d: observed 2D keypoints (cameras x time x joints x 2 or 3)

    If keypoints has a confidence, will use robust triangulation. Also does not score the
    residual for any unobserved keypoints, which should be NaN.

    Returns: the distance between the predicted and observed 2D keypoints
    """

    if keypoints_2d.shape[-1] == 3:
        est_keypoints_3d, keypoint_weights = robust_triangulate_points(camera_params, keypoints_2d, return_weights=True, sigma=sigma)
        keypoints_2d = keypoints_2d[..., :2]
        # normalize keypoint_weights along the last axis
        keypoint_weights = keypoint_weights / jnp.sum(keypoint_weights, axis=-1, keepdims=True)
        keypoint_weights = (keypoint_weights > 0.5) * 1.0
        keypoint_weights = jax.lax.stop_gradient(keypoint_weights)

    else:
        est_keypoints_3d = triangulate_point(camera_params, keypoints_2d)
        keypoint_weights = 1.0

    # backpropgating through SVD takes a huge amount of time
    est_keypoints_3d = jax.lax.stop_gradient(est_keypoints_3d)
    residuals = reprojection_error(camera_params, keypoints_2d, est_keypoints_3d)

    # now any points that are nan in keypoints_2d where not seen and should be zeroed in the
    # residuals
    mask = jnp.isnan(keypoints_2d)
    residuals = residuals * ~mask

    if False:
        residuals = jnp.linalg.norm(residuals, axis=-1)

        residuals = residuals * keypoint_weights

        # add some non linearities for outliers and such
        residuals = residuals**0.5
        residuals = residuals * (residuals < nonlinear_threshold) * residuals + (residuals >= nonlinear_threshold) * (
            nonlinear_threshold + (residuals - nonlinear_threshold) * 1e-5
        )

    # mask any nan values that still sneak through :(
    residuals = jnp.nan_to_num(residuals)

    return residuals


def huber_loss(err: float, delta: float = 1.0) -> float:
    """Huber loss.

    Args:
      err: difference between prediction and target
      delta: radius of quadratic behavior
    Returns:
      loss value

    References:
      https://en.wikipedia.org/wiki/Huber_loss
    """
    # err = jnp.linalg.norm(err, axis=-1)  # take euclidean distance first
    err = jnp.abs(err)
    return jnp.where(err > delta, delta * (err - 0.5 * delta), 0.5 * err**2)


def keypoint3d_reprojection_residuals(params, keypoints2d):
    """
    Compute the residual distance between estimated keypoints reprojected

    Parameters:
        params: parameters of the camera and also the checkerboard orientation
        checkerboard_points: observed 2D keypoints (cameras x time x joints x 2 or 3)
        checkerboard_shape: the shape of the checkerboard
    """

    keypoints3d = params["keypoints3d"]

    residuals = reprojection_error(params, keypoints2d, keypoints3d)

    mask = jnp.isnan(keypoints2d)
    residuals = residuals * ~mask

    residuals = jnp.abs(residuals)

    # mask any nan values that still sneak through :(
    residuals = jnp.nan_to_num(residuals)

    return residuals


def checkerboard_reprojection_residuals(params, checkerboard_points, checkerboard_shape, samples=None):
    """
    Compute the residual distance between reconstructed and reprojected checkboard

    Parameters:
        params: parameters of the camera and also the checkerboard orientation
        checkerboard_points: observed 2D keypoints (cameras x time x joints x 2 or 3)
        checkerboard_shape: the shape of the checkerboard
    """
    checkerboard_rvecs = params["checkerboard_rvecs"]
    checkerboard_tvecs = params["checkerboard_tvecs"]

    estimated_3d_points = vmap(lambda a, b: get_checkboard_3d(a, b, checkerboard_shape))(checkerboard_rvecs, checkerboard_tvecs)

    if samples is not None:
        residuals = reprojection_error(params, checkerboard_points[:, samples], estimated_3d_points[samples])
        mask = jnp.isnan(checkerboard_points[:, samples])
    else:
        residuals = reprojection_error(params, checkerboard_points, estimated_3d_points)
        mask = jnp.isnan(checkerboard_points)

    residuals = residuals * ~mask

    residuals = jnp.abs(residuals)

    # mask any nan values that still sneak through :(
    residuals = jnp.nan_to_num(residuals)

    return residuals


def checkerboard_and_keypoints_residuals(params, checkerboard_points, checkerboard_shape, keypoints2d, checkerboard_ratio=0.9):
    """
    Compute the residual distance between reconstructed and reprojected checkboard

    This combines the benefit of a strong geometric prior on the checkerboard with the
    flexibility of the keypoints, which also provide more observations across cameras

    Parameters:
        params: parameters of the camera and also the checkerboard orientation
        checkerboard_points: observed 2D keypoints (cameras x time x joints x 2 or 3)
        checkerboard_shape: the shape of the checkerboard
        keypoints2d: observed 2D keypoints (cameras x time x joints x 2 or 3)
    """

    L1 = checkerboard_reprojection_residuals(params, checkerboard_points, checkerboard_shape)
    L2 = keypoint3d_reprojection_residuals(params, keypoints2d)

    L1 = huber_loss(L1)
    L2 = huber_loss(L2)

    # get the checkerboard parameters and smoothness for tvec and rvec
    # checkerboard_rvecs = params["checkerboard_rvecs"] # ignore for now until handling unwrapping
    checkerboard_tvecs = params["checkerboard_tvecs"]
    delta_reg = jnp.mean(jnp.abs(jnp.diff(checkerboard_tvecs, axis=1)))

    # get the keypoint parameters and smoothness
    keypoints3d = params["keypoints3d"]
    delta_keypoints = jnp.mean(jnp.abs(jnp.diff(keypoints3d, axis=1)))

    if False:
        # for soem reason this introduces nan values???

        # order L2 by the error and compute the mean of the lowest half to discard outliers
        L2 = jnp.sort(L2, axis=-1)

        L2_main = L2[..., : L2.shape[-1] // 8]
        L2_outliers = L2[..., L2.shape[-1] // 8 :]

        # clip L2 outliers to 1000
        max_value = 1000.0
        L2_outliers = jnp.clip(L2_outliers, a_max=max_value)
        # and force all nan values to this max
        L2_outliers = jnp.nan_to_num(L2_outliers, nan=max_value)

        L2 = jnp.mean(L2_main) + jnp.mean(L2_outliers) * 1e-3

        print(
            "L1",
            jnp.mean(L1),
            "L2",
            jnp.mean(L2),
            "L2_main",
            jnp.mean(L2_main),
            "L2_outliers",
            jnp.mean(L2_outliers),
        )

    L = jnp.mean(L1) * checkerboard_ratio + L2 * (1 - checkerboard_ratio)

    return L  # + delta_keypoints + delta_reg


def camera_regularizer(params):
    """
    Add a regularization term to the camera parameters

    This is used to keep the parameters within reasonable bounds
    """

    mtx = params["mtx"]
    dist = params["dist"]

    # regularize the focal length
    focal_loss = jnp.sum(jax.nn.relu(-mtx[:, :2])) * 1e5

    # regularize the principal point
    principal_loss = jnp.sum(jax.nn.relu(-mtx[:, 2:])) * 1e0

    # regularize the distortion
    dist_loss = jnp.sum(dist**2) * 1e0

    # isotropic focal length
    focal_loss = focal_loss + jnp.sum((mtx[:, 0] - mtx[:, 1]) ** 2) * 1e1

    # principal offset
    expected_principal = jnp.array([[2047 / 2, 1535 / 2]]) / 1000.0
    principal_loss = principal_loss + jnp.sum((mtx[:, 2:] - expected_principal) ** 2) * 1e1

    return focal_loss + principal_loss + dist_loss


def make_residual_fun_wrapper(
    fun: Callable,
    initial_params: Dict,
    exclude_parameters: List[str] = [],
    reduce=False,
    regularizer: Callable = camera_regularizer,
):
    """
    Wraps the residual function to serialize/deserialize the parameters

    Parameters:
        fun: the residual function to wrap
        initial_params: the initial parameters to use
        exclude_parameters: parameters to exclude from the serialization
        reduce: whether to reduce the residuals to a single value
        regularizer: a function to apply to the residuals

    Returns:
        - Wrapped residual function
        - Initial vectorized parameters
    """

    import copy
    import jax.flatten_util

    # create a dictionary stripping out all elements in exclude_parameters
    params_dict = {k: v for k, v in initial_params.items() if k not in exclude_parameters}
    excluded_dict = {k: v for k, v in initial_params.items() if k in exclude_parameters}

    x, params_unpack = jax.flatten_util.ravel_pytree(params_dict)

    def restore_params(x):
        params = params_unpack(x)
        params = {**params, **excluded_dict}
        return params

    def residual_fun(x, *arg, **kwargs):
        params = restore_params(x)
        residuals = fun(params, *arg, **kwargs)
        residuals = residuals.reshape(-1)

        if reduce:
            residuals = jnp.mean(residuals)

            if regularizer:
                residuals = residuals + regularizer(params)

        return residuals

    return residual_fun, x, restore_params


def checkerboard_initialize(
    camera_params: Dict,
    checkerboard_points: jnp.ndarray,
):
    """
    Initialize the checkerboard parameters

    Parameters:
        camera_params: the camera parameters
        checkerboard_points: the checkerboard points
    """

    N = checkerboard_points.shape[1]

    checkerboard_rvecs = np.zeros((N, 3))
    checkerboard_tvecs = np.zeros((N, 3))

    checkerboard_3d = triangulate_point(camera_params, checkerboard_points)
    checkerboard_3d = robust_triangulate_points(camera_params, checkerboard_points)[..., :-1]  # drop confidence

    checkerboard_tvecs = jnp.mean(checkerboard_3d, axis=1)  # average over the spatial dimensions of checkerboard
    checkerboard_tvecs = checkerboard_tvecs / 1000

    # TODO: initialize rvecs at some point

    checkerboard_params = {
        "checkerboard_rvecs": checkerboard_rvecs,
        "checkerboard_tvecs": checkerboard_tvecs,
    }
    return checkerboard_params


def checkerboard_bundle_calibrate(
    parsers: List[CheckerboardAccumulator],
    max_frames_init: int = 30,
    iterations: int = 25,
    initial_params: Dict = None,
    initial_method_aniposelib: bool = False,
    threshold: float = 0.2,
    anneal: bool = True,
    checkerboard_reset_n: int = 0,
    random_sample_size: int = 50,
    return_error: bool = False,
):
    """
    Calibrate the camera and checkerboard using bundle adjustment

    This estimates the position and rotation of the checkerboard with the
    camera parameters. Uses an annealing schedule to add in the camera
    calibration matrix and distortion parameters.

    Parameters:
        parsers: the list of CheckerboardAccumulator parsers to use
        max_frames_init: the maximum number of frames to use for initialization
        iterations: the number of iterations to run
        initial_params: the initial parameters to use (optional)
        anneal: whether to anneal the set of parameters
    """

    if initial_params is None:
        if initial_method_aniposelib:
            from multi_camera.analysis.calibration import calibrate_bundle

            cgroup = calibrate_bundle(parsers, bundle_adjust=False)[1]
            initial_params = {
                "mtx": np.array(
                    [
                        [
                            c["matrix"][0][0],
                            c["matrix"][1][1],
                            c["matrix"][0][2],
                            c["matrix"][1][2],
                        ]
                        for c in cgroup
                    ]
                )
                / 1000.0,
                "dist": np.array([c["distortions"] for c in cgroup]),
                "rvec": np.array([c["rotation"] for c in cgroup]),
                "tvec": np.array([c["translation"] for c in cgroup]) / 1000.0,
            }
        else:
            initial_params = initialize_group_calibration(parsers, max_cv2_frames=max_frames_init)[0]

    print("initial_params", initial_params)

    checkerboard_points = get_checkerboard_points(parsers)
    checkerboard_points = filter_keypoints(checkerboard_points, 2)

    checkerboard_params = checkerboard_initialize(initial_params, checkerboard_points)
    params = {**initial_params, **checkerboard_params}

    residual_fun, x, restore_params = make_residual_fun_wrapper(
        checkerboard_reprojection_residuals,
        params,
        ["mtx", "dist", "rvec", "tvec"],
        reduce=True,
    )
    res = residual_fun(x, checkerboard_points=checkerboard_points, checkerboard_shape=parsers[0].objp)
    print("Initial residuals: ", res)
    optimizer = jaxopt.GradientDescent(
        fun=residual_fun,
        verbose=False,
        maxiter=5000,
        acceleration=True,
    )

    def refine_checkerboard_position(x):
        res = optimizer.run(
            x,
            checkerboard_points=checkerboard_points,
            checkerboard_shape=parsers[0].objp,
        )
        x = res.params
        res = residual_fun(
            x,
            checkerboard_points=checkerboard_points,
            checkerboard_shape=parsers[0].objp,
        )
        return x, res

    x, res = refine_checkerboard_position(x)
    camera_params = restore_params(x)
    print("After initializing checkerboard position: ", res)

    residual_fun, x, restore_params = make_residual_fun_wrapper(checkerboard_reprojection_residuals, camera_params, ["mtx", "dist"], reduce=True)
    optimizer = jaxopt.GradientDescent(fun=residual_fun, verbose=False, maxiter=5000, acceleration=True)

    total_samples = checkerboard_points.shape[1]

    for i in range(iterations):
        if i < iterations // 2 and random_sample_size > 0:
            this_sample_size = int(random_sample_size + (total_samples - random_sample_size) * (i / (iterations // 2)))
            random_samples = np.random.choice(total_samples, size=this_sample_size, replace=False)

            lm = False
            residual_fun, x, restore_params = make_residual_fun_wrapper(
                partial(checkerboard_reprojection_residuals, samples=random_samples),
                camera_params,
                ["mtx", "dist"],
                reduce=not lm,
            )
            if lm:
                optimizer = jaxopt.LevenbergMarquardt(residual_fun, verbose=True, maxiter=10)
            else:
                optimizer = jaxopt.GradientDescent(
                    fun=residual_fun,
                    verbose=False,
                    maxiter=1500,
                    stepsize=-10.0,
                    acceleration=True,
                )

            # adding this makes diagnostics more sensible by optimize the positions that were excluded
            # previously, but also slows things down without actually improve convergence
            x = refine_checkerboard_position(x)[0]
            camera_params = restore_params(x)

        pre_err = residual_fun(
            x,
            checkerboard_points=checkerboard_points,
            checkerboard_shape=parsers[0].objp,
        )
        res = optimizer.run(
            x,
            checkerboard_points=checkerboard_points,
            checkerboard_shape=parsers[0].objp,
        )
        x = res.params

        err = residual_fun(
            res.params,
            checkerboard_points=checkerboard_points,
            checkerboard_shape=parsers[0].objp,
        )
        camera_params = restore_params(x)
        cycle_err = cycle_residual_fun(camera_params, checkerboard_points)
        cycle_err = (cycle_err**2).mean()

        if lm:
            pre_err = np.mean(pre_err)
            err = np.mean(err)
        # monitor the cycle error, but it's not easy to use directly during optimization
        print(f"Iteration {i} error: {pre_err:0.3f} -> {err:0.3f}, cycle error: {cycle_err:0.3f}")

        if checkerboard_reset_n > 0 and i % checkerboard_reset_n == 0:
            checkerboard_params = checkerboard_initialize(initial_params, checkerboard_points)
            params["checkerboard_tvecs"] = checkerboard_params["checkerboard_tvecs"]

        # if anneal and i == iterations // 4:
        #     print("Adding distortion parameters")
        #     residual_fun, x, restore_params = make_residual_fun_wrapper(
        #         checkerboard_reprojection_residuals, camera_params, ["mtx"], reduce=True
        #     )
        #     residual_fun = jax.jit(residual_fun)
        #     optimizer = jaxopt.GradientDescent(fun=residual_fun, verbose=False, maxiter=5000, acceleration=True)

        # if anneal and i == (2 * iterations) // 4:
        #     print("Adding camera calibration matrix")

        #     # now allow the camera calibration matrix to change
        #     residual_fun, x, restore_params = make_residual_fun_wrapper(
        #         checkerboard_reprojection_residuals,
        #         camera_params,
        #         ["checkerboard_rvecs", "checkerboard_tvecs"],
        #         reduce=True,
        #     )
        #     residual_fun = jax.jit(residual_fun)
        #     optimizer = jaxopt.GradientDescent(fun=residual_fun, verbose=False, maxiter=5000, acceleration=True)

        if anneal and i == (iterations - 5):  # (1 * iterations) // 2:
            print("Allowing everything to change")

            this_sample_size = int(random_sample_size)
            random_samples = np.random.choice(total_samples, size=this_sample_size, replace=False)

            # now allow the everything to change
            residual_fun, x, restore_params = make_residual_fun_wrapper(
                # partial(checkerboard_reprojection_residuals, samples=random_samples),
                checkerboard_reprojection_residuals,
                camera_params,
                reduce=True,
            )
            residual_fun = jax.jit(residual_fun)
            optimizer = jaxopt.GradientDescent(
                fun=residual_fun,
                verbose=False,
                maxiter=15000,
                acceleration=True,
                stepsize=-0.1,
            )

        if err < threshold:
            break

    camera_params = restore_params(x)
    camera_params.pop("checkerboard_rvecs")
    camera_params.pop("checkerboard_tvecs")

    if return_error:
        error = float(checkerboard_loss(checkerboard_params, camera_params, checkerboard_points, parsers[0].objp))
        return camera_params, error

    return camera_params


def bundle_calibrate(
    initial_params: Dict,
    keypoints2d: np.ndarray,
    iterations: int = 25,
    threshold: float = 0.2,
):
    """
    Calibrate the camera using bundle adjustment for any 2D keypoints

    Compared to checkerboard_bundle_calibrate, this function does not constrain
    the keypoints to any geometry, so this can be applied to detection such as
    people in the scene.

    Parameters:
        initial_params: the initial parameters to use
        keypoints2d: the 2D keypoints to use
        iterations: the number of iterations to run
    """

    keypoints3d = triangulate_point(initial_params, keypoints2d)
    params = {"keypoints3d": keypoints3d, **initial_params}

    residual_fun, x, restore_params = make_residual_fun_wrapper(keypoint3d_reprojection_residuals, params, ["mtx", "dist"], reduce=True)
    residual_fun = jax.jit(residual_fun)

    res = residual_fun(x, keypoints2d=keypoints2d)
    print("Initial residuals: ", res)

    optimizer = jaxopt.GradientDescent(fun=residual_fun, verbose=False, maxiter=5000, acceleration=True, stepsize=10.0)

    for i in range(iterations):
        res = optimizer.run(x, keypoints2d=keypoints2d)
        x = res.params

        err = residual_fun(res.params, keypoints2d=keypoints2d)
        camera_params = restore_params(res.params)
        cycle_err = cycle_residual_fun(camera_params, keypoints2d)
        cycle_err = (cycle_err**2).mean()

        # monitor the cycle error, but it's not easy to use directly during optimization
        print(f"Iteration {i} error: {err}, cycle error: {cycle_err}")

        if i == (iterations // 4):
            residual_fun, x, restore_params = make_residual_fun_wrapper(
                keypoint3d_reprojection_residuals,
                camera_params,
                ["mtx", "dist"],
                reduce=True,
            )
            residual_fun = jax.jit(residual_fun)
            optimizer = jaxopt.GradientDescent(fun=residual_fun, verbose=False, maxiter=5000)

        if i == (2 * iterations) // 4:
            # now allow the camera calibration matrix to change
            residual_fun, x, restore_params = make_residual_fun_wrapper(
                keypoint3d_reprojection_residuals,
                camera_params,
                ["checkerboard_rvecs", "checkerboard_tvecs"],
                reduce=True,
            )
            residual_fun = jax.jit(residual_fun)
            optimizer = jaxopt.GradientDescent(fun=residual_fun, verbose=False, maxiter=5000, stepsize=-1.0)

        if i == (3 * iterations) // 4:
            # now allow the everything to change
            residual_fun, x, restore_params = make_residual_fun_wrapper(
                keypoint3d_reprojection_residuals,
                camera_params,
                reduce=True,
            )
            residual_fun = jax.jit(residual_fun)
            optimizer = jaxopt.GradientDescent(fun=residual_fun, verbose=False, maxiter=5000, stepsize=-1.0)

        if err < threshold:
            break

    camera_params.pop("keypoints3d")

    return camera_params


def bundle_checkerboard_and_keypoints_calibrate(
    parsers: List[CheckerboardAccumulator],
    keypoints2d: np.ndarray,
    initial_params: Dict = None,
    max_frames_init: int = 50,
    iterations: int = 25,
    threshold: float = 0.2,
    checkerboard_reset_n=5,
    keypoint_reset_n=5,
    anneal: bool = False,
):
    """
    Calibrate the camera using bundle adjustment for any 2D keypoints

    Compared to checkerboard_bundle_calibrate, this function does not constrain
    the keypoints to any geometry, so this can be applied to detection such as
    people in the scene.

    Parameters:
        parsers: the list of CheckerboardAccumulator parsers to use
        initial_params: the initial parameters to use
        keypoints2d: the 2D keypoints to use
        iterations: the number of iterations to run
    """

    if initial_params is None:
        initial_params = initialize_group_calibration(parsers, max_cv2_frames=max_frames_init)[0]

    # keypoints3d = triangulate_point(initial_params, keypoints2d)
    keypoints3d = robust_triangulate_points(initial_params, keypoints2d)

    checkerboard_shape = parsers[0].objp

    checkerboard_points = get_checkerboard_points(parsers)
    checkerboard_points = filter_keypoints(checkerboard_points, 2)

    checkerboard_params = checkerboard_initialize(initial_params, checkerboard_points)
    params = {**initial_params, "keypoints3d": keypoints3d, **checkerboard_params}

    # first, just reposition the checkerboard and keypoints
    residual_fun, x, restore_params = make_residual_fun_wrapper(
        checkerboard_and_keypoints_residuals,
        params,
        ["mtx", "dist", "rvec", "tvec"],
        reduce=True,
    )

    res = residual_fun(
        x,
        keypoints2d=keypoints2d,
        checkerboard_points=checkerboard_points,
        checkerboard_shape=checkerboard_shape,
    )
    print("Initial residuals: ", res)

    optimizer = jaxopt.GradientDescent(
        fun=residual_fun,
        verbose=False,
        maxiter=20000,
        acceleration=True,
        decrease_factor=0.1,
    )
    x = optimizer.run(
        x,
        keypoints2d=keypoints2d,
        checkerboard_points=checkerboard_points,
        checkerboard_shape=checkerboard_shape,
    ).params
    res = residual_fun(
        x,
        keypoints2d=keypoints2d,
        checkerboard_points=checkerboard_points,
        checkerboard_shape=checkerboard_shape,
    )
    params = restore_params(x)
    print(f"Residuals after first repositioning with fixed calibration: {res}")

    # now repositioning camera, but keep the camera intrinsics fixed
    residual_fun, x, restore_params = make_residual_fun_wrapper(
        checkerboard_and_keypoints_residuals,
        params,
        ["mtx", "dist"] if anneal else [],
        reduce=True,
    )
    optimizer = jaxopt.GradientDescent(fun=residual_fun, verbose=False, maxiter=5000, acceleration=True, stepsize=-1.0)

    for i in range(iterations):
        res = optimizer.run(
            x,
            checkerboard_points=checkerboard_points,
            checkerboard_shape=checkerboard_shape,
            keypoints2d=keypoints2d,
        )
        x = res.params

        if keypoint_reset_n > 0 and i % keypoint_reset_n == 0:
            print("resetting keypoints3d")
            keypoints3d = robust_triangulate_points(initial_params, keypoints2d)
            params["keypoints3d"] = keypoints3d

        if checkerboard_reset_n > 0 and i % checkerboard_reset_n == 0:
            print("resetting checkerboard")
            checkerboard_params = checkerboard_initialize(initial_params, checkerboard_points)
            params["checkerboard_tvecs"] = checkerboard_params["checkerboard_tvecs"]

        err = residual_fun(
            res.params,
            checkerboard_points=checkerboard_points,
            checkerboard_shape=checkerboard_shape,
            keypoints2d=keypoints2d,
        )
        camera_params = restore_params(res.params)
        kp_cycle_err = cycle_residual_fun(camera_params, keypoints2d)
        if True:
            # sort the squared error by the keypoints
            mask = ~jnp.isnan(keypoints2d)
            kp_cycle_err = kp_cycle_err[mask].reshape(-1)
            kp_cycle_err = jnp.sort(kp_cycle_err**2)
            # take the lowest 50% of the error
            kp_cycle_err = jnp.mean(kp_cycle_err[: len(kp_cycle_err) // 8])
        else:
            kp_cycle_err = (kp_cycle_err**2).mean()

        checkboard_cycle_err = cycle_residual_fun(camera_params, checkerboard_points)
        checkboard_cycle_err = (checkboard_cycle_err**2).mean()

        # monitor the cycle error, but it's not easy to use directly during optimization
        print(f"Iteration {i} error: {err}, kp_cycle_err {kp_cycle_err}, checkboard_cycle_err {checkboard_cycle_err}")

        if anneal and i == (iterations) // 4:
            print("Adding camera matrix and distortions, freezing checkerboard")
            # now allow the camera calibration matrix to change
            residual_fun, x, restore_params = make_residual_fun_wrapper(
                checkerboard_and_keypoints_residuals,
                camera_params,
                ["checkerboard_rvecs", "checkerboard_tvecs"],
                reduce=True,
            )
            residual_fun = jax.jit(residual_fun)
            optimizer = jaxopt.GradientDescent(fun=residual_fun, verbose=False, maxiter=5000, stepsize=-1.0)

        if anneal and i == (1 * iterations) // 4:
            print("Allowing everything to change")
            # now allow the everything to change
            residual_fun, x, restore_params = make_residual_fun_wrapper(
                checkerboard_and_keypoints_residuals,
                camera_params,
                ["dist"],
                reduce=True,
            )
            residual_fun = jax.jit(residual_fun)
            optimizer = jaxopt.GradientDescent(fun=residual_fun, verbose=False, maxiter=5000)

        if err < threshold:
            break

    camera_params = restore_params(x)
    camera_params.pop("keypoints3d")

    return camera_params


def dual_calibration_procedure(
    calibration_points: np.ndarray,
    keypoints2d: List[np.ndarray],
    calibration_shape: np.ndarray = None,
    init_camera_params: Dict = None,
    iterations: int = 25,
):
    """
    This function calibrates the cameras and the checkerboard at the same time.

    This is a convenience wrapper for this sequence that seems to work quite well. Will
    likely be modified and rolled into a part of the initial calibration.

    Parameters:
        calibration_points: the 3D points of the checkerboard
        keypoints2d: the 2D keypoints to use
        calibration_shape: the shape of the checkerboard
        init_camera_params: the initial camera parameters to use. If None, will compute
        iterations: the number of iterations to run each step
    """

    parsers = [CheckerboardAccumulator().recreate(calibration_points[i]) for i in range(calibration_points.shape[0])]

    assert calibration_shape is None, "Not implemented"

    if init_camera_params is None:
        try:
            # init methods using aniposelib code
            from multi_camera.analysis.calibration import calibrate_bundle

            error, cgroup = calibrate_bundle(parsers, bundle_adjust=False, verbose=False)

            init_camera_params = {
                "mtx": np.array(
                    [
                        [
                            c["matrix"][0][0],
                            c["matrix"][1][1],
                            c["matrix"][0][2],
                            c["matrix"][1][2],
                        ]
                        for c in cgroup
                    ]
                )
                / 1000.0,
                "dist": np.array([c["distortions"] for c in cgroup]),
                "rvec": np.array([c["rotation"] for c in cgroup]),
                "tvec": np.array([c["translation"] for c in cgroup]) / 1000.0,
            }

        except:
            print('Using "initialize_group_calibration" to initialize camera parameters. Likely incomplete linkage in video.')
            (
                init_camera_params,
                checkerboard_params,
                checkerboard_points,
            ) = initialize_group_calibration(parsers)

    # now performan optimzation using just the checkerboard
    checkerboard_calibrate = checkerboard_bundle_calibrate(parsers, initial_params=init_camera_params, anneal=True, iterations=iterations)

    # now clean the keypoints and perform this step
    from pose_pipeline.wrappers.bridging import normalized_joint_name_dictionary

    keypoints2d_cleaned = np.stack(keypoints2d, axis=0)

    if keypoints2d_cleaned.shape[-1] == 3:
        poor_confidence = keypoints2d_cleaned[..., 2] < 0.8
        keypoints2d_cleaned[poor_confidence, :] = np.nan

        # drop confidence now
        keypoints2d_cleaned = keypoints2d_cleaned[..., :2]

    if keypoints2d_cleaned.shape[2] == 87:
        assert keypoints2d_cleaned.shape[2] == 87, "expects MOVI 87 keypoints"
        # https://github.com/peabody124/PosePipeline/blob/main/pose_pipeline/wrappers/bridging.py
        joint_names = normalized_joint_name_dictionary["bml_movi_87"]
        track_joints = [
            "Pelvis",
            "Left Ankle",
            "Left Knee",
            "Left Hip",
            "Right Ankle",
            "Right Knee",
            "Right Hip",
            "Right Wrist",
            "Right Elbow",
            "Right Shoulder",
            "Left Wrist",
            "Left Elbow",
            "Left Shoulder",
        ]

        joint_idx = np.array([joint_names.index(j) for j in track_joints])
        keypoints2d_cleaned = keypoints2d_cleaned[:, :, joint_idx, :]

        print(keypoints2d_cleaned.shape)

    print("Starting second stage optimization")
    camera_params = bundle_checkerboard_and_keypoints_calibrate(
        parsers,
        keypoints2d_cleaned,
        initial_params=checkerboard_calibrate,
        iterations=iterations,
        anneal=True,
    )

    # now zero coordinates
    checkerboard_points = get_checkerboard_points(parsers)
    x0, board_rotation = extract_origin(camera_params, checkerboard_points[:, 5:])
    camera_params_zeroed = shift_calibration(camera_params, x0, board_rotation, zoffset=1245)
    params_dict = jax.tree_map(np.array, camera_params_zeroed)

    return params_dict


def test():
    from multi_camera.datajoint.multi_camera_dj import (
        MultiCameraRecording,
        SingleCameraVideo,
        CalibratedRecording,
        Calibration,
    )
    from pose_pipeline.pipeline import TopDownPerson, TopDownMethodLookup

    test_key = (
        MultiCameraRecording * CalibratedRecording & 'video_base_filename="p617_fast_20230601_091424" and cal_timestamp < "2024-01-01"'
    ).fetch1("KEY")

    keypoints2d, camera_name = (TopDownPerson * SingleCameraVideo * TopDownMethodLookup & test_key & "top_down_method=12").fetch(
        "keypoints", "camera_name"
    )
    cal_camera_names, saved_camera_params = (Calibration * CalibratedRecording & test_key).fetch1("camera_names", "camera_calibration")
    calibration_points, calibration_shape = (Calibration & test_key).fetch1("calibration_points", "calibration_shape")

    calibration = dual_calibration_procedure(calibration_points, keypoints2d, iterations=10)


def plot_cal(camera_params):
    import cv2
    from matplotlib import pyplot as plt

    tvec = camera_params["tvec"]
    rvec = camera_params["rvec"]

    pixel_size_micron = 3.45
    pixels_per_mm = 1000 / pixel_size_micron

    # Removed display function calls to prevent error
    # Assuming camera_params['mtx'] and camera_params['dist'] are well-defined

    # Convert rotation vectors to rotation matrices
    rmats = [cv2.Rodrigues(np.array(r[None, :]))[0].T for r in rvec]

    pos = np.array([-R.dot(t) for R, t in zip(rmats, tvec)])

    drange = np.max(np.abs(pos)) * 1.1

    fig = plt.figure(figsize=(9, 4))

    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c="r", marker="o")

    ax.set_xlim(-drange, drange)
    ax.set_ylim(-drange, drange)
    ax.set_zlim(-drange, drange)

    # 3D orientation visualization
    for i, (R, t) in enumerate(zip(rmats, pos)):
        length = drange / 5.0
        x_axis = np.array([length, 0, 0])
        y_axis = np.array([0, length, 0])
        z_axis = np.array([0, 0, length])
        x_axis = R.dot(x_axis) + t
        y_axis = R.dot(y_axis) + t
        z_axis = R.dot(z_axis) + t
        ax.quiver(
            t[0],
            t[1],
            t[2],
            x_axis[0] - t[0],
            x_axis[1] - t[1],
            x_axis[2] - t[2],
            color="r",
            arrow_length_ratio=0.5,
        )
        ax.quiver(
            t[0],
            t[1],
            t[2],
            y_axis[0] - t[0],
            y_axis[1] - t[1],
            y_axis[2] - t[2],
            color="g",
            arrow_length_ratio=0.1,
        )
        ax.quiver(
            t[0],
            t[1],
            t[2],
            z_axis[0] - t[0],
            z_axis[1] - t[1],
            z_axis[2] - t[2],
            color="b",
            arrow_length_ratio=0.1,
        )

    ax = fig.add_subplot(122)
    ax.scatter(pos[:, 0], pos[:, 1], c="r", marker="o")
    ax.set_xlim(-drange, drange)
    ax.set_ylim(-drange, drange)

    # 2D orientation visualization
    for i, (R, t) in enumerate(zip(rmats, pos)):
        length = drange / 10.0
        x_axis = np.array([length, 0, 0])
        y_axis = np.array([0, 0, length])
        x_axis = R.dot(x_axis) + t
        y_axis = R.dot(y_axis) + t
        # ax.quiver(t[0], t[1], x_axis[0]-t[0], x_axis[1]-t[1], color='r', angles='xy', scale_units='xy', scale=0.5, headlength=4, headwidth=4)
        ax.quiver(
            t[0],
            t[1],
            y_axis[0] - t[0],
            y_axis[1] - t[1],
            color="g",
            angles="xy",
            scale_units="xy",
            scale=0.5,
            headlength=4,
            headwidth=4,
        )

    # Display numbers by the cameras
    for i in range(pos.shape[0]):
        ax.text(pos[i, 0] + 0.2, pos[i, 1] + 0.2, str(i), color="k")

    plt.show()


def zero_calibration(camera_params: Dict, parsers: List[CheckerboardAccumulator]):
    """
    Zero the calibration by shifting the camera parameters to the origin

    Parameters:
        camera_params: the camera parameters
        parsers: the list of CheckerboardAccumulator parsers to use

    Returns:
        the zeroed camera parameters
    """

    checkerboard_points = get_checkerboard_points(parsers)
    x0, board_rotation = extract_origin(camera_params, checkerboard_points[:, 5:])
    camera_params_zeroed = shift_calibration(camera_params, x0, board_rotation, zoffset=1245)
    params_dict = jax.tree_map(np.array, camera_params_zeroed)

    return params_dict


if __name__ == "__main__":
    test()
