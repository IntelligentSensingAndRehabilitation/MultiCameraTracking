
import cv2
import numpy as np
from tqdm import trange

class CheckerboardAccumulator:

    def __init__(self, checkerboard_size=110.0, cherboard_dim=(4,6), downsample=1):
        self.rows, self.cols = cherboard_dim

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((self.rows*self.cols,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.cols,0:self.rows].T.reshape(-1,2) * checkerboard_size # cm

        self.frames = []
        self.corners = []

        self.shape = None

        self.downsample=downsample

    def process_frame(self, idx, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        #chessboard_flags = cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_LARGER + chessboard_flags

        gray_ds = cv2.resize(gray, (img.shape[1] // self.downsample, img.shape[0] // self.downsample))
        ret, corners = cv2.findChessboardCorners(gray, (self.cols, self.rows), chessboard_flags)


        if not self.shape:
            self.shape = img.shape

        if ret:
            corners = corners * self.downsample

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

            self.frames.append(idx)
            self.corners.append(corners2)

        return ret

    def calibrate_camera(self):
        N = len(self.frames)
        objpoints = [self.objp] * N
        imgpoints = self.corners

        h, w, _ = self.shape

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (h, w), None, None)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        return {'mtx': mtx, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs,
                'newcameramtx': newcameramtx, 'roi': roi}

    def get_points(self, idx):
        return [self.objp] * len(idx), list(np.array(self.corners)[idx])


def get_checkerboards(filenames, max_frames=None, skip=1, downsample=1):

    num_views = len(filenames)

    caps = [cv2.VideoCapture(f) for f in filenames]
    parsers = [CheckerboardAccumulator(downsample=downsample) for _ in range(num_views)]

    frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    frames = min(max_frames or frames, frames)

    for i in trange(0, frames, skip):

        for c, p in zip(caps, parsers):

            c.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, img = c.read()

            if not ret or img is None:
                break

            if (i % 50) != 0:
                continue

            p.process_frame(i, img)

    for c in caps:
        c.release()

    return parsers


def calibrate_pair(p1, p2,
                   stereocalib_flags = cv2.CALIB_USE_INTRINSIC_GUESS):
    _, idx0, idx1 = np.intersect1d(p1.frames, p2.frames, return_indices=True)

    assert len(idx0) > 0, "No overlapping frames"

    objpoints, im1points = p1.get_points(idx0)
    _, im2points = p2.get_points(idx1)

    h, w, _ = p1.shape

    calibration1 = p1.calibrate_camera()
    calibration2 = p2.calibrate_camera()

    stereocalib_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6)

    res = cv2.stereoCalibrate(objpoints, im1points, im2points,
                              calibration1['mtx'].copy(), calibration1['dist'].copy(),
                              calibration2['mtx'].copy(), calibration2['dist'].copy(),
                              (h, w),
                              criteria = stereocalib_criteria,
                              flags = stereocalib_flags)
    stereocalib_retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = res

    cal = {'cameraMatrix1': cameraMatrix1, 'distCoeffs1': distCoeffs1,
           'cameraMatrix2': cameraMatrix2, 'distCoeffs2': distCoeffs2,
           'R': R, 'T': T, 'E': E, 'F': F, 'N': len(idx0)}

    return cal