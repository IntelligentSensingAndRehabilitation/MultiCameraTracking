import pytest
import numpy as np
import cv2


def test_project():
    from multi_camera.analysis.camera import project, project_distortion

    K = np.diag([100.0, 100.0, 1.0])
    K[0,2] = 50
    K[1,2] = 50
    R = np.random.normal(size=(1,3)) * 0.1
    T = np.array([[0.0, 0.0, 0.0]])
    dist = np.zeros((1,5))
    camera_params = {'mtx': np.array([[K[0,0], K[1,1], K[0,2], K[1,2]]]) / 1000.0,
                     'rvec': R,
                     'tvec': T / 1000.0,
                     'dist': dist}

    objp = np.random.normal(size=(5,3))*5 + np.array([[100.0, 0.0, 100.0]])

    cv2_projected = cv2.projectPoints(objp, R, T, K, dist)[0][:, 0]
    projected = project_distortion(camera_params, 0, objp)

    err = np.linalg.norm(cv2_projected - projected, axis=-1)

    assert np.all(err < 1e-1)


def test_undistort():
    from multi_camera.analysis.camera import undistort_points

    N = 9

    p = np.random.normal(size=(N,2)) * 20
    dist = np.random.uniform(size=(1,5)) * 1e-2
    #dist = np.array([[-0.1, 0.1, 0.1, 0.2, 0.0]])

    cv2_undistorted = cv2.undistortPoints(p, np.eye(3), dist)[:, 0]
    undistorted = undistort_points(p, np.eye(3), dist)

    err = np.linalg.norm(cv2_undistorted - undistorted, axis=-1)
    print(err)
    assert np.all(err < 1e-5)

def test_distort3d():
    from multi_camera.analysis.camera import get_intrinsic, distort_3d, project_distortion

    K = np.diag([100.0, 100.0, 1.0])
    K[0,2] = 50
    K[1,2] = 50
    R = np.random.normal(size=(1,3)) * 0.1
    T = np.array([[0.0, 0.0, 0.0]])
    dist = np.zeros((1,5))
    camera_params = {'mtx': np.array([[K[0,0], K[1,1], K[0,2], K[1,2]]]) / 1000.0,
                     'rvec': R,
                     'tvec': T / 1000.0,
                     'dist': dist}

    objp = np.random.normal(size=(5,3))*5 + np.array([[100.0, 0.0, 100.0]])

    # manually go through two steps
    distorted = distort_3d(camera_params, 0, objp)
    K = get_intrinsic(camera_params, 0)
    proj = (K @ distorted[..., None])
    uv_1 = proj[..., :2, 0] / proj[..., 2, :]

    # use the direct method (tested above)
    uv_2 = project_distortion(camera_params, 0, objp)

    err = np.linalg.norm(uv_1 - uv_2, axis=-1)
    assert np.all(err < 1e-5)
