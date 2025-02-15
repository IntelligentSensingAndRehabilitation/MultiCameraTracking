import pytest
import numpy as np
from jax import numpy as jnp
import cv2


def test_project():
    from multi_camera.analysis.camera import project, project_distortion

    K = np.diag([100.0, 100.0, 1.0])
    K[0, 2] = 50
    K[1, 2] = 50
    R = np.random.normal(size=(1, 3)) * 0.1
    T = np.array([[0.0, 0.0, 0.0]])
    dist = np.zeros((1, 5))
    camera_params = {"mtx": np.array([[K[0, 0], K[1, 1], K[0, 2], K[1, 2]]]) / 1000.0, "rvec": R, "tvec": T / 1000.0, "dist": dist}

    objp = np.random.normal(size=(5, 3)) * 5 + np.array([[100.0, 0.0, 100.0]])

    cv2_projected = cv2.projectPoints(objp, R, T, K, dist)[0][:, 0]
    projected = project_distortion(camera_params, 0, objp)

    err = np.linalg.norm(cv2_projected - projected, axis=-1)

    assert np.all(err < 1e-1)


def test_undistort():
    from multi_camera.analysis.camera import undistort_points

    N = 9

    p = np.random.normal(size=(N, 2)) * 20
    dist = np.random.uniform(size=(1, 5)) * 1e-2
    # dist = np.array([[-0.1, 0.1, 0.1, 0.2, 0.0]])

    cv2_undistorted = cv2.undistortPoints(p, np.eye(3), dist)[:, 0]
    undistorted = undistort_points(p, np.eye(3), dist)

    err = np.linalg.norm(cv2_undistorted - undistorted, axis=-1)
    print(err)
    assert np.all(err < 1e-5)


def test_distort3d():
    from multi_camera.analysis.camera import get_intrinsic, distort_3d, project_distortion

    K = np.diag([100.0, 100.0, 1.0])
    K[0, 2] = 50
    K[1, 2] = 50
    R = np.random.normal(size=(1, 3)) * 0.1
    T = np.array([[0.0, 0.0, 0.0]])
    dist = np.zeros((1, 5))
    camera_params = {"mtx": np.array([[K[0, 0], K[1, 1], K[0, 2], K[1, 2]]]) / 1000.0, "rvec": R, "tvec": T / 1000.0, "dist": dist}

    objp = np.random.normal(size=(5, 3)) * 5 + np.array([[100.0, 0.0, 100.0]])

    # manually go through two steps
    distorted = distort_3d(camera_params, 0, objp)
    K = get_intrinsic(camera_params, 0)
    proj = K @ distorted[..., None]
    uv_1 = proj[..., :2, 0] / proj[..., 2, :]

    # use the direct method (tested above)
    uv_2 = project_distortion(camera_params, 0, objp)

    err = np.linalg.norm(uv_1 - uv_2, axis=-1)
    assert np.all(err < 1e-5)


def test_robust_triangulate_point_single():
    """
    Test robust_triangulate_point_single for a single (time, joint) sample.
    """
    from multi_camera.analysis.camera import robust_triangulate_point_single
    from jax import numpy as jnp

    # --- Set up a two-camera system with an increased baseline ---
    K = np.diag([100.0, 100.0, 1.0])
    K[0, 2] = 50
    K[1, 2] = 50

    # Camera 1: no rotation, zero translation.
    R1 = np.array([[0.0, 0.0, 0.0]])
    T1 = np.array([[0.0, 0.0, 0.0]])
    # Camera 2: small rotation and a larger translation along x.
    R2 = np.array([[0.05, 0.0, 0.0]])
    T2 = np.array([[1500.0, 0.0, 0.0]])  # baseline increased to 50 mm

    # Prepare intrinsics for each camera.
    mtx1 = np.array([[K[0, 0], K[1, 1], K[0, 2], K[1, 2]]])
    mtx2 = np.array([[K[0, 0], K[1, 1], K[0, 2] + 5, K[1, 2]]])
    mtx = np.vstack([mtx1, mtx2]) / 1000.0

    rvec = np.vstack([R1, R2])
    tvec = np.vstack([T1, T2]) / 1000.0  # convert from mm to meters
    dist = np.zeros((2, 5))

    camera_params = {
        "mtx": mtx,
        "rvec": rvec,
        "tvec": tvec,
        "dist": dist,
    }

    # --- Create synthetic 2D observations ---
    # Define a known 3D point (in mm)
    point3d_true = np.array([100.0, 0.0, 100.0])

    projected_points = []
    for i in range(2):
        # cv2.projectPoints expects translation in the same units as the 3D point.
        if i == 0:
            proj, _ = cv2.projectPoints(point3d_true[None, :], R1, T1, K, dist[i])
        else:
            proj, _ = cv2.projectPoints(point3d_true[None, :], R2, T2, K, dist[i])
        # proj has shape (1, 1, 2)
        projected_points.append(proj[0, 0])
    projected_points = np.array(projected_points)  # shape: (2, 2)

    # Append a confidence channel (set to 1 for all observations).
    points2d = np.concatenate([projected_points, np.ones((2, 1))], axis=-1)  # shape: (2, 3)
    points2d = jnp.array(points2d)

    # --- Run robust triangulation for the single (time, joint) sample ---
    robust_point3d = robust_triangulate_point_single(camera_params, points2d, sigma=1000, threshold=0.5)
    robust_point3d_np = np.array(robust_point3d)
    err = np.linalg.norm(robust_point3d_np[:3] - point3d_true)
    # Allow a tolerance of 5 mm.
    assert err < 5.0, f"Robust triangulation error too high: {err:.2f} mm"


def test_robust_triangulate_points():
    """
    Test robust_triangulate_points over multiple time frames and joints.
    """
    from multi_camera.analysis.camera import robust_triangulate_points
    from jax import numpy as jnp

    # --- Set up a two-camera system (same as above) ---
    K = np.diag([100.0, 100.0, 1.0])
    K[0, 2] = 50
    K[1, 2] = 50

    R1 = np.array([[0.0, 0.0, 0.0]])
    T1 = np.array([[0.0, 0.0, 0.0]])
    R2 = np.array([[0.05, 0.0, 0.0]])
    T2 = np.array([[1500.0, 0.0, 0.0]])  # increased baseline

    mtx1 = np.array([[K[0, 0], K[1, 1], K[0, 2], K[1, 2]]])
    mtx2 = np.array([[K[0, 0], K[1, 1], K[0, 2] + 5, K[1, 2]]])
    mtx = np.vstack([mtx1, mtx2]) / 1000.0

    rvec = np.vstack([R1, R2])
    tvec = np.vstack([T1, T2]) / 1000.0
    dist = np.zeros((2, 5))

    camera_params = {
        "mtx": mtx,
        "rvec": rvec,
        "tvec": tvec,
        "dist": dist,
    }

    # --- Create synthetic 2D observations for multiple time frames and joints ---
    T_steps = 100  # number of time frames
    J = 10  # number of joints
    N = 2  # number of cameras

    point3d_true = np.array([100.0, 0.0, 100.0])

    # Allocate array for 2D points: shape (num_cameras, T, J, 3)
    points2d = np.zeros((N, T_steps, J, 3))
    for i in range(N):
        for t in range(T_steps):
            for j in range(J):
                if i == 0:
                    proj, _ = cv2.projectPoints(point3d_true[None, :], R1, T1, K, dist[i])
                else:
                    proj, _ = cv2.projectPoints(point3d_true[None, :], R2, T2, K, dist[i])
                uv = proj[0, 0]
                points2d[i, t, j, :2] = uv
                points2d[i, t, j, 2] = 1.0  # confidence

    points2d = jnp.array(points2d)

    # --- Run robust triangulation for all time and joints ---
    robust_points3d, weights = robust_triangulate_points(camera_params, points2d, sigma=150, threshold=0.5, return_weights=True)
    robust_points3d_np = np.array(robust_points3d)

    # check size of weights and robust_points3d
    assert weights.shape == (N, T_steps, J)
    assert robust_points3d_np.shape == (T_steps, J, 4)

    # Check that each triangulated point is close to the true 3D point.
    errs = np.linalg.norm(robust_points3d_np[..., :3] - point3d_true, axis=-1)
    assert np.all(errs < 5.0), f"Some robust triangulation errors are too high: {errs}"


def test_real_cam_robust_triangulate_points():
    """
    Test robust_triangulate_points over multiple time frames and joints
    using real camera calibration data.

    We fetch a calibration record, generate synthetic 3D scene points (in mm),
    project them into each camera to obtain 2D observations, run robust
    triangulation, and compare the reconstructed 3D points (converted back to mm)
    to the original scene points.
    """
    from multi_camera.analysis.camera import robust_triangulate_points
    from multi_camera.datajoint.multi_camera_dj import Calibration

    # Fetch a single calibration record.
    camera_params = Calibration.fetch("camera_calibration", limit=1)[0]

    # Determine number of cameras from calibration.
    N = camera_params["mtx"].shape[0]

    # Simulation dimensions.
    T_steps = 10  # number of time frames
    J = 5  # number of joints

    # Generate random 3D scene points (in mm) with shape (T_steps, J, 3).
    point3d_true = np.random.normal(size=(T_steps, J, 3))
    # Convert to meters (since calibrations typically use meters).
    point3d_true_m = point3d_true / 1000.0

    # Allocate an array for 2D observations with shape (num_cameras, T_steps, J, 3).
    points2d = np.zeros((N, T_steps, J, 3))
    for i in range(N):
        # Reconstruct the intrinsic matrix for camera i.
        fx, fy, cx, cy = camera_params["mtx"][i]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        # Extract rotation (rvec) and translation (tvec) for camera i.
        rvec = camera_params["rvec"][i : i + 1]  # shape (1, 3)
        T = camera_params["tvec"][i : i + 1]  # shape (1, 3)
        # Get distortion coefficients.
        dist = camera_params["dist"][i]
        # Loop over time frames and joints.
        for t in range(T_steps):
            for j in range(J):
                # Get the 3D point for this time and joint.
                p = point3d_true_m[t, j, :]  # shape (3,)
                # cv2.projectPoints expects input shape (N,3).
                proj, _ = cv2.projectPoints(p[None, :], rvec, T, K, dist)
                uv = proj[0, 0]  # shape (2,)
                points2d[i, t, j, :2] = uv
                points2d[i, t, j, 2] = 1.0  # full confidence

    # Convert synthetic 2D observations to a JAX array.
    points2d = jnp.array(points2d)

    # Run robust triangulation.
    robust_points3d = robust_triangulate_points(camera_params, points2d, sigma=150, threshold=0.5)
    robust_points3d_np = np.array(robust_points3d)
    # Assume robust_points3d are in meters; convert to mm.
    robust_points3d_mm = robust_points3d_np[..., :3] * 1000.0

    # Compute the Euclidean error for each (time, joint) sample.
    errs = np.linalg.norm(robust_points3d_mm - point3d_true, axis=-1)
    print("Reprojection errors (mm):", errs)
    # Assert that all errors are below 10 mm.
    assert np.all(errs < 10.0), f"Some robust triangulation errors are too high: {errs}"

    # Make sure we get the same results without fast inference

    robust_points3d_mm_vmap = robust_triangulate_points(camera_params, points2d, sigma=150, threshold=0.5, fast_inference=False)
    robust_points3d_mm_vmap_np = np.array(robust_points3d_mm_vmap)
    # convert to mm
    robust_points3d_mm_vmap_mm = robust_points3d_mm_vmap_np[..., :3] * 1000.0

    # check the two are the same within epsilon
    assert np.allclose(robust_points3d_mm, robust_points3d_mm_vmap_mm, atol=1e-5)


def test_compare_real_cam_robust_triangulate_points():
    """
    Test robust_triangulate_points over multiple time frames and joints
    using real camera calibration data.

    We fetch a calibration record, generate synthetic 3D scene points (in mm),
    project them into each camera to obtain 2D observations, run robust
    triangulation, and compare the reconstructed 3D points (converted back to mm)
    to the original scene points.
    """
    from multi_camera.analysis.camera import robust_triangulate_points, robust_triangulate_points_old
    from multi_camera.datajoint.multi_camera_dj import Calibration

    # Fetch a single calibration record.
    camera_params = Calibration.fetch("camera_calibration", limit=1)[0]

    # Determine number of cameras from calibration.
    N = camera_params["mtx"].shape[0]

    # Simulation dimensions.
    T_steps = 10  # number of time frames
    J = 5  # number of joints

    # Generate random 3D scene points (in mm) with shape (T_steps, J, 3).
    point3d_true = np.random.normal(size=(T_steps, J, 3))
    # Convert to meters (since calibrations typically use meters).
    point3d_true_m = point3d_true / 1000.0

    # Allocate an array for 2D observations with shape (num_cameras, T_steps, J, 3).
    points2d = np.zeros((N, T_steps, J, 3))
    for i in range(N):
        # Reconstruct the intrinsic matrix for camera i.
        fx, fy, cx, cy = camera_params["mtx"][i]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        # Extract rotation (rvec) and translation (tvec) for camera i.
        rvec = camera_params["rvec"][i : i + 1]  # shape (1, 3)
        T = camera_params["tvec"][i : i + 1]  # shape (1, 3)
        # Get distortion coefficients.
        dist = camera_params["dist"][i]
        # Loop over time frames and joints.
        for t in range(T_steps):
            for j in range(J):
                # Get the 3D point for this time and joint.
                p = point3d_true_m[t, j, :]  # shape (3,)
                # cv2.projectPoints expects input shape (N,3).
                proj, _ = cv2.projectPoints(p[None, :], rvec, T, K, dist)
                uv = proj[0, 0]  # shape (2,)
                points2d[i, t, j, :2] = uv
                points2d[i, t, j, 2] = 1.0  # full confidence

    # Convert synthetic 2D observations to a JAX array.
    points2d = jnp.array(points2d)

    # Run robust triangulation.
    robust_points3d, weights = robust_triangulate_points(camera_params, points2d, sigma=150, threshold=0.5, return_weights=True)
    robust_points_3d_old, weights_old = robust_triangulate_points_old(camera_params, points2d, sigma=150, threshold=0.5, return_weights=True)

    # compare shapes are the same, and then compare the two results

    assert robust_points3d.shape == robust_points_3d_old.shape, f"Keypoint shape changed. {robust_points3d.shape} != {robust_points_3d_old.shape}"
    assert weights.shape == weights_old.shape, f"Weight shape changed. {weights.shape} != {weights_old.shape}"

    assert np.allclose(robust_points3d, robust_points_3d_old, atol=1e-5)
    assert np.allclose(weights, weights_old, atol=1e-5)
