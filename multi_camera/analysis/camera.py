'''
Tools for a camera model implementation in Jax
'''

from lib2to3 import pytree
import numpy as np
import jax
from jax import numpy as jnp
from jax import vmap, jit
from jaxlie import SO3, SE3


def get_intrinsic(camera_params, i):
    mtx = jnp.take(camera_params['mtx'], i, axis=0)
    return jnp.array([[mtx[0]* 1000.0, 0, mtx[2]* 1000.0],
                      [0, mtx[1]* 1000.0, mtx[3]* 1000.0],
                      [0, 0, 1]])


def get_extrinsic(camera_params, i):
    rvec = jnp.take(camera_params['rvec'], i, axis=0)
    tvec = jnp.take(camera_params['tvec'], i, axis=0) * 1000.0
    rot = SO3.exp(rvec)
    return SE3.from_rotation_and_translation(rot, tvec).as_matrix()


def get_projection(camera_params, i):
    intri = get_intrinsic(camera_params, i)
    extri = get_extrinsic(camera_params, i)
    return intri @ extri[:3]


def project(camera_params, i, points):
    # make sure to use homogeneous coordinates
    if points.shape[-1] == 3:
        points = jnp.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1)

    # last dimension ensures broadcasting works
    proj = get_projection(camera_params, i) @ points[..., None]
    proj = proj[..., 0]

    # remove affine dimension and get u,v coordinates
    return proj[..., :-1] / proj[..., -1:]


@jit
def project_distortion(camera_params, i, points):
    intri = get_intrinsic(camera_params, i)
    extri = get_extrinsic(camera_params, i)

    # make sure to use homogeneous coordinates
    if points.shape[-1] == 3:
        points = jnp.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1)

    # transform the points into the camera perspective
    # last dimension ensures broadcasting works
    transformed = (extri @ points[..., None])[..., 0]

    xp = transformed[..., 0] / transformed[..., 2]
    yp = transformed[..., 1] / transformed[..., 2]
    r2 = xp ** 2 + yp ** 2

    dist = camera_params['dist'][i]
    gamma = 1.0 + dist[0] * r2 + dist[1] * r2**2 + dist[4] * r2**3

    xpp = gamma * xp + 2*dist[2]*xp*yp + dist[3] * (r2 + 2 * xp**2)
    ypp = gamma * yp + dist[2]*(r2 + 2*yp**2) + 2*dist[3]*xp*yp

    points = jnp.stack([xpp, ypp, jnp.ones(xpp.shape)], axis=-1)
    proj = (intri @ points[..., None])[..., 0]
    # remove affine dimension and get u,v coordinates
    return proj[..., :-1] / proj[..., -1:]


@jit
def undistort_points(points: jnp.array, K: jnp.array, dist: jnp.array, num_iters: int = 5) -> jnp.array:
    r"""Compensate for lens distortion a set of 2D image points.

    Radial :math:`(k_1, k_2, k_3, k_4, k_5, k_6)`,
    tangential :math:`(p_1, p_2)`, thin prism :math:`(s_1, s_2, s_3, s_4)`, and tilt :math:`(\tau_x, \tau_y)`
    distortion models are considered in this function.

    Args:
        points: Input image points with shape :math:`(*, N, 2)`.
        K: Intrinsic camera matrix with shape :math:`(*, 3, 3)`.
        dist: Distortion coefficients
            :math:`(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]])`. This is
            a vector with 4, 5, 8, 12 or 14 elements with shape :math:`(*, n)`.
        num_iters: Number of undistortion iterations. Default: 5.
    Returns:
        Undistorted 2D points with shape :math:`(*, N, 2)`.

    Example:
        >>> x = np.rand(1, 4, 2)
        >>> K = np.eye(3)[None]
        >>> dist = torch.rand(1, 4)
        >>> undistort_points(x, K, dist)
        tensor([[[-0.1513, -0.1165],
                 [ 0.0711,  0.1100],
                 [-0.0697,  0.0228],
                 [-0.1843, -0.1606]]])

    Refs:
        Based on https://github.com/opencv/opencv/blob/master/modules/calib3d/src/undistort.dispatch.cpp#L384
        and https://kornia.readthedocs.io/en/latest/_modules/kornia/geometry/calibration/undistort.html#undistort_points

    """

    # Adding zeros to obtain vector with 14 coeffs.
    if dist.shape[-1] < 14:
        pad_dist = jnp.zeros((*dist.shape[:-1], 14))
        pad_dist = pad_dist.at[..., :dist.shape[-1]].set(dist)
        dist = pad_dist

    # Convert 2D points from pixels to normalized camera coordinates
    cx = K[..., 0:1, 2]  # princial point in x (Bx1)
    cy = K[..., 1:2, 2]  # princial point in y (Bx1)
    fx = K[..., 0:1, 0]  # focal in x (Bx1)
    fy = K[..., 1:2, 1]  # focal in y (Bx1)

    # This is equivalent to K^-1 [u,v,1]^T
    x = (points[..., 0] - cx) / fx  # (BxN - Bx1)/Bx1 -> BxN
    y = (points[..., 1] - cy) / fy  # (BxN - Bx1)/Bx1 -> BxN

    if False:
        # Compensate for tilt distortion
        if torch.any(dist[..., 12] != 0) or torch.any(dist[..., 13] != 0):
            inv_tilt = tilt_projection(dist[..., 12], dist[..., 13], True)

            # Transposed untilt points (instead of [x,y,1]^T, we obtain [x,y,1])
            x, y = transform_points(inv_tilt, torch.stack([x, y], dim=-1)).unbind(-1)

    # Iteratively undistort points
    x0, y0 = x, y
    for _ in range(num_iters):
        r2 = x * x + y * y

        inv_rad_poly = (1 + dist[..., 5:6] * r2 + dist[..., 6:7] * r2 * r2 + dist[..., 7:8] * r2**3) / (
            1 + dist[..., 0:1] * r2 + dist[..., 1:2] * r2 * r2 + dist[..., 4:5] * r2**3
        )
        deltaX = (
            2 * dist[..., 2:3] * x * y
            + dist[..., 3:4] * (r2 + 2 * x * x)
            + dist[..., 8:9] * r2
            + dist[..., 9:10] * r2 * r2
        )
        deltaY = (
            dist[..., 2:3] * (r2 + 2 * y * y)
            + 2 * dist[..., 3:4] * x * y
            + dist[..., 10:11] * r2
            + dist[..., 11:12] * r2 * r2
        )

        x = (x0 - deltaX) * inv_rad_poly
        y = (y0 - deltaY) * inv_rad_poly

    # Convert points from normalized camera coordinates to pixel coordinates
    new_cx = K[..., 0:1, 2]  # princial point in x (Bx1)
    new_cy = K[..., 1:2, 2]  # princial point in y (Bx1)
    new_fx = K[..., 0:1, 0]  # focal in x (Bx1)
    new_fy = K[..., 1:2, 1]  # focal in y (Bx1)
    x = new_fx * x + new_cx
    y = new_fy * y + new_cy

    return jnp.stack([x, y], -1)


@jit
def triangulate_point(camera_params, points2d):
    r"""Triangulate multiple 2D observations in 3D

    Parameters:
        camera_params: pytree of the camera calibrations
        points2d: the observations, which can optionally have a confidence
    Returns:
        3d point locations

    Refs:
        - https://amytabb.com/tips/tutorials/2021/10/31/triangulation-DLT-2-3/
    """

    projections = jnp.array([get_projection(camera_params, i) for i in range(8)])

    assert points2d.shape[0] == projections.shape[0]

    if points2d.shape[-1] == 3:
        weight = points2d[..., -1:]
        points2d = points2d[..., :-1]
    else:
        weight = np.ones((*points2d.shape[:-1], 1))

    points2d = jnp.array([undistort_points(points2d[i], get_intrinsic(camera_params, i), camera_params['dist'][i]) for i in range(8)])

    # use broadcasting to build a DLT matrix. this will have a shape
    # number of cameras X 2 x (point dimensions) x 4 and then gets reshaped
    # to number of camers * 2 long. the code below vmaps over the point dimensions
    projections_shape = (points2d.shape[0], * [1] * (len(points2d.shape) - 2), 3, 4)
    projections = projections.reshape(projections_shape)

    A = projections[..., -1:, :] * points2d[..., None] - projections[..., :-1, :]
    A = A * weight[..., None]

    def _triangulate_A(A):
        assert len(A.shape) == 3

        if True:
            # one approach just keep all of them, but zero nans
            A = jnp.nan_to_num(A, nan=0)
        else:
            valid = ~np.isnan(A[:, 0, 0]) # only check x coordinate
            A = A[valid]

        A = A.reshape((A.shape[0] * A.shape[1], *A.shape[2:]))
        _, _, vh = jnp.linalg.svd(A, full_matrices=False)
        p3d = vh[-1]
        p3d = p3d[:3] / p3d[3]
        return p3d

    if len(A.shape) == 3:
        return _triangulate_A(A)

    elif len(A.shape) == 4:
        return vmap(_triangulate_A, in_axes=1, out_axes=0)(A)

    elif len(A.shape) == 5:
        return vmap(vmap(_triangulate_A, in_axes=1, out_axes=0), in_axes=2, out_axes=1)(A)

    else:
        raise Exception("Unsupported shape")


def reprojection_error(camera_params, points2d, points3d):
    N = camera_params['mtx'].shape[0]
    project_cameras = vmap(project_distortion, in_axes=(None, 0, None), out_axes=(0))
    projected_points = project_cameras(camera_params, jnp.arange(N), points3d)
    return points2d - projected_points


def reconstruction_error(camera_params, points2d, points3d, stop_grad=False):
    est_points3d = triangulate_point(camera_params, points2d)
    if stop_grad:
        est_points3d = jax.lax.stop_gradient(est_points3d)
    return est_points3d - points3d


def get_checkboard_3d(rvec, tvec, objp):
    transform = SE3.from_rotation_and_translation(SO3.exp(rvec), tvec * 1000.0)
    return vmap(transform.apply)(objp)