import jax
from jax import numpy as jnp
from .camera import project_distortion


def reprojection_quality(keypoints3d, camera_params, keypoints2d, per_joint_metrics=False):
    if keypoints3d.shape[-1] == 4:
        keypoints3d = keypoints3d[..., :3]

    keypoints2d_proj = jnp.array([project_distortion(camera_params, i, keypoints3d) for i in range(camera_params["mtx"].shape[0])])

    projection_error = jnp.linalg.norm(keypoints2d_proj - keypoints2d[..., :2], axis=-1)
    keypoint_conf = keypoints2d[..., 2]

    axis = (0, 1) if per_joint_metrics else None

    def pck(x, c):
        return jnp.sum(jnp.logical_and(projection_error < x, keypoint_conf >= c), axis=axis) / jnp.sum(keypoint_conf >= c, axis=axis)

    thresh = jnp.linspace(0, 200.0, 200)
    confidence = jnp.linspace(0, 0.9, 10)

    perf = lambda x: jnp.array([pck(x, c) for c in confidence])

    metrics = jax.vmap(perf)(thresh)

    return metrics, thresh, confidence
