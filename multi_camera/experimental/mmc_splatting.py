"""
Code to use the Gaussian Splatting code and apply it to data from MMC

Expects the gaussian-splatting code to be in the path when these
are called
"""

import os
import cv2
import numpy as np
import torch


def create_uniform_gaussians(num_points: int, scale: float = 1.0, sh_degree: int = 3):
    """
    Create a set of uniform gaussians

    Args:
        num_points: Number of points to create
        scale: Scale of the points
        sh_degree: Degree of the SH functions

    Returns:
        gaussians: GaussianModel object
    """

    # https://github.com/graphdeco-inria/gaussian-splatting/blob/414b553ef1d3032ea634b7b7a1d121173f36592c/scene/dataset_readers.py#L237

    from scene import Scene, GaussianModel
    from scene.gaussian_model import BasicPointCloud
    from utils.sh_utils import SH2RGB

    # We create random points inside the bounds of the synthetic Blender scenes
    xyz = np.random.random((num_points, 3)) * (2 * scale) - scale
    shs = np.random.random((num_points, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_points, 3)))

    gaussians = GaussianModel(3)
    gaussians.create_from_pcd(pcd, spatial_lr_scale=0.0)

    return gaussians


def get_camera(
    mtx,
    rvec,
    tvec,
    camera_idx,
    znear=1.0,
    zfar=100.0,
    scale=1.0,
    trans=np.array([0.0, 0.0, 0.0]),
    height=1536,
    width=2048,
):
    """
    Get a camera object from the camera parameters

    Args:
        mtx: Camera intrinsic matrix
        rvec: Rotation vector
        tvec: Translation vector
        camera_idx: Camera index

    Returns:
        camera: Camera object
    """

    from utils.graphics_utils import getWorld2View, getWorld2View2, getProjectionMatrix
    from scene.cameras import MiniCam

    cam_mtx = mtx[camera_idx]  # intrinsic matrix for camera. K[0,0], K[1,1], K[0,2], K[1,2]

    # compute FoVx and FoVy based on the width and height
    FoVx = 2 * np.arctan(width / (2 * cam_mtx[0]))
    FoVy = 2 * np.arctan(height / (2 * cam_mtx[1]))

    # convert rvec to rotation matrix
    R = cv2.Rodrigues(rvec[None, camera_idx])[0]  # .T
    T = tvec[camera_idx]

    # OpenCV has translation as the world to camera transform, so we need to invert it
    # T = -R.T @ T
    # print(T)

    # their implementation has transposes but I think our OpenCV parameters don't need this
    # as the representation is already about the camera frame
    # world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()

    world_view_transform = torch.tensor(getWorld2View2(R.T, T, trans, scale)).transpose(0, 1).cuda()
    # world_view_transform = torch.tensor(getWorld2View(R, T)).transpose(0, 1).cuda()

    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0, 1).cuda()

    # width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform
    camera = MiniCam(width, height, FoVx, FoVy, znear, zfar, world_view_transform, projection_matrix)
    # print(
    #     camera.FoVx,
    #     camera.FoVy,
    #     camera.camera_center,
    #     camera.full_proj_transform,
    #     camera.world_view_transform,
    #     camera.zfar,
    #     camera.znear,
    # )
    return camera


def get_cameras(key: dict, *args, **kwargs):
    """
    Get the camera parameters for a given key

    Args:
        key: Key for the calibration
        scale: Scale of the points (default: 1.0, 1000.0 for mm)
    """

    from multi_camera.datajoint.multi_camera_dj import Calibration

    cal, camera_names = (Calibration & key).fetch1("camera_calibration", "camera_names")

    print("Camera names: ", camera_names)

    # dist = cal["dist"]
    mtx = cal["mtx"] * 1000.0
    rvec = cal["rvec"]
    tvec = cal["tvec"]
    # print(tvec)

    num_cameras = len(mtx)
    cameras = [get_camera(mtx, rvec, tvec, i, *args, **kwargs) for i in range(num_cameras)]
    return cameras


def get_images(key: dict, frame_num: int):
    """
    Get the camera parameters for a given key

    Args:
        key: Key for the calibration
        frame_num: Frame number to get the images for
    """

    from pose_pipeline.pipeline import Video
    from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording, SingleCameraVideo

    assert len(MultiCameraRecording & key) == 1, "Key must be unique"

    videos, camera_names = (Video * SingleCameraVideo & key).fetch("video", "camera_name")
    print("Camera names: ", camera_names)

    # use cv2 to read the specific frame from each camera
    images = []
    for v in videos:
        cap = cv2.VideoCapture(v)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(frame)
        cap.release()
        os.remove(v)

    return images


class Pipeline:
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False


class OptimizationParams:
    iterations = 30_000
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30_000
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = 0.005
    rotation_lr = 0.001
    percent_dense = 0.01
    lambda_dssim = 0.2
    densification_interval = 100
    opacity_reset_interval = 3000
    densify_from_iter = 500
    densify_until_iter = 15_000
    densify_grad_threshold = 0.0002


def optimize_key(key: dict, frame: int, iterations: int = 2_000, num_points: int = 1000):
    import torch
    from random import randint
    from utils.loss_utils import l1_loss, ssim
    from gaussian_renderer import render

    # TODO: handle distoritions
    images = get_images(key, frame)
    cameras = get_cameras(key, scale=1.0, zfar=100.0)

    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  # white background

    pipe = Pipeline()
    opt = OptimizationParams()

    gaussians = create_uniform_gaussians(num_points, 1)
    gaussians.training_setup(opt)

    train_idx = []

    for iteration in range(1, iterations):
        if not train_idx:
            train_idx = list(range(len(images) - 1))

        view_idx = train_idx.pop(randint(0, len(train_idx) - 1))

        render_pkg = render(cameras[view_idx], gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        gt_image = torch.tensor(images[view_idx]).cuda()

        # TODO: move this to earlier
        gt_image = gt_image.permute(2, 0, 1).float() / 255.0

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        print(iteration, view_idx, Ll1.item())

        with torch.no_grad():
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    camera_extent = 0.1
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, camera_extent, size_threshold)

                if (
                    iteration % opt.opacity_reset_interval == 0
                ):  # or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

    return gaussians, cameras, images
