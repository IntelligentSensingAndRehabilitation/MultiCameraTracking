import numpy as np
from scipy import interpolate
from dataclasses import dataclass


model_path = '/home/jcotton/projects/pose/EasyMocap/data/smplx'


def mvmp_association_and_tracking(dataset, keypoints='body25'):
    """
    Associate and track people across views and time

    This uses the EasyMocap interface and algorithms, and is largely
    designed to run the steps described at
        https://github.com/zju3dv/EasyMocap/blob/master/doc/mvmp.md

    The modifications are primarily to directly run this without requiring
    all the filesystem intermediate steps. It should work with any interface
    compatible with easymocap.dataset.mvmpmf.MVMPMF but specifically
    has been tested with our wrapper to MultiCamera datajoint.

    Params:
        dataset (MVMPMF) : dataset containing the keypoints and camera calibration
        keypoints (String) : specify the expected keypoint set. 'body25' for OpenPose

    Returns a list of dictionaries for each frame containing the ids and 3d keypoints
    """

    from tqdm import trange
    from easymocap.dataset.base import MVBase
    from easymocap.config.mvmp1f import Config
    from easymocap.assignment.group import PeopleGroup
    from easymocap.assignment.associate import simple_associate
    from easymocap.affinity.affinity import ComposedAffinity
    from easymocap.assignment.track import Track3D
    # currently need to remove some of the additional criteria for association. likely
    # related to scale or keypoint ordering that makes them appear un-anatomic.
    cfg = Config.load('/home/jcotton/projects/pose/EasyMocap/config/exp/mvmp1f_me.yml')

    affinity_model = ComposedAffinity(cameras=dataset.cameras, basenames=dataset.cams, cfg=cfg.affinity)
    group = PeopleGroup(Pall=dataset.Pall, cfg=cfg.group)
    tracker = Track3D(with2d=False, WINDOW_SIZE=10, MIN_FRAMES=10, SMOOTH_SIZE=5, out=None, path=None)

    results = []
    for i in trange(len(dataset), desc='Associating'):
        group.clear()
        images, annots = dataset[i]
        affinity, dimGroups = affinity_model(annots, images=images)
        group = simple_associate(annots, affinity, dimGroups, dataset.Pall, group, cfg=cfg.associate)

        results.append([{'id': k, 'keypoints3d': v.keypoints3d, 'bbox': v.bbox, 'height': v.height} for k, v in group.items()])

    edges = tracker.compute_dist(results)
    results = tracker.associate(results, edges)
    results, occupancy = tracker.reset_id(results)
    results, occupancy = tracker.smooth(results, occupancy)

    return results


def interpolate_points(points3d, method='cubic'):
    """
    Replace any NaN values with interpolated ones
    """
    def interpolate_nan_col(col):

        idx = np.arange(len(col))
        idx_good = np.where(np.isfinite(col))[0] #index of non zeros
        if len(idx_good) <= 10: return col

        f_interp = interpolate.interp1d(idx_good, col[idx_good], kind=method, bounds_error=False)
        col_interp = np.where(np.isfinite(col), col, f_interp(idx)) #replace nans with interpolated values
        col_interp = np.where(np.isfinite(col_interp), col_interp, np.nanmean(col_interp)) #replace remaining nans

        return col_interp

    points3d = points3d.copy()
    for i in range(points3d.shape[1]):
        for j in range(points3d.shape[2]):
            points3d[:, i, j] = interpolate_nan_col(points3d[:, i, j])
    return points3d


def easymocap_fit_smpl_3d(joints3d, model_path=model_path, verbose=True, smooth3d=True, robust3d=True):

    from easymocap.pipeline import smpl_from_keypoints3d
    from easymocap.dataset import CONFIG
    from easymocap.smplmodel.body_param import load_model

    body_model = load_model(model_path=model_path)

    @dataclass
    class EasymocapArgs:
        model = 'smpl'
        gender = 'neutral'
        opts = {'smooth_poses': 1e-4, 'reg_poses': 1e-5, 'smooth_body': 5e-3}

    args = EasymocapArgs
    args.verbose = verbose
    args.robust3d = robust3d
    args.smooth3d = smooth3d
    config = CONFIG

    def add_dim(p):
        # not quite sure why this extra dimension is used in. perhaps could be visible flag
        # https://github.com/zju3dv/EasyMocap/blob/584ba2c1e85c626e90bbcfa6931faf8998c5ba84/easymocap/pyfitting/optimize_simple.py#L33

        return np.concatenate([p, np.ones((p.shape[0], p.shape[1], 1))], axis=-1)

    joints3d = interpolate_points(joints3d)
    res = smpl_from_keypoints3d(body_model, add_dim(joints3d), config, args)
    return res


def get_joint_openpose(res, model_path=model_path):
    from easymocap.smplmodel.body_param import load_model

    body_model = load_model(model_path=model_path)
    return body_model(return_verts=False, return_joints=True, return_tensor=False, **res)


def get_vertices(res, model_path=model_path):
    from easymocap.smplmodel.body_param import load_model

    body_model = load_model(model_path=model_path)
    return body_model(return_verts=True, return_tensor=False, **res)

def get_faces():
    from easymocap.smplmodel.body_param import load_model
    body_model = load_model(model_path=model_path)
    return body_model.faces

