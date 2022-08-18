import numpy as np
from scipy import interpolate
from dataclasses import dataclass


model_path = '/home/jcotton/projects/pose/EasyMocap/data/smplx'


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
        opts = {}

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