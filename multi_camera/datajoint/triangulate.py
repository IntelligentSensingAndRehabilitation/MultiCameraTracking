import numpy as np

def triangulate_with_occlusion(p : np.array, cgroup, threshold=0.25) -> np.array:
    '''
    Triangulate a single point in an occlusion aware manner

        Parameters:
            p (np.array) : C x 3 array of 2D keypoints from C cameras
            cgroup : calibrated camera group

        Returns:
            3 vector
    '''

    visible = p[:, 2] > threshold
    idx = list(np.where(visible)[0])
    if len(idx) == 0:
        return np.nan * np.ones((3))

    cgroup_subset = cgroup.subset_cameras(idx)
    return cgroup_subset.triangulate(p[np.array(idx), None, :2])[0]


def reconstruct(recording_key, calibration_key, top_down_method=0):
    from tqdm import tqdm
    from einops import rearrange
    from .calibrate_cameras import Calibration
    from .multi_camera_dj import MultiCameraRecording, SingleCameraVideo
    from pose_pipeline import TopDownPerson
    from aniposelib.cameras import CameraGroup

    camera_calibration = (Calibration & calibration_key).fetch1('camera_calibration')
    cgroup = CameraGroup.from_dicts(camera_calibration)

    keypoints, camera_name = (TopDownPerson * SingleCameraVideo * MultiCameraRecording &
                              recording_key & {'top_down_method': top_down_method}).fetch('keypoints', 'camera_name')

    # if one video has one less frames then crop it
    N = min([k.shape[0] for k in keypoints])
    assert all([k.shape[0] - N < 2 for k in keypoints])
    keypoints = [k[:N] for k in keypoints]

    # work out the order that matches the calibration
    order = [list(camera_name).index(c) for c in cgroup.get_names()]

    points2d = np.stack([keypoints[o][:, :, :] for o in order], axis=1)

    # collapse joints and time onto one axis
    points2d_flat = rearrange(points2d, 'n c j i -> (n j) c i')

    points3d_flat = np.array([triangulate_with_occlusion(p, cgroup) for p in points2d_flat])

    points3d = rearrange(points3d_flat, '(n j) i -> n j i', j=17)

    return points3d