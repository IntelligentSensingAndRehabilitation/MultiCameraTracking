
def reconstruct(recording_key, calibration_key, top_down_method=0):
    import numpy as np
    from pose_pipeline import TopDownPerson
    from aniposelib.cameras import CameraGroup
    from .calibrate_cameras import Calibration
    from .multi_camera_dj import MultiCameraRecording, SingleCameraVideo
    from ..analysis.reconstruction import reconstruct as reconstruct_lib

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

    return reconstruct_lib(points2d, cgroup)