import os
import numpy as np
import tensorflow as tf


lower_body_joints = [
    "Neck",
    "Right Shoulder",
    "Left Shoulder",
    "Right Hip",
    "Left Hip",
    "Right Knee",
    "Left Knee",
    "Right Ankle",
    "Left Ankle",
    "Right Heel",
    "Left Heel",
    "Right Little Toe",
    "Left Little Toe",
    "Right Big Toe",
    "Left Big Toe",
]

upper_body_joints = [
    "Neck",
    "Right Shoulder",
    "Left Shoulder",
    "Right Elbow",
    "Left Elbow",
    "Right Wrist",
    "Left Wrist",
]

lower_body_markers = [
    "C7_study",
    "r_shoulder_study",
    "L_shoulder_study",
    "r.ASIS_study",
    "L.ASIS_study",
    "r.PSIS_study",
    "L.PSIS_study",
    "r_knee_study",
    "L_knee_study",
    "r_mknee_study",
    "L_mknee_study",
    "r_ankle_study",
    "L_ankle_study",
    "r_mankle_study",
    "L_mankle_study",
    "r_calc_study",
    "L_calc_study",
    "r_toe_study",
    "L_toe_study",
    "r_5meta_study",
    "L_5meta_study",
    "r_thigh1_study",
    "r_thigh2_study",
    "r_thigh3_study",
    "L_thigh1_study",
    "L_thigh2_study",
    "L_thigh3_study",
    "r_sh1_study",
    "r_sh2_study",
    "r_sh3_study",
    "L_sh1_study",
    "L_sh2_study",
    "L_sh3_study",
    "RHJC_study",
    "LHJC_study",
]

upper_body_markers = [
    "r_lelbow_study",
    "L_lelbow_study",
    "r_melbow_study",
    "L_melbow_study",
    "r_lwrist_study",
    "L_lwrist_study",
    "r_mwrist_study",
    "L_mwrist_study",
]


def get_model(model_path):
    with open(os.path.join(model_path, "model.json"), "r") as f:
        model = tf.keras.models.model_from_json(f.read())
    model.load_weights(os.path.join(model_path, "weights.h5"))

    train_features_mean = np.load(os.path.join(model_path, "mean.npy"), allow_pickle=True)
    train_features_std = np.load(os.path.join(model_path, "std.npy"), allow_pickle=True)

    return {"model": model, "features_mean": train_features_mean, "features_std": train_features_std}


def get_normalized_joints(kp3d, joint_names, keep_joints, height, weight, model):

    joint_idx = [joint_names.index(j) for j in keep_joints]
    delta_joints = kp3d[:, joint_idx] - kp3d[:, joint_names.index("Pelvis"), None]
    delta_joints = delta_joints[:, :, :3] / height  # drop confidence and scale by height

    # flatten keypoints after reordering axes
    delta_joints = np.take(delta_joints, [0, 1, 2], axis=-1)
    delta_joints = delta_joints.reshape([delta_joints.shape[0], -1])

    # and add height and weight as the last two columns
    delta_joints = np.concatenate([delta_joints, height * np.ones((delta_joints.shape[0], 1))], axis=1)
    delta_joints = np.concatenate([delta_joints, weight * np.ones((delta_joints.shape[0], 1))], axis=1)

    delta_joints = delta_joints.reshape(-1, model["features_mean"].shape[0])
    delta_joints = (delta_joints - model["features_mean"]) / model["features_std"]

    # add addition dimension for batch size
    return delta_joints[None, ...]


def predict_and_denormalize(kp3d, joint_names, keep_joints, height, weight, model):
    delta_joints = get_normalized_joints(kp3d, joint_names, keep_joints, height, weight, model)
    pred = model["model"].predict(delta_joints)[0]
    pred = pred * height

    pred = pred.reshape(pred.shape[0], -1, 3) + kp3d[:, joint_names.index("Pelvis"), None, :3]

    return pred


def convert_markers(key, trange=None):
    from ...datajoint.multi_camera_dj import MultiCameraRecording, PersonKeypointReconstruction, SingleCameraVideo
    from pose_pipeline import TopDownPerson, TopDownMethodLookup

    top_down_method_name = (TopDownMethodLookup * SingleCameraVideo & key).fetch("top_down_method_name", limit=1)[0]
    print("top_down_method_name", top_down_method_name)
    joints = TopDownPerson.joint_names(top_down_method_name)

    if top_down_method_name == "OpenPose":
        # replace "Sternum" in joints list with "Neck"
        joints[joints.index("Sternum")] = "Neck"

    # get path to current file
    path = os.path.dirname(os.path.abspath(__file__))
    augmenter_model_dir = os.path.join(path, "OpenCap_LSTM/")

    lower_model = get_model(os.path.join(augmenter_model_dir, "v0.2_lower"))
    upper_model = get_model(os.path.join(augmenter_model_dir, "v0.2_upper"))

    kp3d = (PersonKeypointReconstruction & key).fetch1("keypoints3d")
    kp3d = kp3d / 1000.0  # convert to meters

    kp3d = np.take(kp3d, [1, 2, 0], axis=-1)  # convert to OpenSim convention

    if trange is not None:
        timestamps = (MultiCameraRecording & key).fetch_timestamps()
        kp3d = kp3d[np.logical_and(timestamps >= trange[0], timestamps <= trange[1])]

    if "Head" in joints:
        height = kp3d[:, joints.index("Head")] - kp3d[:, joints.index("Right Heel")]
    else:
        height = kp3d[:, joints.index("Nose")] - kp3d[:, joints.index("Right Heel")]

    height = np.median(np.linalg.norm(height[:, :3], axis=-1))
    print(f"height: {height}")

    lower_markers = predict_and_denormalize(kp3d, joints, lower_body_joints, height, 60, lower_model)
    upper_markers = predict_and_denormalize(kp3d, joints, upper_body_joints, height, 60, upper_model)

    markers = np.concatenate([lower_markers, upper_markers], axis=1)
    marker_names = lower_body_markers + upper_body_markers
    return markers, marker_names


if __name__ == "__main__":

    import datetime
    from pose_pipeline import VideoInfo
    from multi_camera.datajoint.multi_camera_dj import (
        PersonKeypointReconstruction,
        SingleCameraVideo,
        MultiCameraRecording,
    )
    from multi_camera.analysis.biomechanics import bilevel_optimization
    from multi_camera.analysis.biomechanics import opencap_augmenter
    from multi_camera.datajoint.gaitrite_comparison import get_walking_time_range

    key = (
        PersonKeypointReconstruction * MultiCameraRecording
        & 'video_base_filename LIKE "p104_GaitRite_20221208_101420%" and reconstruction_method=0 and top_down_method=4'
    ).fetch1("KEY")

    trange = get_walking_time_range({**key, "top_down_method": 2, "reconstruction_method": 0})
    markers, marker_names = opencap_augmenter.convert_markers(key, trange)

    def map_frame(kp3d):
        return {j: k for j, k in zip(marker_names, kp3d)}

    kp3d = [map_frame(k) for k in markers]

    bilevel_optimization.fit_markers([kp3d], model_name="Rajagopal2015_Augmenter")
