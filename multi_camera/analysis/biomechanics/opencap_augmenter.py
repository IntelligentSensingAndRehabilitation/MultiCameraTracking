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


def get_normalized_joints(kp3d, joint_names, height, weight, model):
    from pose_pipeline import TopDownPerson

    joints = TopDownPerson.joint_names("MMPoseHalpe")

    joint_idx = [joints.index(j) for j in joint_names]
    delta_joints = kp3d[:, joint_idx] - kp3d[:, joints.index("Pelvis"), None]
    delta_joints = delta_joints[:, :, :3] / height  # drop confidence and scale by height

    # now flatten delta_joints and add height and weight as the last two columns
    delta_joints = delta_joints.reshape(-1, delta_joints.shape[1] * delta_joints.shape[2])
    delta_joints = np.concatenate([delta_joints, height * np.ones((delta_joints.shape[0], 1))], axis=1)
    delta_joints = np.concatenate([delta_joints, weight * np.ones((delta_joints.shape[0], 1))], axis=1)

    delta_joints = delta_joints.reshape(-1, model["features_mean"].shape[0])
    delta_joints = (delta_joints - model["features_mean"]) / model["features_std"]

    # add addition dimension for batch size
    return delta_joints[None, ...]


def predict_and_denormalize(kp3d, joint_names, height, weight, model):
    from pose_pipeline import TopDownPerson

    joints = TopDownPerson.joint_names("MMPoseHalpe")

    delta_joints = get_normalized_joints(kp3d, joint_names, height, weight, model)
    pred = model["model"].predict(delta_joints)[0]
    pred = pred * height

    # note the scale of 1000 as we work in mm
    pred = pred.reshape(pred.shape[0], -1, 3) * 1000.0 + kp3d[:, joints.index("Pelvis"), None, :3]

    return pred


def convert_markers(key):
    from ...datajoint.multi_camera_dj import PersonKeypointReconstruction
    from pose_pipeline import TopDownPerson

    joints = TopDownPerson.joint_names("MMPoseHalpe")

    # get path to current file
    path = os.path.dirname(os.path.abspath(__file__))
    augmenter_model_dir = os.path.join(path, "OpenCap_LSTM/")

    lower_model = get_model(os.path.join(augmenter_model_dir, "v0.2_lower"))
    upper_model = get_model(os.path.join(augmenter_model_dir, "v0.2_upper"))

    kp3d = (PersonKeypointReconstruction & key).fetch1("keypoints3d")

    height = kp3d[:, joints.index("Head")] - kp3d[:, joints.index("Right Heel")]
    height = np.median(np.linalg.norm(height[:, :3], axis=-1)) / 1000.0

    lower_markers = predict_and_denormalize(kp3d, lower_body_joints, height, 60, lower_model)
    upper_markers = predict_and_denormalize(kp3d, upper_body_joints, height, 60, upper_model)

    markers = np.concatenate([lower_markers, upper_markers], axis=1)
    marker_names = lower_body_markers + upper_body_markers
    return markers, marker_names
