import numpy as np
import pandas as pd
from einops import rearrange
from multi_camera.datajoint.gaitrite_comparison import (
    GaitRiteRecordingAlignment,
    GaitRiteCalibration,
    GaitRiteRecording,
    fetch_data,
)
from multi_camera.validation.biomechanics.biomechanics import BiomechanicalReconstruction


def get_average_steps(key):
    t_offset = (GaitRiteRecordingAlignment & key).fetch1("t_offset")
    R, t = (GaitRiteCalibration & key).fetch1("r", "t")

    dt, kp3d, df = fetch_data(key)

    # account for the calibrated offset
    df[["First Contact Time", "Last Contact Time"]] += t_offset

    # make sure dimensions are correct and account for room calibration
    if kp3d.shape[-1] == 4:
        conf = kp3d[:, :, 3]
    else:
        conf = np.ones_like(kp3d[:, :, 0])
    if R.shape[0] == 3:
        kp3d = kp3d[:, :, :3] @ R + t
    else:
        kp3d = kp3d[:, :, :2] @ R + t

    def extract_steps(idx):
        # extract the trace for the idx-th step and interpolate to 101 points

        t0 = df.loc[idx, "First Contact Time"]
        te = df.loc[idx + 2, "First Contact Time"]

        # index 1 is left big toe and 3 is right big toe
        trace = kp3d[(dt >= t0) & (dt <= te)][:, [1, 3], 2]  # returns time x 4 x 3

        # create fake time that goes from 0 to 1
        t = np.linspace(0, 1, trace.shape[0])

        # interpolate to 101 points along the first dimension
        t_interp = np.linspace(0, 1, 101)
        trace = np.stack([np.interp(t_interp, t, trace[:, i]) for i in range(trace.shape[1])], axis=1)

        # trace = rearrange(trace_interp, "t (c d) -> t c d", c=4)  # returns time x 4 x 3

        # now detrend the data by removing a line that goes from the first and last points
        # using linear regression. first compute the baseline prediction
        h0 = np.mean(trace[:10], axis=0)
        he = np.mean(trace[-10:], axis=0)
        baseline = h0 + t_interp[:, None] * (he - h0)
        trace = trace - baseline

        return trace

    # run extract_steps for all teh rows of df where the foot is left
    left_idx = np.where(df["Left Foot"])[0][:-1]
    left_steps = np.array([extract_steps(idx) for idx in left_idx])
    right_idx = np.where(~df["Left Foot"])[0][:-1]
    right_steps = np.array([extract_steps(idx) for idx in right_idx])

    return {"left_steps": left_steps, "right_steps": right_steps}


default_biomechanics_model = {
    "model_name": "Rajagopal_Neck_mbl_movi_87_rev15",
    "reconstruction_method_name": "Implicit Optimization KP Conf, MaxHuber=10",
    "bilevel_settings": 19,
}


def get_gait_cycle_aligned_pose(key, model=default_biomechanics_model):
    t_offset = (GaitRiteRecordingAlignment & key).fetch1("t_offset")

    df = pd.DataFrame((GaitRiteRecording & key).fetch1("gaitrite_dataframe"))
    dt, poses = (BiomechanicalReconstruction.Trial & key & model).fetch1("timestamps", "poses")

    # account for the calibrated offset
    df[["First Contact Time", "Last Contact Time"]] += t_offset

    # only keep valid rows where both First Contact Time and Last Contact Time are in the dt range
    df = df[(df["First Contact Time"] >= dt[0]) & (df["Last Contact Time"] <= dt[-1])]

    def extract_steps(idx):
        # extract the trace for the idx-th step and interpolate to 101 points

        t0 = df.loc[idx, "First Contact Time"]
        te = df.loc[idx + 2, "First Contact Time"]

        # index 1 is left big toe and 3 is right big toe
        trace = poses[(dt >= t0) & (dt <= te)]  # returns time x joints
        trace = trace[:, [6, 13, 9, 16, 10, 17]]  # discard root node position and orientation)

        # create fake time that goes from 0 to 1
        t = np.linspace(0, 1, trace.shape[0])

        # interpolate to 101 points along the first dimension
        t_interp = np.linspace(0, 1, 101)
        trace = np.stack([np.interp(t_interp, t, trace[:, i]) for i in range(trace.shape[1])], axis=1)

        return trace

    # run extract_steps for all teh rows of df where the foot is left
    left_idx = np.where(df["Left Foot"])[0][:-1]
    left_steps = np.array([extract_steps(idx) for idx in left_idx])
    right_idx = np.where(~df["Left Foot"])[0][:-1]
    right_steps = np.array([extract_steps(idx) for idx in right_idx])

    return {"left_steps": left_steps, "right_steps": right_steps}
