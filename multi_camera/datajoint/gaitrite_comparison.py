import os
import datajoint as dj
import numpy as np
import pandas as pd

from typing import List

from pose_pipeline import PersonBbox, VideoInfo, TopDownPerson, TopDownMethodLookup
from .multi_camera_dj import MultiCameraRecording, SingleCameraVideo, PersonKeypointReconstruction
from ..analysis.gaitrite_comparison import parse_gaitrite, extract_traces, find_best_alignment, get_offset_range

schema = dj.schema("multicamera_tracking_gaitrite")

# Schema organization for GaitRite sessions. Currently the multi camera system has no
# session structure either, which would likely help for this. However, might as well
# work on it as is.
#
# We can define a GaitRite session (manual) with a primary key of the subject and
# date. Then desending from that can be a table containing each GaitRite data frame
# that also inherits from the corresponding MultiCameraRecording.
#
# Right now, I'm choosing not to inherit from the Subjects table for the PosePipe
# analysis framework just to minimize interdependecy. I'll use the same field names
# though to allow performing a join.

# Timestamps: a big issue is synchronizing the video and GaitRite data. The GaitRite file
# has a t0 (with only 1 second resolution) and then has relative times in the dataframe.
# The video timestamps are in absolute time. Consistent with the other analyses, we use
# the relative video timestamps as the local timebase. To account for this, we apply this
# relative offset to the timestamps in the GaitRite dataframes. However, this is still only
# a crude approximation prior to further refinement. The t_offset we compute as a refinement
# is added to the GaitRite event times before extracting data from the traces.


@schema
class GaitRiteSession(dj.Manual):
    definition = """
    subject_id: int
    gaitrite_sesion_date: timestamp
    ---
    """


@schema
class GaitRiteRecording(dj.Manual):
    definition = """
    -> GaitRiteSession
    -> MultiCameraRecording
    ---
    gaitrite_filename: varchar(255)
    gaitrite_dataframe: longblob
    gaitrite_t0 : timestamp
    """


@schema
class GaitRiteCalibration(dj.Computed):
    definition = """
    -> GaitRiteSession
    ---
    r: longblob            # rotation matrix
    t: longblob            # translation vector
    t_offsets : longblob   # time offsets
    score : float          # score of the best alignment
    """

    def make(self, key):
        recording_keys = (
            GaitRiteRecording * PersonKeypointReconstruction & key & "reconstruction_method=0 and top_down_method=2"
        ).fetch("KEY")
        data = [fetch_data(k) for k in recording_keys]
        R, t, t_offsets, best_score = find_best_alignment(data)

        self.insert1(dict(**key, r=R, t=t, t_offsets=t_offsets, score=best_score))

    @property
    def key_source(self):
        """Only calibrate if all the reconstruction methods are computed"""
        return (
            GaitRiteSession
            - (
                GaitRiteRecording - (PersonKeypointReconstruction & "reconstruction_method=0 and top_down_method=2")
            ).proj()
        )


@schema
class GaitRiteRecordingAlignment(dj.Computed):
    definition = """
    -> GaitRiteCalibration
    -> GaitRiteRecording
    -> PersonKeypointReconstruction
    ---
    t_offset               : float
    residuals              : longblob
    """

    def make(self, key):
        R, t = (GaitRiteCalibration & key).fetch1("r", "t")
        dt, kp3d, df = fetch_data(key)
        kp3d_aligned = kp3d[:, :, :3] @ R + t
        kp3d_confidence = kp3d[..., -1:]

        def get_residuals(t_offset):
            d = extract_traces(dt, kp3d_aligned, df, t_offset)
            gt = np.concatenate([d["left_heel_gt"], d["right_heel_gt"], d["left_toe_gt"], d["right_toe_gt"]], axis=0)
            measurements = np.concatenate(
                [
                    d["left_heel_measurement"],
                    d["right_heel_measurement"],
                    d["left_toe_measurement"],
                    d["right_toe_measurement"],
                ],
                axis=0,
            )
            return measurements - gt

        def get_score(t_offset):
            d = extract_traces(dt, kp3d_aligned, df, t_offset)
            c = extract_traces(dt, kp3d_confidence, df, t_offset, 1)

            gt = np.concatenate([d["left_heel_gt"], d["right_heel_gt"], d["left_toe_gt"], d["right_toe_gt"]], axis=0)
            measurements = np.concatenate(
                [
                    d["left_heel_measurement"],
                    d["right_heel_measurement"],
                    d["left_toe_measurement"],
                    d["right_toe_measurement"],
                ],
                axis=0,
            )
            noise = np.concatenate(
                [d["left_heel_range"], d["right_heel_range"], d["left_toe_range"], d["right_toe_range"]]
            )

            confidence = np.concatenate(
                [
                    c["left_heel_measurement"],
                    c["right_heel_measurement"],
                    c["left_toe_measurement"],
                    c["right_toe_measurement"],
                ],
                axis=0,
            )

            return np.nansum(np.abs(measurements - gt) * confidence) / (1e-9 + np.nansum(confidence)) + np.nansum(
                noise * confidence
            ) / (1e-9 + np.nansum(confidence))

        offset_range = get_offset_range(dt, df)
        t_offsets = np.arange(offset_range[0], offset_range[1], 0.03)
        scores = [get_score(t) for t in t_offsets]
        t_offset = t_offsets[np.argmin(scores)]
        residuals = get_residuals(t_offset)

        self.insert1(dict(**key, t_offset=t_offset, residuals=residuals))


@schema
class GaitRiteRecordingStepPositionError(dj.Computed):
    definition = """
    -> GaitRiteRecordingAlignment
    ---
    mean_heel_error : float
    mean_toe_error : float
    """

    # note that side is a bit redundant in the primary key, but putting it here
    # lets us joint on the steps from the length errors too
    class Step(dj.Part):
        definition = """
        -> master
        step_id : int
        side          : enum('Left', 'Right')
        ---
        heel_error    : float
        toe_error     : float
        heel_noise    : float
        toe_noise     : float
        heel_conf     : float
        toe_conf      : float
        heel_x        : float
        """

    def make(self, key):

        t_offset = (GaitRiteRecordingAlignment & key).fetch1("t_offset")

        dt, kp3d, df = fetch_data(key)
        R, t = (GaitRiteCalibration & key).fetch1("r", "t")

        conf = kp3d[:, :, 3]
        kp3d = kp3d[:, :, :3] @ R + t

        step_keys = []
        for i, step in df.iterrows():
            step_key = key.copy()
            step_key["step_id"] = i
            step_key["side"] = "Left" if step["Left Foot"] else "Right"

            trace_idx = np.logical_and(
                dt >= step["First Contact Time"] + t_offset, dt <= step["Last Contact Time"] + t_offset
            )
            if step_key["side"] == "Left":
                heel_trace = kp3d[trace_idx, 0, 0]
                heel_conf = conf[trace_idx, 0]
                toe_trace = kp3d[trace_idx, 1, 0]
                toe_conf = conf[trace_idx, 1]
            else:
                heel_trace = kp3d[trace_idx, 2, 0]
                heel_conf = conf[trace_idx, 2]
                toe_trace = kp3d[trace_idx, 3, 0]
                toe_conf = conf[trace_idx, 3]

            # step_key["heel_error"] = np.sqrt(np.mean((heel_trace - step["Heel X"]) ** 2))
            step_key["heel_error"] = np.sqrt(np.mean(np.abs(heel_trace - step["Heel X"])))
            # step_key["toe_error"] = np.sqrt(np.mean((toe_trace - step["Toe X"]) ** 2))
            step_key["toe_error"] = np.sqrt(np.mean(np.abs(toe_trace - step["Toe X"])))
            step_key["heel_noise"] = np.std(heel_trace)
            step_key["toe_noise"] = np.std(toe_trace)
            step_key["heel_conf"] = np.mean(heel_conf)
            step_key["toe_conf"] = np.mean(toe_conf)
            step_key["heel_x"] = step["Heel X"]

            step_keys.append(step_key)

        key["mean_heel_error"] = np.mean([k["heel_error"] for k in step_keys])
        key["mean_toe_error"] = np.mean([k["toe_error"] for k in step_keys])

        self.insert1(key)
        self.Step.insert(step_keys)


@schema
class GaitRiteRecordingStepLengthError(dj.Computed):
    definition = """
    -> GaitRiteRecordingAlignment
    ---
    mean_step_length_error : float
    mean_stride_length_error : float
    """

    class Step(dj.Part):
        definition = """
        -> master
        step_id : int
        side          : enum('Left', 'Right')
        ---
        step_length_error    : float
        stride_length_error  : float
        """

    def make(self, key):

        t_offset = (GaitRiteRecordingAlignment & key).fetch1("t_offset")

        dt, kp3d, df = fetch_data(key)
        R, t = (GaitRiteCalibration & key).fetch1("r", "t")

        conf = kp3d[:, :, 3]
        kp3d = kp3d[:, :, :3] @ R + t

        last_left_heel = None
        last_right_heel = None

        step_keys = []
        for i, step in df.iterrows():
            step_key = key.copy()
            step_key["step_id"] = i
            step_key["side"] = "Left" if step["Left Foot"] else "Right"

            trace_idx = np.logical_and(
                dt >= step["First Contact Time"] + t_offset, dt <= step["Last Contact Time"] + t_offset
            )
            if step_key["side"] == "Left":
                heel_trace = kp3d[trace_idx, 0, 0]
            else:
                heel_trace = kp3d[trace_idx, 2, 0]

            if i >= 2:
                # GaitRite step lengths only defined after first step
                stride_length = np.abs(
                    np.mean(heel_trace) - (last_left_heel if step_key["side"] == "Left" else last_right_heel)
                )
                step_length = np.abs(
                    np.mean(heel_trace) - (last_right_heel if step_key["side"] == "Left" else last_left_heel)
                )
                step_key["step_length_error"] = step_length - step["Step Length"] * 10.0  # convert to mm
                step_key["stride_length_error"] = stride_length - step["Stride Length"] * 10.0
                step_keys.append(step_key)

            # store this position for next step computation
            if step_key["side"] == "Left":
                last_left_heel = np.mean(heel_trace)
            else:
                last_right_heel = np.mean(heel_trace)

        key["mean_step_length_error"] = np.mean([np.abs(k["step_length_error"]) for k in step_keys])
        key["mean_stride_length_error"] = np.mean([np.abs(k["stride_length_error"]) for k in step_keys])

        self.insert1(key)
        self.Step.insert(step_keys)


def get_walking_time_range(key, margin=0.5):

    assert len(GaitRiteRecordingAlignment & key) <= 1, "Select only one recording"
    assert len(GaitRiteRecordingAlignment & key) == 1, f"No GaitRiteAlignment found for  {key}"
    t_offset = (GaitRiteRecordingAlignment & key).fetch1("t_offset")
    dt, kp3d, df = fetch_data(key)
    first_step = df["First Contact Time"].min()
    last_step = df["Last Contact Time"].max()
    return first_step + t_offset - margin, last_step + t_offset + margin


def match_data(filename):

    t0, df = parse_gaitrite(filename)

    delta_t = f'ABS(TIMESTAMP(recording_timestamps) - TIMESTAMP("{t0[0]}"))'
    vid_key = MultiCameraRecording.proj(x=delta_t).fetch("KEY", "x", order_by="x ASC", limit=1, as_dict=True)[0]

    return t0[0], df, vid_key


def fetch_data(key, only_present=False):
    """Fetch the data from the database for a given GaitRite recording.

    Traces are ordered as follows: left heel,  left toe, right heel, right toe

    Args:
        key (dict): The key of the GaitRiteRecording.
        only_present (bool, optional): Whether to only return the data for the frames where the person is present. Defaults to False.
    Returns:
        dt (np.array): The timestamps of the video frames.
        kp3d (np.array): The 3D keypoints of the person.
        df (pd.DataFrame): The GaitRite data.
    """

    t0, df = (GaitRiteRecording & key).fetch1("gaitrite_t0", "gaitrite_dataframe")
    df = pd.DataFrame(df)

    # GaitRite uses the opposite handedness to our system
    df[["Heel Y", "Toe Y"]] = df[["Heel Y", "Toe Y"]] * -1

    timestamps = (VideoInfo * SingleCameraVideo * MultiCameraRecording & key).fetch("timestamps")[0]
    kp3d = (PersonKeypointReconstruction & key).fetch1("keypoints3d")
    present = (PersonBbox * SingleCameraVideo & key).fetch("present", limit=1)[0]
    top_down_method_name = (TopDownMethodLookup & key).fetch1("top_down_method_name")
    joint_names = TopDownPerson.joint_names(top_down_method_name)

    # when the terminal frame is missing
    timestamps = timestamps[: kp3d.shape[0]]
    present = present[: kp3d.shape[0]]
    dt = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
    if only_present:
        kp3d = kp3d[present]
        dt = dt[present]

    target_names = ["Left Heel", "Left Big Toe", "Right Heel", "Right Big Toe"]
    joint_idx = np.array([joint_names.index(j) for j in target_names])
    kp3d = kp3d[:, joint_idx]

    gaitrite_offset = (t0 - timestamps[0]).total_seconds()
    df["First Contact Time"] += gaitrite_offset
    df["Last Contact Time"] += gaitrite_offset

    return dt, kp3d, df


def plot_data(key, t_offset=None, axis=0):

    import matplotlib.pyplot as plt

    if t_offset is None:
        t_offset = (GaitRiteRecordingAlignment & key).fetch1("t_offset")

    dt, kp3d, df = fetch_data(key)
    R, t = (GaitRiteCalibration & key).fetch1("r", "t")

    if kp3d.shape[-1] == 4:
        conf = kp3d[:, :, 3]
    else:
        conf = np.ones_like(kp3d[:, :, 0])
    if R.shape[0] == 3:
        kp3d = kp3d[:, :, :3] @ R + t
    else:
        print(kp3d.shape, R.shape, t.shape)
        kp3d = kp3d[:, :, :2] @ R + t

    idx = df["Left Foot"]

    _, ax = plt.subplots(3, 2, sharex=True, sharey=True)
    ax = ax.flatten()

    def step_plot(df, field, style, size, ax):
        ax.plot(
            df[["First Contact Time", "Last Contact Time"]].T + t_offset,
            np.stack([df[field].values, df[field].values]),
            style,
            markersize=size,
        )

    for i in range(4):
        ax[i].plot(dt, kp3d[:, i, axis], "k")

        if axis == 0:
            a = "X"
        elif axis == 1:
            a = "Y"
        else:
            continue

        if i == 0:
            step_plot(df.loc[idx], f"Heel {a}", "bo-", 2.5, ax[i])
        elif i == 1:
            step_plot(df.loc[idx], f"Toe {a}", "bo-", 1.5, ax[i])
        elif i == 2:
            step_plot(df.loc[~idx], f"Heel {a}", "ro-", 2.5, ax[i])
        elif i == 3:
            step_plot(df.loc[~idx], f"Toe {a}", "ro-", 1.5, ax[i])

    ax[4].plot(dt, conf[:, 0] * 5000, "b")
    ax[4].plot(dt, conf[:, 2] * 5000, "r")
    ax[5].plot(dt, conf[:, 1] * 5000, "b")
    ax[5].plot(dt, conf[:, 3] * 5000, "r")

    ax[0].set_title("Left Heel")
    ax[1].set_title("Left Toe")
    ax[2].set_title("Right Heel")
    ax[3].set_title("Right Toe")
    ax[4].set_ylabel("Confidence")
    ax[5].set_ylabel("Confidence")
    ax[4].set_xlabel("Time (s)")
    ax[5].set_xlabel("Time (s)")
    ax[0].set_ylabel("Position (mm)")
    ax[2].set_ylabel("Position (mm)")
    plt.tight_layout()


def import_gaitrite_files(subject_id: int, filenames: List[str]):
    """Import GaitRite files into the database.

    This expects all the filenames correspond to one subject, but nothing
    about the code will enforce this.

    Args:
        subject_id (int): The subject ID to associate with the files.
        filenames (List[str]): The list of GaitRite files to import.
    """

    data = [match_data(filename) for filename in filenames]
    min_t0 = min([d[0] for d in data])

    # open a datajoint transaction
    with dj.conn().transaction:

        # insert the session
        key = dict(subject_id=subject_id, gaitrite_sesion_date=min_t0)
        GaitRiteSession.insert1(key)

        # insert the recordings
        for filename, (t0, df, vid_key) in zip(filenames, data):

            x = vid_key.pop("x")

            if np.abs(x) > 15:
                print(f"Skipping {filename} due to large time offset: {x} seconds")
                continue

            # update the key with the video key
            key.update(vid_key)

            # get the filename without extension without from the full path
            stripped_filename = os.path.split(os.path.splitext(filename)[0])[1]
            print(stripped_filename)

            # convert the pandas dataframe to a list of dictionaries:
            df_dict = df.to_dict("records")

            print(key)

            GaitRiteRecording.insert1(
                dict(**key, gaitrite_filename=stripped_filename, gaitrite_dataframe=df_dict, gaitrite_t0=t0),
                skip_duplicates=True,
            )
