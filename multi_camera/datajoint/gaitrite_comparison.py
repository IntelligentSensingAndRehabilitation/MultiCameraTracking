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
    s: float               # scale
    t_offsets : longblob   # time offsets
    score : float          # score of the best alignment
    """

    def make(self, key):
        recording_keys = (
            GaitRiteRecording * PersonKeypointReconstruction & key & "reconstruction_method=0 and top_down_method=2"
        ).fetch("KEY")
        data = [fetch_data(k) for k in recording_keys]
        R, t, scale, t_offsets, best_score = find_best_alignment(data)

        self.insert1(dict(**key, r=R, t=t, s=scale, t_offsets=t_offsets, score=best_score))

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
        import scipy

        R, t, s = (GaitRiteCalibration & key).fetch1("r", "t", "s")
        dt, kp3d, df = fetch_data(key)
        kp3d_aligned = kp3d[:, :, :3] @ R * s + t
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

            confidence = np.clip(confidence, a_min=0.5, a_max=None)  # want to slightly count all steps

            return np.nansum(np.abs(measurements - gt) * confidence) / (1e-9 + np.nansum(confidence)) + np.nansum(
                noise * confidence
            ) / (1e-9 + np.nansum(confidence))

        present = np.mean(kp3d[:, :, 3], axis=1)
        present = scipy.signal.medfilt(present, 9) > 0.3
        offset_range = None

        # search around the offset found during calibration
        offsets = (GaitRiteCalibration & key).fetch1("t_offsets")
        trials = (GaitRiteRecording & (GaitRiteCalibration & key)).fetch("KEY")
        match_key = (GaitRiteRecording & key).fetch1("KEY")
        # technically this isn't very valid DJ behavior as it assumes the order the keys
        # are returned from the database. however, we didn't store this data in the best
        # format to do this "correctly" and these errors would show up when looking at results.
        offset = [offset for trial, offset in zip(trials, offsets) if trial == match_key]
        assert len(offset) == 1
        offset_range = (offset[0] - 0.5, offset[0] + 0.5)
        # offset_range = get_offset_range(dt[present], df)

        print(offset_range)
        t_offsets = np.arange(offset_range[0], offset_range[1], 0.03)
        scores = [get_score(t) for t in t_offsets]
        print(t_offsets, scores)
        t_offset = t_offsets[np.argmin(scores)]
        residuals = get_residuals(t_offset)

        self.insert1(dict(**key, t_offset=t_offset, residuals=residuals))


@schema
class GaitRiteRecordingStepPositionError(dj.Computed):
    definition = """
    -> GaitRiteRecordingAlignment
    ---
    mean_forward_heel_error : float
    mean_lateral_heel_error : float
    mean_forward_toe_error : float
    mean_lateral_toe_error : float
    """

    # note that side is a bit redundant in the primary key, but putting it here
    # lets us joint on the steps from the length errors too
    class Step(dj.Part):
        definition = """
        -> master
        step_id : int
        side          : enum('Left', 'Right')
        ---
        heel_forward_error    : float
        heel_lateral_error    : float
        toe_forward_error     : float
        toe_lateral_error     : float
        heel_noise            : float
        toe_noise             : float
        heel_conf             : float
        toe_conf              : float
        heel_x                : float
        """

    def make(self, key):

        t_offset = (GaitRiteRecordingAlignment & key).fetch1("t_offset")

        dt, kp3d, df = fetch_data(key)
        R, t, s = (GaitRiteCalibration & key).fetch1("r", "t", "s")

        conf = kp3d[:, :, 3]
        kp3d = kp3d[:, :, :3] @ R * s + t

        # process the data as if they always walk one way to reverse the sign of the error.
        # this means we get the actual offset of the heel and toe from the ground truth
        backwards = (df.iloc[-1]["Heel X"] - df.iloc[0]["Heel X"]) < 0
        sign = -1 if backwards else 1

        step_keys = []
        for i, step in df.iterrows():
            step_key = key.copy()
            step_key["step_id"] = i
            step_key["side"] = "Left" if step["Left Foot"] else "Right"

            trace_idx = np.logical_and(
                dt >= step["First Contact Time"] + t_offset, dt <= step["Last Contact Time"] + t_offset
            )

            if sum(trace_idx) == 0:
                continue

            if step_key["side"] == "Left":
                heel_trace = kp3d[trace_idx, 0, :2]
                heel_conf = conf[trace_idx, 0]
                toe_trace = kp3d[trace_idx, 1, :2]
                toe_conf = conf[trace_idx, 1]
            else:
                heel_trace = kp3d[trace_idx, 2, :2]
                heel_conf = conf[trace_idx, 2]
                toe_trace = kp3d[trace_idx, 3, :2]
                toe_conf = conf[trace_idx, 3]

            erf = lambda x: np.mean(x)
            step_key["heel_forward_error"] = erf(heel_trace[:, 0] - step["Heel X"]) * sign
            step_key["heel_lateral_error"] = erf(heel_trace[:, 1] - step["Heel Y"])
            step_key["toe_forward_error"] = erf(toe_trace[:, 0] - step["Toe X"]) * sign
            step_key["toe_lateral_error"] = erf(toe_trace[:, 1] - step["Toe Y"])
            step_key["heel_noise"] = np.std(heel_trace[:, 0])
            step_key["toe_noise"] = np.std(toe_trace[:, 0])
            step_key["heel_conf"] = np.mean(heel_conf)
            step_key["toe_conf"] = np.mean(toe_conf)
            step_key["heel_x"] = step["Heel X"]

            if (
                np.sum(trace_idx) > 0
                and ~np.isnan(step_key["heel_conf"])
                and ~np.isnan(step_key["toe_conf"])
                and step_key["heel_conf"] > 0
                and step_key["toe_conf"] > 0
                and ~np.isnan(step_key["heel_forward_error"])
            ):
                # avoid nans when GaitRite timing doesn't overlap recording. this is also reflected
                # in nan values in the confidence if the person isn't tracked at that time
                step_keys.append(step_key)

        print(len(step_keys))
        erf = lambda x: np.mean(np.abs(x))
        key["mean_forward_heel_error"] = erf([k["heel_forward_error"] for k in step_keys])
        key["mean_lateral_heel_error"] = erf([k["heel_lateral_error"] for k in step_keys])
        key["mean_forward_toe_error"] = erf([k["toe_forward_error"] for k in step_keys])
        key["mean_lateral_toe_error"] = erf([k["toe_lateral_error"] for k in step_keys])

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
        gaitrite_step_length : float
        computed_step_length : float
        """

    def make(self, key):

        t_offset = (GaitRiteRecordingAlignment & key).fetch1("t_offset")

        dt, kp3d, df = fetch_data(key)
        R, t, s = (GaitRiteCalibration & key).fetch1("r", "t", "s")

        conf = kp3d[:, :, 3]
        kp3d = kp3d[:, :, :3] @ R * s + t

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

            if i >= 2 and last_left_heel is not None and last_right_heel is not None:
                # GaitRite step lengths only defined after first step
                stride_length = np.abs(
                    np.median(heel_trace) - (last_left_heel if step_key["side"] == "Left" else last_right_heel)
                )
                step_length = np.abs(
                    np.median(heel_trace) - (last_right_heel if step_key["side"] == "Left" else last_left_heel)
                )
                step_key["step_length_error"] = step_length - step["Step Length"] * 10.0  # convert to mm
                step_key["stride_length_error"] = stride_length - step["Stride Length"] * 10.0
                step_key["gaitrite_step_length"] = step["Step Length"] * 10.0
                step_key["computed_step_length"] = step_length

                if ~np.isnan(stride_length) and ~np.isnan(step_length):
                    # avoid nans when GaitRite timing doesn't overlap recording
                    step_keys.append(step_key)

            # store this position for next step computation
            if sum(trace_idx) > 0 and ~np.isnan(np.mean(heel_trace)):
                if step_key["side"] == "Left":
                    last_left_heel = np.median(heel_trace)
                else:
                    last_right_heel = np.median(heel_trace)

        key["mean_step_length_error"] = np.mean([np.abs(k["step_length_error"]) for k in step_keys])
        key["mean_stride_length_error"] = np.mean([np.abs(k["stride_length_error"]) for k in step_keys])

        self.insert1(key)
        self.Step.insert(step_keys)


@schema
class GaitRiteRecordingStepWidthError(dj.Computed):
    definition = """
    -> GaitRiteRecordingAlignment
    ---
    mean_step_width_error   : float
    """

    class Step(dj.Part):
        definition = """
        -> master
        step_id : int
        side          : enum('Left', 'Right')
        ---
        step_width_error    : float
        gaitrite_step_width : float
        computed_step_width : float
        """

    def make(self, key):
        t_offset = (GaitRiteRecordingAlignment & key).fetch1("t_offset")

        dt, kp3d, df = fetch_data(key)
        R, t, s = (GaitRiteCalibration & key).fetch1("r", "t", "s")

        conf = kp3d[:, :, 3]
        kp3d = kp3d[:, :, :3] @ R * s + t

        last_left_heel = None
        last_right_heel = None
        last_step = None

        step_keys = []
        for i, step in df.iterrows():
            step_key = key.copy()

            trace_idx = np.logical_and(
                dt >= step["First Contact Time"] + t_offset, dt <= step["Last Contact Time"] + t_offset
            )
            if step["Left Foot"]:
                heel_trace = kp3d[trace_idx, 0, :2]
            else:
                heel_trace = kp3d[trace_idx, 2, :2]

            if i >= 2 and last_left_heel is not None and last_right_heel is not None:

                # we need to know the next contralateral foot position to compute the
                # line of progression required for measuring the step width, so update
                # the index accordingly. Note that the code below compares against the
                # last step from the iterator.
                step_key["step_id"] = i - 1
                step_key["side"] = "Left" if last_step["Left Foot"] else "Right"

                # GaitRite step lengths only defined after first step
                last_other_heel = last_right_heel if step["Left Foot"] else last_left_heel
                last_same_heel = last_left_heel if step["Left Foot"] else last_right_heel
                current_heel = np.median(heel_trace, axis=0)

                # compute distance from last_other_heel to the closets point on the line connecting
                # current_heel to last_same_heel
                step_width = np.linalg.norm(
                    np.cross(current_heel - last_same_heel, last_same_heel - last_other_heel)
                ) / np.linalg.norm(current_heel - last_same_heel)

                # step_width = np.linalg.norm(np.median(heel_trace, axis=0) - last_other_heel)
                if False:
                    # for testing and to verify we are extracting the base of support consistently
                    # with their algorithm
                    fields = ["Heel X", "Heel Y"]
                    gaitrite_step_width = np.linalg.norm(
                        np.cross(step[fields] - df.iloc[i - 2][fields], df.iloc[i - 2][fields] - df.iloc[i - 1][fields])
                        / np.linalg.norm(step[fields] - df.iloc[i - 2][fields])
                    )
                else:
                    # Note that GaitRite has a "Step Width" field, which is actually the euclidean
                    # distance between the two foot centers. It also has a "Stride Width" which is
                    # close to what we want but also uses the center of the foot. Finally, the
                    # "Base of Support" measures the distance between each foot step and the line of
                    # progression on the other side. See GAITrite-Walkway-System.pdf for more details.
                    gaitrite_step_width = last_step["Base of Support"] * 10.0  # convert to mm

                step_key["step_width_error"] = step_width - gaitrite_step_width
                step_key["gaitrite_step_width"] = gaitrite_step_width
                step_key["computed_step_width"] = step_width

                if ~np.isnan(step_width):
                    # avoid nans when GaitRite timing doesn't overlap recording
                    step_keys.append(step_key)

            # store this position for next step computation
            if sum(trace_idx) > 0 and ~np.isnan(np.mean(heel_trace)):
                if step["Left Foot"]:
                    last_left_heel = np.median(heel_trace, axis=0)
                else:
                    last_right_heel = np.median(heel_trace, axis=0)
            last_step = step

        key["mean_step_width_error"] = np.mean([np.abs(k["step_width_error"]) for k in step_keys])

        self.insert1(key)
        self.Step.insert(step_keys)


def get_walking_time_range(key, margin=2.5):

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
    R, t, s = (GaitRiteCalibration & key).fetch1("r", "t", "s")

    if kp3d.shape[-1] == 4:
        conf = kp3d[:, :, 3]
    else:
        conf = np.ones_like(kp3d[:, :, 0])
    if R.shape[0] == 3:
        kp3d = kp3d[:, :, :3] @ R * s + t
    else:
        print(kp3d.shape, R.shape, t.shape)
        kp3d = kp3d[:, :, :2] @ R * s + t

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

    from scipy.optimize import linear_sum_assignment

    t0s = []
    dfs = []
    for filename in filenames:
        t0, df = parse_gaitrite(filename)
        t0s.append(t0[0])
        dfs.append(df)
    t0s = np.array(t0s)

    possible_matches = (
        MultiCameraRecording & f'ABS(TIMESTAMP(recording_timestamps) - TIMESTAMP("{t0s[len(t0s) // 2]}")) < 60*90'
    )
    keys, recording_times = possible_matches.fetch("KEY", "recording_timestamps")

    from IPython.display import display

    display(possible_matches)

    # perform a greedy hungarian match between the t0s and the recording times
    # to find the best match
    t0s = t0s[:, np.newaxis]
    recording_times = recording_times[np.newaxis, :]
    delta_t = np.abs(t0s - recording_times)
    delta_t = np.array([[t.total_seconds() for t in ts] for ts in delta_t])
    match_files, match_keys = linear_sum_assignment(delta_t)
    if ~np.all(match_files == np.arange(len(filenames))):
        print("Could not match all filenames to recordings")
        bases = [os.path.basename(f) for f in filenames]
        print(f'Matched files: {", ".join([bases[i] for i in match_files])}')
        print(f'Unmatched files: {", ".join([bases[i] for i in np.setdiff1d(np.arange(len(filenames)), match_files)])}')
        from IPython.display import display

        display(possible_matches)
    assert np.all(match_files == np.arange(len(filenames))), "Could not match all filenames to recordings"
    filenames = [filenames[i] for i in match_files]
    keys = [keys[i] for i in match_keys]

    min_t0 = np.min(t0s)

    # open a datajoint transaction
    with dj.conn().transaction:

        # insert the session
        key = dict(subject_id=subject_id, gaitrite_sesion_date=min_t0)
        GaitRiteSession.insert1(key)

        # insert the recordings
        for filename, t0, df, vid_key in zip(filenames, t0s[:, 0], dfs, keys):
            vid_timestamp = (MultiCameraRecording & vid_key).fetch1("recording_timestamps")
            dt = (t0 - vid_timestamp).total_seconds()

            # update the key with the video key
            key.update(vid_key)

            # get the filename without extension without from the full path
            stripped_filename = os.path.split(os.path.splitext(filename)[0])[1]

            # convert the pandas dataframe to a list of dictionaries:
            df_dict = df.to_dict("records")

            print("Matched GaitRite filename: ", stripped_filename, " to ", vid_key)

            if np.abs(dt) > 60:
                print(f"Skipping {filename} due to large time offset: {dt} seconds\n")
                continue
            print("\n")

            GaitRiteRecording.insert1(
                dict(**key, gaitrite_filename=stripped_filename, gaitrite_dataframe=df_dict, gaitrite_t0=t0),
                skip_duplicates=True,
            )
