import numpy as np
import datajoint as dj

from multi_camera.analysis import gaitrite_comparison as analysis_gaitrite_comparison
from multi_camera.validation.biomechanics.biomechanics import BiomechanicalReconstruction
from multi_camera.datajoint import gaitrite_comparison as dj_gaitrite_comparison
from multi_camera.datajoint.gaitrite_comparison import GaitRiteCalibration, GaitRiteRecordingAlignment

schema = dj.schema("multicamera_tracking_biomechanics_gaitrite")


def get_foot_positions(skeleton, poses):
    heels = []
    for p in poses:
        skeleton.setPositions(p)

        right_heel = None
        right_toes = None
        left_heel = None
        left_toes = None

        for b in skeleton.getBodyNodes():
            n = b.getNumShapeNodes()
            for i in range(n):
                s = b.getShapeNode(i)

                name = s.getName()
                if "calcn_r" in name:
                    right_heel = s.getWorldTransform().matrix()[:3, -1]
                elif "toes_r" in name:
                    right_toes = s.getWorldTransform().matrix()[:3, -1]
                elif "calcn_l" in name:
                    left_heel = s.getWorldTransform().matrix()[:3, -1]
                elif "toes_l" in name:
                    left_toes = s.getWorldTransform().matrix()[:3, -1]

        assert left_heel is not None and right_heel is not None and left_toes is not None and right_toes is not None

        heels.append([left_heel, left_toes, right_heel, right_toes])

    # convert to mm expected by calibration code
    heels = np.array(heels) * 1000.0

    # append ones on last axis to fake confidence estimates
    heels = np.concatenate([heels, np.ones(heels.shape[:-1] + (1,))], axis=-1)

    return heels


def fetch_data(key, skeleton=None):
    from multi_camera.analysis.biomechanics import bilevel_optimization

    df = dj_gaitrite_comparison.fetch_data(key)[-1]

    if skeleton is None:
        model_name, skeleton_def = (BiomechanicalReconstruction & key).fetch1("model_name", "skeleton_definition")
        skeleton = bilevel_optimization.reload_skeleton(model_name, skeleton_def["group_scales"], return_map=False)

    timestamps, poses = (BiomechanicalReconstruction.Trial & key).fetch1("timestamps", "poses")

    foot_positions = get_foot_positions(skeleton, poses)

    return timestamps, foot_positions, df


def align_trials(key):
    from multi_camera.analysis.biomechanics import bilevel_optimization

    assert len(BiomechanicalReconstruction & key) == 1

    model_name, skeleton_def = (BiomechanicalReconstruction & key).fetch1("model_name", "skeleton_definition")
    skeleton = bilevel_optimization.reload_skeleton(model_name, skeleton_def["group_scales"], return_map=False)

    trial_keys = (BiomechanicalReconstruction.Trial & key).fetch("KEY")
    data = [fetch_data(k, skeleton) for k in trial_keys]
    toffsets = [(GaitRiteRecordingAlignment & k).fetch1("t_offset") for k in trial_keys]

    res = analysis_gaitrite_comparison.align_steps_multiple_trials(data, toffsets)
    R, t, s, residuals, grouped_residuals, score = res

    return R, t, s, toffsets, score


@schema
class BiomechanicsCalibration(dj.Computed):
    definition = """
    -> BiomechanicalReconstruction
    ---
    r          : longblob
    t          : longblob
    s          : float
    t_offsets  : longblob   # time offsets
    score      : float      # score of the best alignment
    """

    class Trial(dj.Part):
        definition = """
        -> master
        -> BiomechanicalReconstruction.Trial
        ---
        t_offset   : float      # time offset
        """

    def make(self, key):
        R, t, s, toffsets, score = align_trials(key)
        print(R, t, s)
        self.insert1(dict(key, r=R, t=t, s=s, t_offsets=toffsets, score=score))

        trial_keys = (BiomechanicalReconstruction.Trial & key).fetch("KEY")
        for k, toffset in zip(trial_keys, toffsets):
            self.Trial.insert1(dict(**k, t_offset=toffset))

    @property
    def key_source(self):
        return BiomechanicalReconstruction * GaitRiteCalibration


@schema
class BiomechanicsStepLengthError(dj.Computed):
    definition = """
    -> BiomechanicalReconstruction.Trial
    -> BiomechanicsCalibration.Trial
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
        t_offset = (BiomechanicsCalibration.Trial & key).fetch1("t_offset")

        dt, kp3d, df = fetch_data(key)
        R, t, s = (BiomechanicsCalibration & key).fetch1("r", "t", "s")

        conf = kp3d[:, :, 3]
        kp3d = kp3d[:, :, :3] @ R * s + t

        last_left_heel = None
        last_right_heel = None

        step_keys = []
        for i, step in df.iterrows():
            step_key = key.copy()
            step_key["step_id"] = i
            step_key["side"] = "Left" if step["Left Foot"] else "Right"

            trace_idx = np.logical_and(dt >= step["First Contact Time"] + t_offset, dt <= step["Last Contact Time"] + t_offset)
            if step_key["side"] == "Left":
                heel_trace = kp3d[trace_idx, 0, 0]
            else:
                heel_trace = kp3d[trace_idx, 2, 0]

            if i >= 2 and last_left_heel is not None and last_right_heel is not None:
                # GaitRite step lengths only defined after first step
                stride_length = np.abs(np.median(heel_trace) - (last_left_heel if step_key["side"] == "Left" else last_right_heel))
                step_length = np.abs(np.median(heel_trace) - (last_right_heel if step_key["side"] == "Left" else last_left_heel))
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
class BiomechanicsStepWidthError(dj.Computed):
    definition = """
    -> BiomechanicalReconstruction.Trial
    -> BiomechanicsCalibration
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
        t_offset = (BiomechanicsCalibration.Trial & key).fetch1("t_offset")

        dt, kp3d, df = fetch_data(key)
        R, t, s = (BiomechanicsCalibration & key).fetch1("r", "t", "s")

        conf = kp3d[:, :, 3]
        kp3d = kp3d[:, :, :3] @ R * s + t

        last_left_heel = None
        last_right_heel = None
        last_step = None

        step_keys = []
        for i, step in df.iterrows():
            step_key = key.copy()

            trace_idx = np.logical_and(dt >= step["First Contact Time"] + t_offset, dt <= step["Last Contact Time"] + t_offset)
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
                step_width = np.linalg.norm(np.cross(current_heel - last_same_heel, last_same_heel - last_other_heel)) / np.linalg.norm(
                    current_heel - last_same_heel
                )

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


def plot_data(key, t_offset=None, axis=0):
    import matplotlib.pyplot as plt

    if t_offset is None:
        t_offset = (BiomechanicsCalibration.Trial & key).fetch1("t_offset")

    dt, kp3d, df = fetch_data(key)
    R, t, s = (BiomechanicsCalibration & key).fetch1("r", "t", "s")

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
