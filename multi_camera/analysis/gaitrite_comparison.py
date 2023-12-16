import re
import pandas as pd
import numpy as np
from typing import List


def parse_gaitrite(filename: str):
    """Parse a Gaitrite file into a pandas dataframe.

    Args:
        filename (str): The path to the Gaitrite file.
    Returns: A tuple of the start time and the dataframe.
    """

    from datetime import datetime, timedelta

    columns = [
        "Heel X",
        "Heel Y",
        "Toe X",
        "Toe Y",
        "Lowest X",
        "Highest X",
        "Lowest Y",
        "Highest Y",
        "First Contact Time",
        "Last Contact Time",
        "Foot Flat Time",
        "Begin Time",
        "End Time",
        "Left/Right Foot",
        "Step Length",
        "Stride Length",
        "Base of Support",
        "Step Time",
        "Stride Time",
        "Swing Time",
        "Stance Time",
        "Single Support Time (sec)",
        "Double Support Time (sec)",
        "Stride Velocity",
        "Real Foot Flag",
        "Pass Number",
        "Toe In / Out",
        "Step Width",
        "Stride Width",
        "Heel On",
        "Heel Off",
        "Mid On",
        "Mid Off",
        "Toe On",
        "Toe Off ",
        "Heel off/on",
        "Double Support Loading ",
        "Double Support Unloading ",
    ]

    df = pd.read_csv(filename, sep="\t", skipinitialspace=True)
    # df = pd.read_csv(filename, sep='\s*\t\s*', skipinitialspace=True)

    t0 = (df.iloc[0]["Date / Time Stamp"], df.iloc[0]["Computer Time (MSec)"])

    try:
        t0 = (datetime.strptime(t0[0], "%m/%d/%Y %I:%M:%S %p"), t0[1])
    except:
        # some files only have minute precision so use the file timestamp instead

        # Use a regular expression to extract the date and time component from the filename
        date_time_string = re.search(r"\d{8}_\d{6}", filename).group(0)
        # Parse the date and time component using the strptime method
        t0 = (datetime.strptime(date_time_string, "%Y%m%d_%H%M%S"), t0[1])

    if False:
        # add 5 hours to t0 to account for timezone difference, use time delta
        print(f"before {t0}")
        t0 = (t0[0] + timedelta(hours=5), t0[1])
        print(f"after {t0}")

    print("\n", filename, t0)

    df = df[columns]
    df["Left Foot"] = df["Left/Right Foot"] < 0.5
    df = df.dropna()

    # Convert to mm
    # get scaling from Heel X field to cm
    ratio = 2.54 / 2  # seems to be an odd ratio related to inches to cm

    m_ratio = np.mean(ratio)
    assert np.std(m_ratio) < 0.01, "Scaling factor is not constant"
    df[["Heel X", "Heel Y", "Toe X", "Toe Y"]] = df[["Heel X", "Heel Y", "Toe X", "Toe Y"]] * m_ratio * 10

    df.columns = [r.rstrip() for r in df.columns]
    return t0, df


def trace_average(timestamps: np.array, trace: np.array, intervals: np.array):
    """Compute the average of a trace over a set of intervals.

    Args:
        timestamps (np.ndarray): The timestamps of the trace.
        trace (np.ndarray): The trace to average
        intervals (np.ndarray): The intervals to average over.
    Returns: The average of the trace over the intervals.
    """

    # Initialize an array to store the averages
    averages = []
    stds = []

    # Loop over each interval
    for interval in intervals:
        # Find the indices of the timestamps that fall within the interval
        start_index = np.searchsorted(timestamps, interval[0], side="left")
        end_index = np.searchsorted(timestamps, interval[1], side="right")

        if end_index == start_index:
            averages.append(np.empty(shape=(trace.shape[1])) * np.nan)
            stds.append(np.empty(shape=(trace.shape[1])) * np.nan)
            continue

        # Compute the average of the trace over the interval
        average = np.median(trace[start_index:end_index], axis=0)
        std = np.std(trace[start_index:end_index], axis=0)
        std = np.percentile(trace[start_index:end_index], 90, axis=0) - np.percentile(
            trace[start_index:end_index], 10, axis=0
        )

        # Append the average to the list of averages
        averages.append(average)
        stds.append(std)

    # Return the list of averages
    return np.array(averages), np.array(stds)


def procrustes(measurements, ground_truth):
    import scipy.linalg

    # for certain time ranges no valid data is available so discard
    nan = np.any(np.isnan(measurements), axis=1)
    measurements = measurements[~nan]
    ground_truth = ground_truth[~nan]

    # Center the points around their means
    m_mean = np.mean(measurements, axis=0, keepdims=True)
    g_mean = np.mean(ground_truth, axis=0, keepdims=True)
    measurements_centered = measurements - m_mean
    ground_truth_centered = ground_truth - g_mean

    # dividing by norm makes the trace (A*B') = 1 and the scale computed validly
    norm_A = np.linalg.norm(measurements_centered)
    norm_B = np.linalg.norm(ground_truth_centered)
    A = measurements_centered / norm_A
    B = ground_truth_centered / norm_B
    # from scipy.linalg.orthogonal_procrustes
    u, w, vt = scipy.linalg.svd(B.T.dot(A).T)
    R = u.dot(vt)

    scale = w.sum() * norm_B / norm_A

    # Compute the translation vector by taking the difference
    # between the means of the centered matrices
    t = g_mean - np.dot(m_mean, R)

    # Return the rotation matrix and translation vector
    return R, t, scale


def extract_traces(
    timestamps: np.array, keypoints: np.array, gaitrite_df: pd.DataFrame, t_offset: float = 0.0, cutoff=3
):
    """Extract the traces from the Gaitrite dataframe.

    Args:
        timestamps (np.ndarray): The timestamps of the keypoints.
        keypoints (np.ndarray): The keypoints.
        gaitrite_df (pd.DataFrame): The Gaitrite dataframe.
        t_offset (float): The offset to add to the Gaitrite timestamps.
    Returns: A tuple of the left and right heel and toe traces.
    """

    idx = gaitrite_df["Left Foot"]

    left_intervals = gaitrite_df.loc[idx, ["First Contact Time", "Last Contact Time"]].values + t_offset
    left_heel_gt = gaitrite_df.loc[idx, ["Heel X", "Heel Y"]].values
    left_toe_gt = gaitrite_df.loc[idx, ["Toe X", "Toe Y"]].values
    left_heel_gt = np.concatenate([left_heel_gt, np.zeros([left_heel_gt.shape[0], 1])], axis=1)
    left_toe_gt = np.concatenate([left_toe_gt, np.zeros([left_toe_gt.shape[0], 1])], axis=1)

    right_intervals = gaitrite_df.loc[~idx, ["First Contact Time", "Last Contact Time"]].values + t_offset
    right_heel_gt = gaitrite_df.loc[~idx, ["Heel X", "Heel Y"]].values
    right_toe_gt = gaitrite_df.loc[~idx, ["Toe X", "Toe Y"]].values
    right_heel_gt = np.concatenate([right_heel_gt, np.zeros([right_heel_gt.shape[0], 1])], axis=1)
    right_toe_gt = np.concatenate([right_toe_gt, np.zeros([right_toe_gt.shape[0], 1])], axis=1)

    (
        left_heel_measurement,
        s1,
    ) = trace_average(timestamps, keypoints[:, 0, :cutoff], left_intervals)
    (
        left_toe_measurement,
        s2,
    ) = trace_average(timestamps, keypoints[:, 1, :cutoff], left_intervals)
    (
        right_heel_measurement,
        s3,
    ) = trace_average(timestamps, keypoints[:, 2, :cutoff], right_intervals)
    (
        right_toe_measurement,
        s4,
    ) = trace_average(timestamps, keypoints[:, 3, :cutoff], right_intervals)

    return {
        "left_heel_measurement": left_heel_measurement,
        "left_heel_range": s1,
        "left_toe_measurement": left_toe_measurement,
        "left_toe_range": s2,
        "right_heel_measurement": right_heel_measurement,
        "right_heel_range": s3,
        "right_toe_measurement": right_toe_measurement,
        "right_toe_range": s4,
        "left_heel_gt": left_heel_gt,
        "left_toe_gt": left_toe_gt,
        "right_heel_gt": right_heel_gt,
        "right_toe_gt": right_toe_gt,
    }


def score_extraction(extraction: dict):
    """Score the extraction."""

    ranges = np.concatenate(
        [
            extraction["left_heel_range"],
            extraction["left_toe_range"],
            extraction["right_heel_range"],
            extraction["right_toe_range"],
        ],
        axis=0,
    )
    ranges = ranges[:, :2]

    measurements = np.concatenate(
        [
            extraction["left_heel_measurement"],
            extraction["left_toe_measurement"],
            extraction["right_heel_measurement"],
            extraction["right_toe_measurement"],
        ],
        axis=0,
    )
    scores = measurements[:, -1]

    return np.nansum(ranges * scores[:, None]) / (1e-9 + np.nansum(scores))


def get_offset_range(dt, df):
    """Get the range of the temporal offset to search.

    Args:
        dt (np.ndarray): The timestamps of the keypoints.
        df (pd.DataFrame): The Gaitrite dataframe.
    Returns: A list of the minimum and maximum offset to search.
    """

    t0 = min(df["First Contact Time"].min(), df["Last Contact Time"].min())
    tl = max(df["First Contact Time"].max(), df["Last Contact Time"].max())

    t_range = [dt[0] - t0, dt[-1] - tl]
    if True:  # t_range[0] > (t_range[1] - 1):
        # add a bit of slop
        t_range[0] = t_range[0] - 4
        t_range[1] = t_range[1] + 4
    return t_range


def find_local_minima(data: tuple, t_range: List[float] = None, ret_scores=False):
    """Find the local minima of the scores separated by at least 10 frames.

    Args:
        data (tuple): The data to temporally align, which is a tuple of the timestamps, keypoints and Gaitrite dataframe.
        t_range (float): range of the temporal offset to search.
        ret_scores (bool): Whether to return the t_offsets and scores for each
    Returns: The time offsets of the minima.
    """

    from scipy.signal import argrelextrema

    if t_range is None:
        kp = data[1]
        present = np.sum(kp[:, :, 3], axis=1) > 0
        t_range = get_offset_range(data[0][present], data[2])
        print(f"Offset range: {t_range}")

    t_offsets = np.arange(t_range[0], t_range[1], 0.05)
    scores = np.array([score_extraction(extract_traces(*data, t, 4)) for t in t_offsets])

    # Find the local minima
    minima = argrelextrema(scores, np.less, order=5)[0]

    if ret_scores:
        return t_offsets[minima], t_offsets, scores

    # Return the indices of the minima
    return t_offsets[minima]


def align_steps(timestamps: np.array, keypoints: np.array, gaitrite_df: pd.DataFrame, t_offset: float = 0.0):
    """Align the keypoints to the gaitrite data.

    Finds a rotation and translation between these two coordinate frames that best
    aligns a single trial of data. Typically not used as running on all the trials
    in a session is more reliable.

    Args:
        timestamps (np.ndarray): The timestamps of the keypoints.
        keypoints (np.ndarray): The keypoints to align. Order should be [left heel, left toe, right heel, right toe].
        gaitrite_df (pd.DataFrame): The gaitrite data.
    Returns: The aligned keypoints.
    """

    d = extract_traces(timestamps, keypoints, gaitrite_df, t_offset)

    gt = np.concatenate([d["left_heel_gt"], d["left_toe_gt"], d["right_heel_gt"], d["right_toe_gt"]], axis=0)
    measurements = np.concatenate(
        [
            d["left_heel_measurement"],
            d["left_toe_measurement"],
            d["right_heel_measurement"],
            d["right_toe_measurement"],
        ],
        axis=0,
    )

    R, t, s = procrustes(measurements, gt)

    residuals = np.dot(measurements, R) * s + t - gt

    stability = np.concatenate(
        [d["left_heel_range"], d["left_toe_range"], d["right_heel_range"], d["right_toe_range"]], axis=0
    )
    stability = np.mean(stability, axis=0)

    return R, t, s, residuals, stability


def align_steps_multiple_trials(data, t_offsets):
    """Align the keypoints to the gaitrite data for multiple recordings.

    Args:
        data (list): The data to align, each entry is a tuple of (timestamps, keypoints, gaitrite_df).
        t_offsets (list): The time offsets to use for each recording.
    Returns: The rotation rector, translation, and residuals
    """

    gt = []
    measurements = []
    noise = []
    confidence = []
    idxs = []

    for i, (d, t) in enumerate(zip(data, t_offsets)):
        c = d[1][..., -1:]
        c = extract_traces(d[0], c, d[2], t, 1)
        d = extract_traces(*d, t, 3)

        trial = np.concatenate([d["left_heel_gt"], d["left_toe_gt"], d["right_heel_gt"], d["right_toe_gt"]], axis=0)
        gt.append(trial)
        measurements.append(
            np.concatenate(
                [
                    d["left_heel_measurement"],
                    d["left_toe_measurement"],
                    d["right_heel_measurement"],
                    d["right_toe_measurement"],
                ],
                axis=0,
            )
        )
        noise.append(
            np.concatenate(
                [d["left_heel_range"], d["right_heel_range"], d["left_toe_range"], d["right_toe_range"]], axis=0
            )
        )
        confidence.append(
            np.concatenate(
                [
                    c["left_heel_measurement"],
                    c["right_heel_measurement"],
                    c["left_toe_measurement"],
                    c["right_toe_measurement"],
                ],
                axis=0,
            )
        )
        idxs.append(np.zeros((trial.shape[0],)) + i)

    gt = np.concatenate(gt)
    measurements = np.concatenate(measurements)
    noise = np.concatenate(noise)
    confidence = np.concatenate(confidence)
    idxs = np.concatenate(idxs)

    R, t, s = procrustes(measurements, gt)

    residuals = np.dot(measurements, R) * s + t - gt

    score = np.nansum(np.abs(residuals) * confidence) / (1e-9 + np.nansum(confidence)) / 2  # two columns
    if s < 0.9 or s > 1.1:
        score += 1e4

    # np.nansum(noise * confidence) / (1e-9 + np.nansum(confidence))

    grouped_residuals = []
    for i in np.unique(idxs):
        r = np.mean(np.abs(residuals[idxs == i, :]))
        grouped_residuals.append(r)

    return R, t, s, residuals, np.array(grouped_residuals), score


def find_best_alignment(data: list, maxiters=10):
    """Find the best alignment of the data.

    For each trial, three are multiple possible time offsets corresponding to
    the periodic nature of gait. An exhaustive grid search on all the permutations
    is impractical, so we use an iterative line search for each trial which fairly
    rapidly converges to a good solution.

    Args:
        data (list): The data to align, each entry is a tuple of (timestamps, keypoints, gaitrite_df).
    Returns: The rotation vector, translation, and time offsets found
    """

    t_offsets = [find_local_minima(d) for d in data]

    # in some cases it might find the wrong set of offsets. by forcing one of the
    # trials, this can be overrided.
    # t_offsets[0] = np.array([-0.242])

    t_idx = [np.argmin(np.abs(t)) for t in t_offsets]
    best_t_idx = t_idx

    best_score = 1e10

    def get_best(t_idx, best_score=1e10):
        best_t_idx = t_idx

        for _ in range(maxiters):
            test_offsets = [t[i] for t, i in zip(t_offsets, t_idx)]
            _, _, _, _, grouped_residuals, score = align_steps_multiple_trials(data, test_offsets)

            if score < best_score:
                best_score = score
                best_t_idx = t_idx
            elif score == best_score:
                break
            print(score, test_offsets)

            order = np.flip(np.argsort(grouped_residuals))
            for trial in order:
                scores = []
                test_offsets = [t[i] for t, i in zip(t_offsets, t_idx)]
                for test_offset in t_offsets[trial]:
                    test_offsets[trial] = test_offset
                    _, _, _, _, _, score = align_steps_multiple_trials(data, test_offsets)
                    scores.append(score)

                t_idx[trial] = np.argmin(scores)
        return best_t_idx, best_score

    test_offsets = [t[i] for t, i in zip(t_offsets, t_idx)]
    print(f"Before: {test_offsets}")

    best_t_idx, best_score = get_best(t_idx)

    test_offsets = [t[i] for t, i in zip(t_offsets, best_t_idx)]
    print(f"Before: {test_offsets}")

    for i in range(5):
        # try again from different initial settings
        t_idx = [np.random.randint(len(t)) for t in t_offsets]
        t_idx, score = get_best(t_idx)

        if score < best_score:
            best_t_idx = t_idx
            best_score = score

    test_offsets = [t[i] for t, i in zip(t_offsets, best_t_idx)]
    R, t, scale, residuals, _, best_score = align_steps_multiple_trials(data, test_offsets)

    print(f"Residuals (mm): {np.mean(np.abs(residuals)):.2f}. Scale: {scale:.6f}. Score: {best_score:.2f}.")

    return R, t, scale, test_offsets, best_score
