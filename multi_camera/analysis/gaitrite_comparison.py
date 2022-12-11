import pandas as pd
import numpy as np


def parse_gaitrite(filename: str):
    """Parse a Gaitrite file into a pandas dataframe.

        Args:
            filename (str): The path to the Gaitrite file.
        Returns: A tuple of the start time and the dataframe.
    """

    from datetime import datetime

    columns = ['Heel X', 'Heel Y', 'Toe X', 'Toe Y',
               'Lowest X', 'Highest X', 'Lowest Y', 'Highest Y', 'First Contact Time', 'Last Contact Time',
               'Foot Flat Time', 'Begin Time', 'End Time', 'Left/Right Foot', 'Step Length', 'Stride Length',
               'Base of Support', 'Step Time', 'Stride Time', 'Swing Time', 'Stance Time',
               'Single Support Time (sec)', 'Double Support Time (sec)', 'Stride Velocity', 'Real Foot Flag',
               'Pass Number', 'Toe In / Out', 'Step Width', 'Stride Width', 'Heel On', 'Heel Off', 'Mid On',
               'Mid Off', 'Toe On', 'Toe Off ', 'Heel off/on', 'Double Support Loading ', 'Double Support Unloading ']

    df = pd.read_csv(filename, sep='\t', skipinitialspace=True)
    #df = pd.read_csv(filename, sep='\s*\t\s*', skipinitialspace=True)


    t0 = (df.iloc[0]['Date / Time Stamp'], df.iloc[0]['Computer Time (MSec)'])

    try:
        t0 = (datetime.strptime(t0[0], '%m/%d/%Y %I:%M:%S %p'), t0[1])
    except:
        print('falling back to alternative time parsing')
        t0 = (datetime.strptime(t0[0], '%m/%d/%Y %H:%M'), t0[1])

    df = df[columns]
    df['Left Foot'] = df['Left/Right Foot'] > 0.5
    df = df.dropna()

    # Convert to mm
    df[['Heel X', 'Heel Y', 'Toe X', 'Toe Y']] = df[['Heel X', 'Heel Y', 'Toe X', 'Toe Y']] * 10 / 0.8

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
        start_index = np.searchsorted(timestamps, interval[0], side='left')
        end_index = np.searchsorted(timestamps, interval[1], side='right')

        if end_index == start_index:
            averages.append(np.empty(shape=(trace.shape[1])) * np.nan)
            stds.append(np.empty(shape=(trace.shape[1])) * np.nan)
            continue

        # Compute the average of the trace over the interval
        average = np.mean(trace[start_index:end_index], axis=0)
        std = np.std(trace[start_index:end_index], axis=0)
        std = np.percentile(trace[start_index:end_index], 90, axis=0) - \
              np.percentile(trace[start_index:end_index], 10, axis=0)

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

    A = measurements_centered
    B = ground_truth_centered
    # from scipy.linalg.orthogonal_procrustes
    u, w, vt = scipy.linalg.svd(B.T.dot(A).T)
    R = u.dot(vt)

    # Compute the translation vector by taking the difference
    # between the means of the centered matrices
    t = g_mean - np.dot(m_mean, R)

    # Return the rotation matrix and translation vector
    return R, t


def extract_traces(timestamps: np.array, keypoints: np.array, gaitrite_df: pd.DataFrame, t_offset: float = 0.0, cutoff=2):
    """Extract the traces from the Gaitrite dataframe.

        Args:
            timestamps (np.ndarray): The timestamps of the keypoints.
            keypoints (np.ndarray): The keypoints.
            gaitrite_df (pd.DataFrame): The Gaitrite dataframe.
            t_offset (float): The offset to add to the Gaitrite timestamps.
        Returns: A tuple of the left and right heel and toe traces.
    """

    idx = gaitrite_df['Left Foot']

    left_intervals = gaitrite_df.loc[idx, ['First Contact Time', 'Last Contact Time']].values + t_offset
    left_heel_gt = gaitrite_df.loc[idx, ['Heel X', 'Heel Y']].values
    left_toe_gt = gaitrite_df.loc[idx, ['Toe X', 'Toe Y']].values

    right_intervals = gaitrite_df.loc[~idx, ['First Contact Time', 'Last Contact Time']].values + t_offset
    right_heel_gt = gaitrite_df.loc[~idx, ['Heel X', 'Heel Y']].values
    right_toe_gt = gaitrite_df.loc[~idx, ['Toe X', 'Toe Y']].values

    left_heel_measurement, s1, = trace_average(timestamps, keypoints[:, 0, :cutoff], left_intervals)
    left_toe_measurement, s2, = trace_average(timestamps, keypoints[:, 1, :cutoff], left_intervals)
    right_heel_measurement, s3, = trace_average(timestamps, keypoints[:, 2, :cutoff], right_intervals)
    right_toe_measurement, s4, = trace_average(timestamps, keypoints[:, 3, :cutoff], right_intervals)

    return {'left_heel_measurement': left_heel_measurement,'left_heel_range': s1,
            'left_toe_measurement': left_toe_measurement, 'left_toe_range': s2,
            'right_heel_measurement': right_heel_measurement, 'right_heel_range': s3,
            'right_toe_measurement': right_toe_measurement, 'right_toe_range': s4,
            'left_heel_gt': left_heel_gt, 'left_toe_gt': left_toe_gt,
            'right_heel_gt': right_heel_gt, 'right_toe_gt': right_toe_gt}


def score_extraction(extraction: dict):
    """Score the extraction."""

    ranges = np.concatenate([extraction['left_heel_range'], extraction['left_toe_range'],
                             extraction['right_heel_range'], extraction['right_toe_range']], axis=0)
    ranges = ranges[:, :2]

    measurements = np.concatenate([extraction['left_heel_measurement'], extraction['left_toe_measurement'],
                                   extraction['right_heel_measurement'], extraction['right_toe_measurement']], axis=0)
    scores = measurements[:, -1]

    return np.nansum(ranges * scores[:, None]) / np.nansum(scores)


def find_local_minima(data: tuple, t_range: float = 14.0, ret_scores=False):
    """Find the local minima of the scores separated by at least 10 frames.

        Args:
            data (tuple): The data to temporally align, which is a tuple of the timestamps, keypoints and Gaitrite dataframe.
            t_range (float): range of the temporal offset to search.
            ret_scores (bool): Whether to return the t_offsets and scores for each
        Returns: The time offsets of the minima.
    """

    from scipy.signal import argrelextrema

    t_offsets = np.linspace(-t_range, t_range, 200)
    scores = np.array([score_extraction(extract_traces(*data, t, 4)) for t in t_offsets])

    # Find the local minima
    minima = argrelextrema(scores, np.less, order=10)[0]

    if ret_scores:
        return t_offsets[minima], t_offsets, scores

    # Return the indices of the minima
    return t_offsets[minima]


def align_steps(timestamps: np.array, keypoints: np.array, gaitrite_df: pd.DataFrame, t_offset: float = 0.0):
    """ Align the keypoints to the gaitrite data.

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

    gt = np.concatenate([d['left_heel_gt'], d['left_toe_gt'], d['right_heel_gt'], d['right_toe_gt']], axis=0)
    measurements = np.concatenate([d['left_heel_measurement'], d['left_toe_measurement'], d['right_heel_measurement'], d['right_toe_measurement']], axis=0)

    R, t = procrustes(measurements, gt)

    residuals = np.dot(measurements, R) + t - gt

    stability = np.concatenate([d['left_heel_range'], d['left_toe_range'], d['right_heel_range'], d['right_toe_range']], axis=0)
    stability = np.mean(stability, axis=0)

    return R, t, residuals, stability


def align_steps_multiple_trials(data, t_offsets):
    """Align the keypoints to the gaitrite data for multiple recordings.

        Args:
            data (list): The data to align, each entry is a tuple of (timestamps, keypoints, gaitrite_df).
            t_offsets (list): The time offsets to use for each recording.
        Returns: The rotation rector, translation, and residuals
    """

    gt = []
    measurements = []
    idxs = []

    for i, (d, t) in enumerate(zip(data, t_offsets)):
        d = extract_traces(*d, t, 2)

        trial = np.concatenate([d['left_heel_gt'], d['left_toe_gt'], d['right_heel_gt'], d['right_toe_gt']], axis=0)
        gt.append(trial)
        measurements.append(np.concatenate([d['left_heel_measurement'], d['left_toe_measurement'], d['right_heel_measurement'], d['right_toe_measurement']], axis=0))
        idxs.append(np.zeros((trial.shape[0],)) + i)

    gt = np.concatenate(gt)
    measurements = np.concatenate(measurements)
    idxs = np.concatenate(idxs)

    R, t = procrustes(measurements, gt)

    residuals = np.dot(measurements, R) + t - gt

    grouped_residuals = []
    for i in np.unique(idxs):
        r = np.mean(np.abs(residuals[idxs == i, :]))
        grouped_residuals.append(r)

    return R, t, residuals, np.array(grouped_residuals)


def find_best_alignment(data: list, maxiters=5):
    """ Find the best alignment of the data.

        For each trial, three are multiple possible time offsets corresponding to
        the periodic nature of gait. An exhaustive grid search on all the permutations
        is impractical, so we use an iterative line search for each trial which fairly
        rapidly converges to a good solution.

        Args:
            data (list): The data to align, each entry is a tuple of (timestamps, keypoints, gaitrite_df).
        Returns: The rotation vector, translation, and time offsets found
    """

    t_offsets = [find_local_minima(d) for d in data]
    t_idx = [np.argmin(np.abs(t)) for t in t_offsets]

    best_score = 1e10

    for _ in range(maxiters):
        test_offsets = [t[i] for t, i in zip(t_offsets, t_idx)]
        R, t, residuals, grouped_residuals = align_steps_multiple_trials(data, test_offsets)

        score = np.nanmean(np.abs(residuals))
        if score < best_score:
            best_score = score
        elif score == best_score:
            break

        order = np.flip(np.argsort(grouped_residuals))
        for trial in order:
            scores = []
            test_offsets = [t[i] for t, i in zip(t_offsets, t_idx)]
            for test_offset in t_offsets[trial]:
                test_offsets[trial] = test_offset
                R, t, residuals, grouped_residuals = align_steps_multiple_trials(data, test_offsets)
                scores.append(np.nanmean(np.abs(residuals)))

            t_idx[trial] = np.argmin(scores)

    return R, t, t_offsets, best_score
