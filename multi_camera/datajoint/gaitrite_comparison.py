import os
import datajoint as dj
import numpy as np
import pandas as pd

from typing import List

from pose_pipeline import VideoInfo, TopDownPerson
from .multi_camera_dj import MultiCameraRecording, SingleCameraVideo, PersonKeypointReconstruction
from ..analysis.gaitrite_comparison import parse_gaitrite, extract_traces, find_best_alignment

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
        recording_keys = (GaitRiteRecording & key).fetch('KEY')
        data = [fetch_data(k) for k in recording_keys]
        R, t, t_offsets, best_score = find_best_alignment(data)

        self.insert1(dict(**key, r=R, t=t, t_offsets=t_offsets, score=best_score))
        
@schema
class GaitRiteRecordingAlignment(dj.Computed):
    definition = """
    -> GaitRiteCalibration
    -> GaitRiteRecording
    ---
    t_offset               : float
    residuals              : longblob
    """

    def make(self, key):
        R, t = (GaitRiteCalibration & key).fetch1('r', 't')
        dt, kp3d, df = fetch_data(key)
        kp3d_aligned = kp3d[:, :, :2] @ R + t
        kp3d_confidence = kp3d[..., -1:]

        def get_residuals(t_offset):
            d = extract_traces(dt, kp3d_aligned, df, t_offset)
            gt = np.concatenate([d['left_heel_gt'], d['right_heel_gt'], d['left_toe_gt'], d['right_toe_gt']], axis=0)
            measurements = np.concatenate([d['left_heel_measurement'], d['right_heel_measurement'],  d['left_toe_measurement'], d['right_toe_measurement']], axis=0)
            return measurements - gt

        def get_score(t_offset):
            d = extract_traces(dt, kp3d_aligned, df, t_offset)
            c = extract_traces(dt, kp3d_confidence, df, t_offset, 1)

            gt = np.concatenate([d['left_heel_gt'], d['right_heel_gt'], d['left_toe_gt'], d['right_toe_gt']], axis=0)
            measurements = np.concatenate([d['left_heel_measurement'], d['right_heel_measurement'],  d['left_toe_measurement'], d['right_toe_measurement']], axis=0)
            noise = np.concatenate([d['left_heel_range'], d['right_heel_range'], d['left_toe_range'], d['right_toe_range']])

            confidence = np.concatenate([c['left_heel_measurement'], c['right_heel_measurement'],  c['left_toe_measurement'], c['right_toe_measurement']], axis=0)

            return np.nansum(np.abs(measurements - gt) * confidence) / np.nansum(confidence) + \
                   np.nansum(noise * confidence) / np.nansum(confidence)

        t_offsets = np.linspace(-5, 5, 1000)
        scores = [get_score(t) for t in t_offsets]
        t_offset = t_offsets[np.argmin(scores)]
        residuals = get_residuals(t_offset)

        self.insert1(dict(**key, t_offset=t_offset, residuals=residuals))
        

def match_data(filename):

    t0, df = parse_gaitrite(filename)

    delta_t = f'ABS(TIMESTAMP(recording_timestamps) - TIMESTAMP("{t0[0]}"))'
    vid_key = MultiCameraRecording.proj(x=delta_t).fetch('KEY', 'x', order_by='x ASC', limit=1, as_dict=True)[0]

    return t0[0], df, vid_key


def fetch_data(key):
    """ Fetch the data from the database for a given GaitRite recording. """

    t0, df = (GaitRiteRecording & key).fetch1('gaitrite_t0', 'gaitrite_dataframe')
    df = pd.DataFrame(df)

    timestamps = (VideoInfo * SingleCameraVideo * MultiCameraRecording & key).fetch('timestamps')[0]
    kp3d = (PersonKeypointReconstruction & key).fetch1('keypoints3d')

    target_names = ['Left Heel', 'Left Big Toe', 'Right Heel', 'Right Big Toe']
    joint_idx = np.array([TopDownPerson.joint_names('MMPoseHalpe').index(j) for j in target_names])
    kp3d = kp3d[:, joint_idx]
    
    # when the terminal frame is missing
    timestamps = timestamps[:kp3d.shape[0]]

    dt = np.array([(t-t0).total_seconds() for t in timestamps])

    return dt, kp3d, df

def plot_data(key, t_offset=None):

    import matplotlib.pyplot as plt

    if t_offset is None:
        t_offset = (GaitRiteRecordingAlignment & key).fetch1('t_offset')

    dt, kp3d, df = fetch_data(key)
    R, t = (GaitRiteCalibration & key).fetch1('r', 't')

    kp3d = kp3d[:, :, :2] @ R + t

    idx = df['Left Foot']

    _, ax = plt.subplots(2,2)
    ax = ax.flatten()

    def step_plot(df, field, style, size, ax):
        ax.plot(df[['First Contact Time', 'Last Contact Time']].T + t_offset, 
                np.stack([df[field].values, df[field].values]), style, markersize=size)

    for i in range(4):
        ax[i].plot(dt, kp3d[:, i, 0], 'k')

        if i == 0:
            step_plot(df.loc[idx], 'Heel X', 'bo-', 2.5, ax[i])
        elif i == 1:
            step_plot(df.loc[idx], 'Toe X', 'bo-', 1.5, ax[i])
        elif i == 2:
            step_plot(df.loc[~idx], 'Heel X', 'ro-', 2.5, ax[i])
        elif i == 3:
            step_plot(df.loc[~idx], 'Toe X', 'ro-', 1.5, ax[i])


def import_gaitrite_files(subject_id: int, filenames: List[str]):
    """ Import GaitRite files into the database. 
    
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

            x = vid_key.pop('x')

            if np.abs(x) > 10:
                print(f'Skipping {filename} due to large time offset: {x} seconds')
                continue

            # update the key with the video key
            key.update(vid_key)

            # get the filename without extension without from the full path
            stripped_filename = os.path.split(os.path.splitext(filename)[0])[1]
            print(stripped_filename)
            
            # convert the pandas dataframe to a list of dictionaries:
            df_dict = df.to_dict('records')

            print(key)

            GaitRiteRecording.insert1(dict(**key, gaitrite_filename=stripped_filename, 
                                      gaitrite_dataframe=df_dict, gaitrite_t0=t0))