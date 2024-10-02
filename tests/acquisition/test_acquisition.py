import pytest
import asyncio
import tempfile
import os
from multi_camera.acquisition.flir_recording_api import FlirRecorder
import json
import cv2
import numpy as np
import pandas as pd

@pytest.mark.parametrize('config_file', [os.path.join(os.path.dirname(__file__), 'test_configs/mobile_system_config.yaml')])

def test_flir_recording(config_file):
    acquisition = FlirRecorder()

    asyncio.run(acquisition.configure_cameras(config_file=config_file))
    print("Camera Status:")
    print(asyncio.run(acquisition.get_camera_status()))

    # Use the mounted volume for output
    output_dir = '/Mocap/tests/testdata'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique filename for this test run
    video_filename = "test_recording"
    recording_path = os.path.join(output_dir, video_filename)
    
    print("Recording Path: ", recording_path)

    acquisition.start_acquisition(recording_path=recording_path, max_frames=500)

    acquisition.close()


def video_quality_test(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video Properties:")
    print(f"Frame Count: {frame_count}")
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")

    # Initialize variables for tests
    actual_frame_count = 0
    prev_timestamp = None
    # frame_diffs = []
    corrupt_frames = 0
    # frozen_frames = 0
    timestamp_gaps = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        actual_frame_count += 1

        # Check for corrupt frames
        if frame is None or frame.size == 0:
            corrupt_frames += 1
            continue

        # Check frame dimensions
        if frame.shape[:2] != (height, width):
            print(f"Inconsistent frame dimensions at frame {actual_frame_count}")

        # Get frame timestamp
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # Check timestamp consistency
        if prev_timestamp is not None:
            if timestamp <= prev_timestamp:
                print(f"Non-monotonic timestamp at frame {actual_frame_count}")
            elif timestamp - prev_timestamp > 1000/fps * 2:  # Allow for 2x expected frame duration
                timestamp_gaps.append((actual_frame_count, timestamp - prev_timestamp))

        prev_timestamp = timestamp
        # print(timestamp, timestamp - prev_timestamp, 1000/fps * 2)
        # # Calculate frame difference
        # if len(frame_diffs) > 0:
        #     diff = np.mean(np.abs(frame.astype(np.float32) - prev_frame.astype(np.float32)))
        #     frame_diffs.append(diff)
        #     if diff < 1.0:  # Threshold for frozen frame detection
        #         frozen_frames += 1

        # prev_frame = frame.copy()

    cap.release()

    # Analyze results
    print(f"\nTest Results:")
    print(f"Actual Frame Count: {actual_frame_count}")
    print(f"Corrupt Frames: {corrupt_frames}")
    # print(f"Frozen Frames: {frozen_frames}")
    print(f"Frame Drop Rate: {(frame_count - actual_frame_count) / frame_count:.2%}")

    if timestamp_gaps:
        print(f"Large Timestamp Gaps: {len(timestamp_gaps)}")
        for frame, gap in timestamp_gaps[:5]:  # Show first 5 gaps
            print(f"  Frame {frame}: {gap:.2f}ms")


def check_lengths(json_data):
    n_frames = len(json_data['real_times'])
    assert n_frames == len(json_data['timestamps']), "Timestamps length mismatch"
    assert n_frames == len(json_data['frame_id']), "Frame ID length mismatch"
    assert all(len(ts) == len(json_data['serials']) for ts in json_data['timestamps']), "Num cameras mismatch in timestamps"
    assert all(len(fid) == len(json_data['serials']) for fid in json_data['frame_id']), "Num cameras mismatch in frame_id"

def check_timestamp_zeros(json_data):
    
    # Create dataframe wtih timestamps
    df = pd.DataFrame(json_data['timestamps'], columns=json_data['serials'])

    print("")
    print(" ======== calculating with raw timestamps ========")
    print(calculate_fps(df))
    calculate_overall_timespread(df)

    # count 0s for each camera
    zero_counts = (df == 0).sum().to_dict()

    # check if any of the cameras have 0s in the timestamps
    if any(zero_counts.values()):

        # replace 0s with NaN
        df = df.replace(0, np.nan)

        # interpolate the zeros
        df_interpolated = df.interpolate(method='linear', axis=0).ffill().bfill()

        print("")
        print(" ======== calculating with interpolated timestamps ========")
        print(calculate_fps(df_interpolated))
        calculate_overall_timespread(df_interpolated)

        interpolated_ts = []

        for i, row in df_interpolated.iterrows():
            interpolated_ts.append(row.to_list())
    else:
        print("")
        print("No 0s in the timestamps")

        zero_counts = None

    return zero_counts

def calculate_fps(timestamp_df):
    # each column has the timestamps for a given camera (the camera id is the column name)
    # calculate the fps for each camera
    fps = {}

    for cam_id, timestamps in timestamp_df.items():
        # calculate the time difference between frames
        time_diff = np.diff(timestamps)
        # calculate the fps
        fps[cam_id] = 1/np.mean(time_diff * 1e-9)

    return fps

def calculate_overall_timespread(timestamp_df):
    # each column has the timestamps for a given camera (the camera id is the column name)
    # calculate the time spread for each camera
    
    # convert the timestamps to ms from ns
    initial_ts = timestamp_df.iloc[0,0]

    dt = (timestamp_df - initial_ts) * 1e-6

    spread = dt.max(axis=1) - dt.min(axis=1)

    print(f"Timestamps showed a maximum spread of {np.max(spread)} ms")

def check_frame_id_skips(json_data):
    # check if the frame id skips are consistent with the frame id
    # the frame id should be increasing by 1 for each frame
    # if the frame id skips, then the frame id should be increasing by more than 1

    # Create dataframe wtih frame ids
    df = pd.DataFrame(json_data['frame_id'], columns=json_data['serials'])

    # calculate the frame id skips
    frame_id_diff = df.diff()

    # count the number of frame id skips
    frame_id_skips = (frame_id_diff > 1).sum().to_dict()

    return frame_id_skips


def json_quality_test(json_path):
    # load the json file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # check the lengths of the data
    check_lengths(data)

    # check for zeros in the timestamps
    zero_counts = check_timestamp_zeros(data)

    if zero_counts is not None:
        print("")
        print("Number of 0s in timestamps per camera")
        print(zero_counts)

    # check for frame id skips
    frame_id_skips = check_frame_id_skips(data)

    print("")
    print("Number of frame id skips per camera")
    print(frame_id_skips)

    

def test_recording_quality():
    # Check the quality of the recorded video
    
    # read the video files and check how many frames are in each and the fps
    # the video filenames are test_recording.cam_id.mp4

    test_data_dir = '/Mocap/tests/testdata'

    # for filename in os.listdir(test_data_dir):
    #     if filename.endswith(".mp4"):
    #         video_quality_test(os.path.join(test_data_dir, filename))

        
    json_quality_test(os.path.join(test_data_dir, 'test_recording.json'))

