"""
SmartWheel GPIO synchronization — DataJoint tables and alignment utilities.

GPIOEdgeTrigger (Computed) extracts the rising and falling edge timestamps
written into the acquisition JSON by GPIOEdgeRecorder and stores them in
the multicamera_tracking_smartwheel schema.

SmartwheelData (Manual) holds the raw SmartWheel time-series uploaded after
recording. It is keyed the same way as the camera recording so the two
tables can be joined for alignment.

compute_smartwheel_times() and align_smartwheel_to_frames() are standalone
utility functions that do not require DataJoint and can be used in notebooks
or analysis scripts directly.
"""

import numpy as np
from typing import Tuple
import datajoint as dj

schema = dj.schema("multicamera_tracking_smartwheel")


@schema
class SmartwheelData(dj.Manual):
    """
    Raw SmartWheel instrumented-wheel data uploaded after recording.

    The primary key matches MultiCameraRecording so GPIOEdgeTrigger can
    join the two tables for alignment.

    Insert after each session:

        SmartwheelData.insert1({
            'participant_id': '101',
            'session_date': date(2025, 6, 20),
            'recording_timestamps': '2025-06-20 10:30:00',
            'camera_config_hash': 'abc123',
            'n_samples': raw.shape[0],
            'sampling_rate': 240.0,
            'data_shape': f"{raw.shape[0]} x {raw.shape[1]}",
            'smartwheel_data': raw,          # ndarray [N, features]
            'metadata': {'serial': 'SW-001', 'calibration': [...]},
        })
    """
    definition = """
    participant_id       : varchar(50)
    session_date         : date
    recording_timestamps : timestamp
    camera_config_hash   : varchar(50)
    ---
    n_samples    : int
    sampling_rate : float                   # Hz, typically 240
    data_shape   : varchar(50)              # e.g. "14400 x 5"
    smartwheel_data : longblob              # ndarray [N_samples, n_features]
    metadata     : longblob                 # dict: calibration, serial, etc.
    """


@schema
class GPIOEdgeTrigger(dj.Computed):
    """
    PTP-synchronized rising and falling edge timestamps extracted from the
    acquisition JSON gpio_line_0 key written by GPIOEdgeRecorder.

    Populated automatically once the acquisition JSON is available. Expects
    exactly one rising edge (SmartWheel start) and one falling edge
    (SmartWheel stop) per recording.

    Timestamps are in the camera's PTP timebase (seconds since epoch) and
    are directly comparable to frame timestamps in MultiCameraRecording.
    """
    definition = """
    -> SmartwheelData
    ---
    rising_time        : double    # PTP timestamp of rising edge (SmartWheel start), seconds
    falling_time       : double    # PTP timestamp of falling edge (SmartWheel stop), seconds
    recording_duration : float     # seconds between rising and falling edge
    """

    def make(self, key):
        import json
        from pathlib import Path
        from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording

        rec_key = {k: key[k] for k in [
            'participant_id', 'session_date',
            'recording_timestamps', 'camera_config_hash',
        ]}
        recording = (MultiCameraRecording & rec_key).fetch1("KEY")

        metadata_files = list(
            Path("/data").glob(f"*/{recording['video_base_filename']}*.json")
        )
        if not metadata_files:
            raise RuntimeError(
                f"No acquisition metadata JSON found for {recording['video_base_filename']}"
            )

        with open(metadata_files[0]) as f:
            metadata = json.load(f)

        gpio_data = metadata.get("gpio_line_0")
        if not gpio_data:
            raise RuntimeError(
                "gpio_line_0 key missing from acquisition metadata. "
                "Ensure GPIOEdgeRecorder is integrated into FlirRecorder "
                "and Line0 (Pin 2, opto-isolated) is physically connected "
                "to the SmartWheel TTL output with Pin 5 as ground."
            )

        ptp_times  = np.array(gpio_data["ptp_times"],  dtype=np.float64)
        edge_types = np.array(gpio_data["edge_types"],  dtype=object)

        if len(ptp_times) != 2:
            raise RuntimeError(
                f"Expected exactly 2 GPIO edges, got {len(ptp_times)}"
            )

        rising_idx  = np.where(edge_types == "rising")[0]
        falling_idx = np.where(edge_types == "falling")[0]

        if len(rising_idx) != 1 or len(falling_idx) != 1:
            raise RuntimeError(
                f"Expected 1 rising + 1 falling edge, "
                f"got {len(rising_idx)} rising / {len(falling_idx)} falling"
            )

        rising_time  = float(ptp_times[rising_idx[0]])
        falling_time = float(ptp_times[falling_idx[0]])

        if falling_time <= rising_time:
            raise RuntimeError("Falling edge must occur after rising edge")

        self.insert1({
            **key,
            "rising_time":        rising_time,
            "falling_time":       falling_time,
            "recording_duration": falling_time - rising_time,
        })


def compute_smartwheel_times(
    rising_time: float,
    falling_time: float,
    smartwheel_sample_rate: float,
) -> Tuple[np.ndarray, str]:
    """
    Compute a PTP timestamp for every SmartWheel sample between the two edges.

    Sample 0 is anchored to rising_time; subsequent samples are spaced at
    1/smartwheel_sample_rate seconds. The total number of samples is inferred
    from the recording duration, matching the sample count the SmartWheel
    hardware produced.

    Args:
        rising_time:            PTP timestamp of rising edge (seconds).
        falling_time:           PTP timestamp of falling edge (seconds).
        smartwheel_sample_rate: SmartWheel sampling rate in Hz (typically 240).

    Returns:
        sample_times: PTP timestamp for each SmartWheel sample [N_samples].
        note:         Human-readable summary of the alignment.
    """
    duration  = falling_time - rising_time
    period    = 1.0 / smartwheel_sample_rate
    n_samples = int(np.round(duration * smartwheel_sample_rate)) + 1

    sample_times = rising_time + np.arange(n_samples) * period

    note = (
        f"SmartWheel GPIO alignment: {n_samples} samples at "
        f"{smartwheel_sample_rate} Hz over {duration:.3f} s "
        f"(rising → falling edge)."
    )
    return sample_times, note


def align_smartwheel_to_frames(
    smartwheel_data: np.ndarray,
    smartwheel_times: np.ndarray,
    frame_times: np.ndarray,
) -> np.ndarray:
    """
    Linearly interpolate SmartWheel data onto camera frame timestamps.

    Because the SmartWheel runs at 240 Hz and the cameras at 30 Hz, each
    camera frame falls between 8 SmartWheel samples on average, so linear
    interpolation introduces negligible error.

    Args:
        smartwheel_data:  Raw SmartWheel samples [N_samples, N_features].
        smartwheel_times: PTP timestamps for each sample [N_samples].
                          Obtain from compute_smartwheel_times().
        frame_times:      PTP timestamps for each camera frame [N_frames].
                          Obtain from MultiCameraRecording.fetch_timestamps()
                          or equivalent.

    Returns:
        aligned: SmartWheel data interpolated to frame times
                 [N_frames, N_features].
    """
    n_frames   = len(frame_times)
    n_features = smartwheel_data.shape[1]
    aligned    = np.zeros((n_frames, n_features))

    for i, t in enumerate(frame_times):
        idx = np.searchsorted(smartwheel_times, t)
        if idx == 0:
            aligned[i] = smartwheel_data[0]
        elif idx >= len(smartwheel_times):
            aligned[i] = smartwheel_data[-1]
        else:
            t0, t1 = smartwheel_times[idx - 1], smartwheel_times[idx]
            frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            aligned[i] = (1.0 - frac) * smartwheel_data[idx - 1] + frac * smartwheel_data[idx]

    return aligned
