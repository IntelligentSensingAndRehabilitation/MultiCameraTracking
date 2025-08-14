"""
Handles deleting a recording along with all of the associated data.

This is necessary due to our inverted dependencies.
"""

from pose_pipeline.pipeline import Video, PersonBbox
from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording, SingleCameraVideo, CalibratedRecording
from multi_camera.datajoint.sessions import Recording
from multi_camera.datajoint.session_calibrations import SessionCalibration

def thorough_delete_recording(recording):

    assert len(Recording & recording) == 1, "Only one recording should match"

    recording_key = (Recording & recording).fetch1("KEY")
    multicamera_key = (MultiCameraRecording & recording).fetch1("KEY")
    single_camera_keys = (SingleCameraVideo & multicamera_key).fetch("KEY")
    video_keys = (Video & single_camera_keys).fetch("KEY")

    assert len(video_keys) < 20, "Too many videos to delete"
    
    print("Deleting recording:", recording_key)
    print("MMC Entry", multicamera_key)
    print("Video Entries", video_keys)

    (Recording & recording).delete()
    (MultiCameraRecording & multicamera_key).delete()
    (Video & video_keys).delete()

def thoruough_delete_calibration(recording, max_recordings=10):

    recording_key = (Recording & recording).fetch("KEY")
    multicamera_key = (MultiCameraRecording & recording_key).fetch("KEY")
    grouping_keys = (SessionCalibration & (SessionCalibration.Grouping & recording_key)).fetch("KEY")
    calibrated_recording_keys = (CalibratedRecording & multicamera_key).fetch("KEY")
    single_camera_keys = (SingleCameraVideo & calibrated_recording_keys).fetch("KEY")
    person_keys = (PersonBbox & single_camera_keys).fetch("KEY")

    if len(calibrated_recording_keys) > max_recordings:
        raise ValueError(f"Too many calibrated recordings to delete: {len(calibrated_recording_keys)}. If you intend this override the max_recordings parameter.")

    print("Deleting bounding boxes", person_keys)
    print("Deleting session calibrations", grouping_keys)
    print("Deleting calibrated recordings", calibrated_recording_keys)

    (PersonBbox & person_keys).delete()
    (SessionCalibration & grouping_keys).delete()
    (CalibratedRecording & calibrated_recording_keys).delete()
