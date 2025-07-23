from pose_pipeline.pipeline import TopDownPerson, TopDownMethodLookup, BottomUpPeople, BottomUpMethodLookup, PersonBbox
from multi_camera.datajoint.sessions import Session, Recording
from multi_camera.datajoint.multi_camera_dj import (
    MultiCameraRecording,
    SingleCameraVideo,
    CalibratedRecording,
    PersonKeypointReconstruction,
)
from multi_camera.datajoint.easymocap import EasymocapSmpl, EasymocapTracking
from body_models.datajoint.kinematic_dj import KinematicReconstruction

tracking_method = 21

bottom_up = BottomUpPeople * BottomUpMethodLookup & {"bottom_up_method_name": "Bridging_OpenPose"}
videos_missing_bottom_up = (Recording * SingleCameraVideo - bottom_up)

# This probably could be less convoluted but I am not sure how
# Getting the total number of trials missing easymocap tracking 
# and subtracting trials that have videos missing bottom up since
# easymocap cannot be run until bridging is run for the entire trial
trials_missing_bottom_up = (Recording & (videos_missing_bottom_up).proj()).proj()
trials_missing_easymocap = (Recording & CalibratedRecording - EasymocapTracking)

top_down = TopDownPerson * TopDownMethodLookup & {"top_down_method_name": "Bridging_bml_movi_87"} & {"tracking_method": tracking_method}
easymocap_bboxes = PersonBbox() & {'tracking_method': tracking_method}
missing = (MultiCameraRecording & Recording - CalibratedRecording).proj()

def get_stats(filter):

    stats = {
        "Number of recordings": Recording & filter,
        "Number of videos missing bottom up": videos_missing_bottom_up & filter,
        "Number missing calibration": missing & filter,
        "Calibrated awaiting initial reconstruction": (trials_missing_easymocap - trials_missing_bottom_up) & filter,
        "Empty tracks": (Recording & (CalibratedRecording * EasymocapTracking & 'num_tracks=0') & filter),
        "Easymocap awaiting EasymocapSmpl": (Recording & CalibratedRecording * EasymocapTracking) & filter - EasymocapSmpl,
        "Reconstructed awaiting annotation": (Recording & CalibratedRecording * EasymocapSmpl) & filter - (SingleCameraVideo * easymocap_bboxes),
        "Annotated videos missing top down": Recording * SingleCameraVideo * easymocap_bboxes - top_down & filter,
        "Annotated without reconstruction": (Recording & CalibratedRecording & (SingleCameraVideo * easymocap_bboxes) - PersonKeypointReconstruction) & filter,
        "Sessions missing kinematic reconstruction method 137": (SessionCalibration.Grouping & (Recording * filter)) - KinematicReconstruction & {"kinematic_reconstruction_settings_num": 137},
    }

    return stats

def get_project_stats_counts(project):
    filter = MultiCameraRecording & {'video_project': project}
    stats = get_stats(filter)
    counts = {k: len(v) for k, v in stats.items()}
    return {'Project': project, **counts}


def get_quality_stats(filter):
    pass