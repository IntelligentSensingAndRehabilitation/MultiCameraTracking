from pose_pipeline.pipeline import TopDownPerson, TopDownMethodLookup, BottomUpPeople, BottomUpMethodLookup, PersonBbox
from multi_camera.datajoint.sessions import Session, Recording
from multi_camera.datajoint.multi_camera_dj import (
    MultiCameraRecording,
    SingleCameraVideo,
    CalibratedRecording,
    PersonKeypointReconstruction,
)
from multi_camera.datajoint.easymocap import EasymocapSmpl, EasymocapTracking

bottom_up = BottomUpPeople * BottomUpMethodLookup & {"bottom_up_method_name": "Bridging_OpenPose"}
top_down = TopDownPerson * TopDownMethodLookup & {"top_down_method_name": "Bridging_bml_movi_87"}
missing = (MultiCameraRecording & Recording - CalibratedRecording).proj()

def get_stats(filter):

    stats = {
        "Number of recordings": Recording & filter,
        "Number of videos missing bottom up": (Recording * SingleCameraVideo - bottom_up) & filter,
        "Number missing calibration": missing & filter,
        "Calibrated awaiting initial reconstruction": (Recording & CalibratedRecording - EasymocapTracking) & filter,
        "Empty tracks": (Recording & (CalibratedRecording * EasymocapTracking & 'num_tracks=0') & filter),
        "Easymocap awaiting EasymocapSmpl": (Recording & CalibratedRecording * EasymocapTracking) & filter - EasymocapSmpl,
        "Reconstructed awaiting annotation": (Recording & CalibratedRecording * EasymocapSmpl) & filter - (SingleCameraVideo * PersonBbox),
        "Annotated videos missing top down": Recording * SingleCameraVideo * PersonBbox - top_down & filter,
        "Annotated without reconstruction": (Recording & CalibratedRecording & (SingleCameraVideo * PersonBbox) - PersonKeypointReconstruction) & filter,
    }

    return stats

def get_project_stats_counts(project):
    filter = MultiCameraRecording & {'video_project': project}
    stats = get_stats(filter)
    counts = {k: len(v) for k, v in stats.items()}
    return {'Project': project, **counts}


def get_quality_stats(filter):
    pass