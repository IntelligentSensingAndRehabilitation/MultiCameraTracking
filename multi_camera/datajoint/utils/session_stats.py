from pose_pipeline.pipeline import TopDownPerson, TopDownMethodLookup, BottomUpPeople, BottomUpMethodLookup, PersonBbox
from multi_camera.datajoint.sessions import Session, Recording
from multi_camera.datajoint.multi_camera_dj import (
    MultiCameraRecording,
    SingleCameraVideo,
    CalibratedRecording,
    PersonKeypointReconstruction,
)
from multi_camera.datajoint.session_calibrations import SessionCalibration
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
    }

    return stats

def get_project_stats_counts(project):
    filter = MultiCameraRecording & {'video_project': project}
    stats = get_stats(filter)
    counts = {k: len(v) for k, v in stats.items()}
    return {'Project': project, **counts}


def get_reconstruction_stats(filter):
    """
    Get reconstruction statistics at the session level.
    Returns stats for both kinematic and probabilistic reconstruction methods.
    """
    from body_models.datajoint.kinematic_dj import KinematicReconstruction
    from body_models.datajoint.probabilistic_dj import ProbabilisticReconstruction

    # Kinematic reconstruction methods to track
    kinematic_methods = [137, 140, 190]

    # Probabilistic reconstruction methods to track
    probabilistic_methods = [75, 77]

    stats = {}

    # Get all sessions for this filter
    all_sessions = SessionCalibration.Grouping & (Recording * filter)

    # Exclude sessions that have trials still awaiting EasyMocap
    trials_missing_easymocap = (Recording & CalibratedRecording - EasymocapTracking) & filter
    sessions_with_pending_easymocap = SessionCalibration.Grouping & trials_missing_easymocap

    # Only include sessions where ALL trials have completed EasyMocap
    sessions_query = all_sessions - sessions_with_pending_easymocap

    # Track kinematic reconstructions
    for method_num in kinematic_methods:
        missing = sessions_query - (KinematicReconstruction & {"kinematic_reconstruction_settings_num": method_num})
        stats[f"Kinematic {method_num} - Missing"] = missing

    # Track probabilistic reconstructions
    for method_num in probabilistic_methods:
        missing = sessions_query - (ProbabilisticReconstruction & {"probabilistic_reconstruction_settings_num": method_num})
        stats[f"Probabilistic {method_num} - Missing"] = missing

    return stats


def get_project_reconstruction_stats_counts(project):
    """Get reconstruction statistics counts for a given project."""
    filter = MultiCameraRecording & {'video_project': project}
    stats = get_reconstruction_stats(filter)
    counts = {k: len(v) for k, v in stats.items()}
    return {'Project': project, **counts}


def get_quality_stats(filter):
    pass