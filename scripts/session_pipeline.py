from typing import List, Dict

import pose_pipeline
from pose_pipeline.pipeline import Video, VideoInfo, TopDownMethodLookup, PersonBbox, TrackingBboxMethodLookup
from pose_pipeline.utils.standard_pipelines import bottom_up_pipeline, top_down_pipeline
from multi_camera.datajoint.sessions import Subject, Recording
from multi_camera.datajoint.multi_camera_dj import (
    MultiCameraRecording,
    PersonKeypointReconstruction,
    PersonKeypointReconstructionMethodLookup,
    SingleCameraVideo,
    CalibratedRecording,
    Calibration,
)
from multi_camera.datajoint.easymocap import EasymocapTracking, EasymocapSmpl
from multi_camera.utils.standard_pipelines import reconstruction_pipeline
import argparse

pose_pipeline.set_environmental_variables()

pose_pipeline.env.pytorch_memory_limit()
pose_pipeline.env.tensorflow_memory_limit()
pose_pipeline.env.jax_memory_limit()


def assign_calibration():
    # find the calibration that is closest in time to each recording that also has a minimum
    # threshold. note that this will possibly allow different calibrations within a session,
    # but this is intentional to also handle using different camera setups in a session

    missing = (MultiCameraRecording & Recording - CalibratedRecording).proj()
    calibration_offset = (Calibration * MultiCameraRecording).proj(calibration_offset="ABS(recording_timestamps-cal_timestamp)")

    # only accept low reprojection errors
    calibration_offset = calibration_offset & (Calibration & "reprojection_error < 0.2")

    # find closest calibrations to an experiment
    viable = missing.aggr(calibration_offset, min_calibration_offset="MIN(calibration_offset)") & "min_calibration_offset < 1000"

    # need to awkwardly use the fetch at end to join on dependent attributes. this is basically performing an
    # argmin operation
    viable = viable.proj(calibration_offset="min_calibration_offset")
    matches = calibration_offset & viable.fetch(as_dict=True)

    CalibratedRecording.insert(matches.fetch("KEY"), skip_duplicates=True)


def preannotation_session_pipeline(keys: List[Dict] = None, bridging: bool = True):
    """
    Perform the initial scene reconstruction for annotation

    This will run the bottom up pipeline and then EasyMocap reconstruction. It is
    possible to select between keypoints from MeTRabs versus OpenPose with the bridging
    flag

    Args:
        keys (List[Dict], optional): list of recording keys. Defaults to None, in which case it
            will run on all recordings that have not been annotated
        bridging (bool, optional): whether to use the bridging keypoints. Defaults to True.

    """

    if keys is None:
        keys = (SingleCameraVideo & Recording - EasymocapSmpl).fetch("KEY")
        print("Computing initial reconstruction for {} videos".format(len(keys)))

    if bridging:
        bottom_up_pipeline(keys, bottom_up_method_name="Bridging_OpenPose", reserve_jobs=True)
    else:
        bottom_up_pipeline(keys, bottom_up_method_name="OpenPose_HR", reserve_jobs=True)

    # now run easymocap
    VideoInfo.populate(SingleCameraVideo * MultiCameraRecording & keys, reserve_jobs=True)
    EasymocapTracking.populate(MultiCameraRecording * CalibratedRecording & keys, reserve_jobs=True, suppress_errors=True)
    EasymocapSmpl.populate(MultiCameraRecording * CalibratedRecording & keys, reserve_jobs=True, suppress_errors=True)


def postannotation_session_pipeline(
    keys: List[Dict] = None,
    tracking_method_name: str = "Easymocap",
    top_down_method_name: str = "Bridging_bml_movi_87",
    reconstruction_method_name: str = "Implicit Optimization KP Conf, MaxHuber=10",
):
    """
    Run the person reconstruction pipeline on the set of recordings

    Args:
        keys (List[Dict], optional): list of recording keys. Defaults to None, in which case it
            will run on all recordings that have not been reconstructed with the given method.
        tracking_method_name (str, optional): name of the tracking method. Defaults to "Easymocap".
        top_down_method_name (str, optional): name of the top down method. Defaults to "Bridging_bml_movi_87".
        reconstruction_method_name (str, optional): name of the reconstruction method. Defaults to "Implicit Optimization KP Conf, MaxHuber=10".
    """

    filt = PersonKeypointReconstructionMethodLookup * TopDownMethodLookup & {
        "top_down_method_name": top_down_method_name,
        "reconstruction_method_name": reconstruction_method_name,
    }

    annotated = CalibratedRecording & (
        Video * SingleCameraVideo * PersonBbox * TrackingBboxMethodLookup & {"tracking_method_name": tracking_method_name}
    )

    if keys is None:
        keys = (CalibratedRecording & Recording & annotated - (PersonKeypointReconstruction & filt)).fetch("KEY")

    reconstruction_pipeline(
        keys,
        top_down_method_name=top_down_method_name,
        reconstruction_method_name=reconstruction_method_name,
        reserve_jobs=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--post_annotation", action="store_true", help="Run post-annotation pipeline")
    parser.add_argument("--participant_id", type=int, help="Participant ID", required=False)
    args = parser.parse_args()

    if args.post_annotation:
        postannotation_session_pipeline()
        postannotation_session_pipeline(top_down_method_name="MMPoseHalpe")
    else:
        # assign_calibration()

        # if args.participant_id is set create a filter
        if args.participant_id:
            keys = (SingleCameraVideo & Recording - EasymocapSmpl & (Recording & f"participant_id IN ({args.participant_id})")).fetch("KEY")
            preannotation_session_pipeline(keys)
        else:
            keys = (SingleCameraVideo & Recording - EasymocapSmpl & (Recording & "participant_id NOT IN (72,73,504)")).fetch("KEY")
            preannotation_session_pipeline(keys)
        postannotation_session_pipeline()
        postannotation_session_pipeline(top_down_method_name="MMPoseHalpe")

    # assign_calibration()
    # keys = (SingleCameraVideo & Recording - EasymocapSmpl & (Recording & 'participant_id NOT IN (72,73,504)')).fetch('KEY')
    # keys = (SingleCameraVideo & Recording - EasymocapSmpl & (Recording & 'participant_id IN (505)')).fetch('KEY')
    # preannotation_session_pipeline(keys)
    # postannotation_session_pipeline()
    # postannotation_session_pipeline(top_down_method_name="MMPoseHalpe")
