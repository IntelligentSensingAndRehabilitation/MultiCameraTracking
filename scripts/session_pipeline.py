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


def preannotation_session_pipeline(keys: List[Dict] = None, bridging: bool = True, easy_mocap: bool = False):
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
    print("populating video info")
    VideoInfo.populate(SingleCameraVideo * MultiCameraRecording & keys, reserve_jobs=True)

    if easy_mocap:
        print("populating easymocaptracking")
        EasymocapTracking.populate(MultiCameraRecording * CalibratedRecording & keys, reserve_jobs=True, suppress_errors=True)
        print("populating easymocapsmpl")
        EasymocapSmpl.populate(MultiCameraRecording * CalibratedRecording & keys, reserve_jobs=True, suppress_errors=True)


def postannotation_session_pipeline(
    keys: List[Dict] = None,
    tracking_method_name: str = "Easymocap",
    top_down_method_name: str = "Bridging_bml_movi_87",
    reconstruction_method_name: str = "Robust Triangulation",
    date_filter: str = None,
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

    if date_filter:
        annotated = annotated & (f"recording_timestamps LIKE '{date_filter}%'")

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
    parser.add_argument("--participant_id", type=str, default=None, help="Participant ID", required=False)
    parser.add_argument("--project", type=str, default=None, help="Video Project", required=False)
    parser.add_argument("--session_date", help="Session Date (YYYY-MM-DD)", required=False)
    parser.add_argument("--post_annotation", action="store_true", help="Run post-annotation pipeline")
    parser.add_argument("--run_easymocap", action="store_true", help="Run the EasyMocap steps")
    args = parser.parse_args()

    if args.post_annotation:
        if args.session_date:
            print("Session Date: ", args.session_date)
            postannotation_session_pipeline(date_filter=args.session_date)
        else:
            postannotation_session_pipeline()
    else:
        # assign_calibration()

        # create a filter for the recording table based on if participant_id and/or session_date is set
        passed_args = []
        if args.participant_id:
            passed_args.append(f"participant_id IN ({args.participant_id})")
        if args.project:
            passed_args.append(f"video_project IN ({args.project})")
        if args.session_date:
            passed_args.append(f"recording_timestamps LIKE '{args.session_date}%'")

        if len(passed_args) > 1:
            # concatenate the filters with an AND
            filter = " AND ".join(passed_args)
        elif len(passed_args) == 1:
            filter = passed_args[0]
        else:
            filter = None

        if filter:
            keys = (SingleCameraVideo & Recording - EasymocapSmpl & (MultiCameraRecording * Recording & filter)).fetch("KEY")
        else:
            keys = (SingleCameraVideo & Recording - EasymocapSmpl & (MultiCameraRecording * Recording & "participant_id NOT IN (72,73,504)")).fetch("KEY")

        preannotation_session_pipeline(keys, easy_mocap=args.run_easymocap)