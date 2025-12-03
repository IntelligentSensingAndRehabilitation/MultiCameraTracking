from typing import List, Dict
import datetime
import sys

import pose_pipeline
from pose_pipeline.pipeline import Video, VideoInfo, TopDownMethodLookup, PersonBbox, TrackingBboxMethodLookup
from pose_pipeline.utils.standard_pipelines import bottom_up_pipeline, top_down_pipeline
from multi_camera.datajoint.sessions import Session, Recording
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

def populate_session_calibration(keys: List[Dict] = None):
    """
    Populate the SessionCalibration table to link Recordings and Calibrations
    """
    from multi_camera.datajoint.sessions import Session
    from multi_camera.datajoint.session_calibrations import SessionCalibration

    if keys is None:
        keys = Session().fetch("KEY")

    SessionCalibration.populate(keys, suppress_errors=True, display_progress=True)

def preannotation_session_pipeline(keys: List[Dict] = None, bottom_up: bool = True, bridging: bool = True, easy_mocap: bool = False):
    """
    Perform the initial scene reconstruction for annotation

    This will run the bottom up pipeline and then EasyMocap reconstruction. It is
    possible to select between keypoints from MeTRabs versus OpenPose with the bridging
    flag

    Args:
        keys (List[Dict], optional): list of recording keys. Defaults to None, in which case it
            will run on all recordings that have not been annotated
        bottom_up (bool, optional): whether to run the bottom up pipeline. Defaults to True.
        bridging (bool, optional): whether to use the bridging keypoints. Defaults to True.
        easy_mocap (bool, optional): whether to run the EasyMocap pipeline. Defaults to False.

    """

    # Configure GPU memory limits only when GPU-intensive operations are needed
    if bottom_up:
        pose_pipeline.env.pytorch_memory_limit()
        pose_pipeline.env.tensorflow_memory_limit()
        pose_pipeline.env.jax_memory_limit()

    print("populating video info")
    VideoInfo.populate(SingleCameraVideo * MultiCameraRecording & keys, reserve_jobs=True)

    if keys is None:
        keys = (SingleCameraVideo & Recording - EasymocapSmpl).fetch("KEY")
        print("Computing initial reconstruction for {} videos".format(len(keys)))

    if bottom_up:
        if bridging:
            bottom_up_pipeline(keys, bottom_up_method_name="Bridging_OpenPose", reserve_jobs=True)
        else:
            bottom_up_pipeline(keys, bottom_up_method_name="OpenPose_HR", reserve_jobs=True)

    # now run easymocap
    if easy_mocap:
        print("populating easymocaptracking")
        EasymocapTracking.populate(MultiCameraRecording * CalibratedRecording & keys, reserve_jobs=True, suppress_errors=True)
        print("populating easymocapsmpl")
        EasymocapSmpl.populate(MultiCameraRecording * CalibratedRecording & keys, reserve_jobs=True, suppress_errors=True)


def postannotation_session_pipeline(
    keys: List[Dict],
    tracking_method_name: str = "Easymocap",
    top_down_method_name: str = "Bridging_bml_movi_87",
    reconstruction_method_name: str = "Robust Triangulation",
    hand_estimation: bool = False,
):
    """
    Run the person reconstruction pipeline on the set of recordings

    Args:
        keys (List[Dict]): Datajoint Table Object.
        tracking_method_name (str, optional): name of the tracking method. Defaults to "Easymocap".
        top_down_method_name (str, optional): name of the top down method. Defaults to "Bridging_bml_movi_87".
        reconstruction_method_name (str, optional): name of the reconstruction method. Defaults to "Implicit Optimization KP Conf, MaxHuber=10".
        hand_estimation (bool, optional): whether to include hand keypoints in the reconstruction. Defaults to False.
    """

    # Configure GPU memory limits for reconstruction pipeline
    pose_pipeline.env.pytorch_memory_limit()
    pose_pipeline.env.tensorflow_memory_limit()
    pose_pipeline.env.jax_memory_limit()

    filt = PersonKeypointReconstructionMethodLookup * TopDownMethodLookup & {
        "top_down_method_name": top_down_method_name,
        "reconstruction_method_name": reconstruction_method_name,
    }

    annotated = CalibratedRecording & (keys & {"tracking_method_name": tracking_method_name})

    not_reconstructed = (CalibratedRecording & Recording & annotated - (PersonKeypointReconstruction & filt)).fetch("KEY")

    reconstruction_pipeline(
        not_reconstructed,
        top_down_method_name=top_down_method_name,
        reconstruction_method_name=reconstruction_method_name,
        reserve_jobs=True,
        hand_estimation=hand_estimation,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant_id", type=str, default=None, help="Participant ID, comma-separated", required=False)
    parser.add_argument("--project", type=str, default=None, help="Video Project(s), comma-separated", required=False)
    parser.add_argument("--session_date", help="Session Date (YYYY-MM-DD)", required=False)
    parser.add_argument("--post_annotation", action="store_true", help="Run post-annotation pipeline")
    parser.add_argument("--bottom_up", action="store_true", help="Run bottom up pipeline")
    parser.add_argument("--run_easymocap", action="store_true", help="Run the EasyMocap steps")
    parser.add_argument("--top_down_method_name", type=str, default="Bridging_bml_movi_87", help="Top down method name")
    parser.add_argument("--reconstruction_method_name", type=str, default="Robust Triangulation", help="Reconstruction method name")
    parser.add_argument("--tracking_method_name", type=str, default="Easymocap", help="Tracking method name")
    parser.add_argument("--hand_estimation", action="store_true", help="Run hand estimation")
    args = parser.parse_args()

    post_annotation_args = {}
    if args.top_down_method_name:
        top_down = len(TopDownMethodLookup & {"top_down_method_name": args.top_down_method_name})
        assert top_down == 1, f"{top_down} matching records found in TopDownMethodLookup for: {args.top_down_method_name}"
        post_annotation_args["top_down_method_name"] = args.top_down_method_name

    if args.reconstruction_method_name:
        reconstruction = len(PersonKeypointReconstructionMethodLookup & {"reconstruction_method_name": args.reconstruction_method_name})
        assert reconstruction == 1, f"{reconstruction} matching records found in PersonKeypointReconstructionMethodLookup for: {args.reconstruction_method_name}"
        post_annotation_args["reconstruction_method_name"] = args.reconstruction_method_name

    if args.tracking_method_name:
        tracking = len(TrackingBboxMethodLookup & {"tracking_method_name": args.tracking_method_name})
        assert tracking == 1, f"{tracking} matching records found in TrackingBboxMethodLookup for: {args.tracking_method_name}"
        post_annotation_args["tracking_method_name"] = args.tracking_method_name

    if args.hand_estimation:
        post_annotation_args["hand_estimation"] = args.hand_estimation

    # create a filter for the recording table based on if participant_id and/or session_date is set
    filters = []
    session_filters = []

    if args.participant_id:
        participant_ids = [id.strip() for id in args.participant_id.split(",")]
        quoted_ids = ", ".join(f"'{id}'" for id in participant_ids)
        filters.append(f"participant_id IN ({quoted_ids})")
        session_filters.append(f"participant_id IN ({quoted_ids})")
    if args.project:
        projects = [proj.strip() for proj in args.project.split(",")]
        quoted_projects = ", ".join(f"'{proj}'" for proj in projects)
        filters.append(f"video_project IN ({quoted_projects})")
    if args.session_date:
        try:
            datetime.datetime.strptime(args.session_date, "%Y-%m-%d")
            filters.append(f"session_date LIKE '{args.session_date}%'")
            session_filters.append(f"session_date LIKE '{args.session_date}%'")
        except ValueError:
            print(f"Error: Invalid date format '{args.session_date}'. Use YYYY-MM-DD.")
            sys.exit(1)

    # Concatenate filters with AND
    filter_str = " AND ".join(filters) if filters else None
    session_filter_str = " AND ".join(session_filters) if session_filters else None

    if session_filter_str:
        session_keys = (Session & session_filter_str).fetch("KEY")
    else:
        session_keys = None

    populate_session_calibration(session_keys)

    if args.post_annotation:
        if filter_str:
            keys = Video * SingleCameraVideo * PersonBbox * TrackingBboxMethodLookup & (MultiCameraRecording * Recording & filter_str)
        else:
            keys = Video * SingleCameraVideo * PersonBbox * TrackingBboxMethodLookup

        post_annotation_args["keys"] = keys

        postannotation_session_pipeline(**post_annotation_args)
    else:
        if filter_str:
            keys = (SingleCameraVideo & Recording - EasymocapSmpl & (MultiCameraRecording * Recording & filter_str)).fetch("KEY")
        else:
            keys = (SingleCameraVideo & Recording - EasymocapSmpl & (MultiCameraRecording * Recording & "participant_id NOT IN (72,73,504)")).fetch("KEY")

        preannotation_session_pipeline(keys, easy_mocap=args.run_easymocap, bottom_up=args.bottom_up)