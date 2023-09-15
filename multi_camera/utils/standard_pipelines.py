from typing import List, Dict

from pose_pipeline import (
    PersonBbox,
    Video,
    TopDownMethodLookup,
    TrackingBboxMethodLookup,
    OpenPosePerson,
)
from ..datajoint.multi_camera_dj import (
    SingleCameraVideo,
    PersonKeypointReconstruction,
    PersonKeypointReconstructionMethod,
    PersonKeypointReconstructionMethodLookup,
)


def reconstruction_pipeline(
    keys: List[Dict],
    top_down_method_name: str = "MMPoseHalpe",
    tracking_method_name: str = "EasyMocap",
    reconstruction_method_name: str = "Robust Triangulation",
    reserve_jobs: bool = True,
):
    from pose_pipeline.utils import standard_pipelines as pose_pipelines

    if type(keys) == dict:
        keys = [keys]

    top_down_method = (TopDownMethodLookup & f'top_down_method_name="{top_down_method_name}"').fetch1("top_down_method")
    tracking_method = (TrackingBboxMethodLookup & f'tracking_method_name="{tracking_method_name}"').fetch1(
        "tracking_method"
    )

    r = {"reconstruction_method_name": reconstruction_method_name}
    assert (
        len(PersonKeypointReconstructionMethodLookup & r) == 1
    ), f"Unable to find reconstruction method {reconstruction_method_name}"
    reconstruction_method = (PersonKeypointReconstructionMethodLookup & r).fetch1("reconstruction_method")

    final_keys = []

    for k in keys:
        k["tracking_method"] = tracking_method

        video_keys = (PersonBbox & (SingleCameraVideo & k)).fetch("KEY")
        print(f"Processing {len(video_keys)} videos with key: {k}")

        for v in video_keys:
            v = (Video & v).fetch1("KEY")
            if top_down_method_name == "OpenPose":
                from pose_pipeline import OpenPosePerson

                OpenPosePerson.populate(v)
            if top_down_method_name in [
                "OpenPose",
                "OpenPose_LR",
                "OpenPose_HR",
                "Bridging_COCO_25",
                "Bridging_bml_movi_87",
            ]:
                pose_pipelines.bottomup_to_topdown(
                    [v], top_down_method_name, tracking_method_name, reserve_jobs=reserve_jobs
                )
            else:
                pose_pipelines.top_down_pipeline(
                    v,
                    tracking_method_name=tracking_method_name,
                    top_down_method_name=top_down_method_name,
                    reserve_jobs=reserve_jobs,
                )

        k["reconstruction_method"] = reconstruction_method
        k["top_down_method"] = top_down_method
        k["tracking_method"] = tracking_method
        PersonKeypointReconstructionMethod.insert1(k, skip_duplicates=True)
        PersonKeypointReconstruction.populate(k, suppress_errors=False, reserve_jobs=reserve_jobs)

        final_keys.append(k)

    return final_keys
