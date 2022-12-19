from typing import List, Dict

from pose_pipeline import (
    PersonBbox,
    TopDownPerson,
    TopDownMethod,
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
    reconstruction_method_name: str = "RobustTriangulation",
):

    if type(keys) == dict:
        keys = [keys]

    top_down_method = (TopDownMethodLookup & f'top_down_method_name="{top_down_method_name}"').fetch1("top_down_method")
    tracking_method = (TrackingBboxMethodLookup & f'tracking_method_name="{tracking_method_name}"').fetch1(
        "tracking_method"
    )
    reconstruction_method = (
        PersonKeypointReconstructionMethodLookup & f'reconstruction_method_name="{reconstruction_method_name}"'
    ).fetch1("reconstruction_method")

    final_keys = []

    for k in keys:

        k["tracking_method"] = tracking_method

        video_keys = (PersonBbox & (SingleCameraVideo & k)).fetch("KEY")
        print(f"Processing {len(video_keys)} videos with key: {k}")

        for v in video_keys:
            v = {"top_down_method": top_down_method, **v}
            TopDownMethod.insert1(v, skip_duplicates=True)
            if top_down_method_name == "OpenPose":
                OpenPosePerson.populate(v, suppress_errors=True, reserve_jobs=True)
            TopDownPerson.populate(v, suppress_errors=True, reserve_jobs=True)

        k["reconstruction_method"] = reconstruction_method
        k["top_down_method"] = top_down_method
        k["tracking_method"] = tracking_method
        PersonKeypointReconstructionMethod.insert1(k, skip_duplicates=True)
        PersonKeypointReconstruction.populate(k)  # , suppress_errors=True, reserve_jobs=True)

        final_keys.append(k)

    return final_keys
