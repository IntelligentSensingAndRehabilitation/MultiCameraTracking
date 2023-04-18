from typing import List, Dict
from scipy.spatial.transform import Rotation as R
import numpy as np
import base64
import json

from pose_pipeline import TopDownMethodLookup
from multi_camera.datajoint.sessions import Recording
from multi_camera.datajoint.multi_camera_dj import SMPLReconstruction, SMPLXReconstruction, MultiCameraRecording


def get_method():
    method = {
        "tracking_method": 21,
        "reconstruction_method": 2,
        "top_down_method": 2,
    }

    return method


def get_smpl_trials(mode: str = "smpl") -> List[Dict]:
    """
    Returns a list of all the trials that have been reconstructed using the SMPLReconstruction pipeline.

    We could return a hierarchical list, but instead returning a flat list of dictionaries.
    """

    method = get_method()

    # sticking with this dirty but effective method that only ends up matching based on camera_config_hash and recording_timestamps
    if mode.upper() == "SMPL":
        reconstructed_trials = SMPLReconstruction & method
    else:
        reconstructed_trials = SMPLXReconstruction & method

    reconstructed_recordings = MultiCameraRecording * Recording & reconstructed_trials
    matches = reconstructed_recordings.fetch("participant_id", "session_date", "video_base_filename", as_dict=True)

    print("Returning matches: ", matches)
    return matches


def get_smpl_trajectory(filename: str, mode: str = "smpl") -> Dict:
    """
    Returns a list of all the trials that have been reconstructed using the SMPLReconstruction pipeline.

    We could return a hierarchical list, but instead returning a flat list of dictionaries.
    """

    method = get_method()

    filt = MultiCameraRecording * Recording & {"video_base_filename": filename}

    if mode.upper() == "SMPL":
        key = (SMPLReconstruction & filt & method).fetch1("KEY")
        faces, vertices = (SMPLReconstruction & key).fetch1("faces", "vertices")
    else:
        key = (SMPLXReconstruction & filt & method).fetch1("KEY")
        faces, vertices = (SMPLXReconstruction & key).fetch1("faces", "vertices")

    # only send every 5th frame and limited number
    vertices = vertices[::5]

    # scale up and convert to int to reduce bandwidth
    vertices = (vertices * 1000).astype(int).tolist()

    # for consistency with other API convert this to a list of list of dictionaries
    # with the same subject ID
    vertices = [[{"id": 0, "verts": v}] for v in vertices]

    vertices = base64.b64encode(json.dumps(vertices).encode("utf-8")).decode("utf-8")

    return {"faces": faces.tolist(), "vertices": vertices}
