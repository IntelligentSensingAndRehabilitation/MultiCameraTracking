from typing import List, Dict
from scipy.spatial.transform import Rotation as R
import numpy as np

from pose_pipeline import TopDownMethodLookup
from multi_camera.datajoint.sessions import Recording
from multi_camera.datajoint.multi_camera_dj import PersonKeypointReconstructionMethodLookup, MultiCameraRecording
from multi_camera.validation.biomechanics.biomechanics import BiomechanicalReconstruction
from multi_camera.analysis.biomechanics import bilevel_optimization
from multi_camera.analysis.biomechanics.render import load_skeleton_meshes


def get_method():
    """Returns the method used for biomechanical reconstruction."""
    reconstruction_method = (
        PersonKeypointReconstructionMethodLookup
        & 'reconstruction_method_name="Implicit Optimization KP Conf, MaxHuber=10"'
    ).fetch1("reconstruction_method")

    top_down_method = (TopDownMethodLookup & 'top_down_method_name="Bridging_bml_movi_87"').fetch1("top_down_method")

    method = {
        "model_name": "Rajagopal_Neck_mbl_movi_87_rev15",
        "bilevel_settings": 19,
        "reconstruction_method": reconstruction_method,
        "top_down_method": top_down_method,
    }

    return method


def get_biomechanics_trials() -> List[Dict]:
    """
    Returns a list of all the trials that have been reconstructed using the biomechanics pipeline.

    We could return a hierarchical list, but instead returning a flat list of dictionaries.
    """

    method = get_method()

    # sticking with this dirty but effective method that only ends up matching based on camera_config_hash and recording_timestamps
    reconstructed_trials = BiomechanicalReconstruction.Trial & method

    reconstructed_recordings = MultiCameraRecording * Recording & reconstructed_trials
    matches = reconstructed_recordings.fetch("participant_id", "session_date", "video_base_filename", as_dict=True)

    return matches


def getBiomechanicalMesh(skeleton):
    meshes = load_skeleton_meshes(skeleton)
    meshes_data = {}
    for name, mesh in meshes.items():
        # MeshData
        meshes_data[name] = {
            "vertices": (mesh.vertices).tolist(),
            "faces": mesh.faces.astype(int).tolist(),
        }
    return meshes_data


def getBiomechanicalTrajectory(skeleton, poses):
    trajectories = {}

    # Create a transformation matrix that swaps Y and Z axes
    r = np.pi / 2
    swap_yz_matrix = np.array([[1, 0, 0, 0], [0, np.cos(r), -np.sin(r), 0], [0, np.sin(r), np.cos(r), 0], [0, 0, 0, 1]])

    for pose in poses:
        skeleton.setPositions(pose)

        for b in skeleton.getBodyNodes():
            n = b.getNumShapeNodes()
            for i in range(n):
                s = b.getShapeNode(i)
                name = s.getName()
                transform_matrix = s.getWorldTransform().matrix()

                # Apply the affine transformation to the world transform matrix
                transformed_matrix = swap_yz_matrix @ transform_matrix

                position = transformed_matrix[:3, 3].tolist()

                rot_matrix = transformed_matrix[:3, :3]
                rotation = R.from_matrix(rot_matrix).as_quat().tolist()

                if name not in trajectories:
                    trajectories[name] = {
                        "position": [],
                        "orientation": [],
                    }

                trajectories[name]["position"].append(position)
                trajectories[name]["orientation"].append(rotation)

    trajectories = {
        # TrajectoryData
        k: {"positions": v["position"], "rotations": v["orientation"]}
        for k, v in trajectories.items()
    }
    return trajectories


def get_biomechanics_trajectory(filename: str):
    """Returns the bone meshes and trajectory for a given trial."""

    method = get_method()
    key = (
        BiomechanicalReconstruction.Trial
        & (MultiCameraRecording * Recording & {"video_base_filename": filename})
        & method
    ).fetch1("KEY")

    model_name, skeleton_def = (BiomechanicalReconstruction & key).fetch1("model_name", "skeleton_definition")

    skeleton = bilevel_optimization.reload_skeleton(model_name, skeleton_def["group_scales"])
    timestamps, poses = (BiomechanicalReconstruction.Trial & key).fetch1("timestamps", "poses")

    meshes = getBiomechanicalMesh(skeleton)
    traj = getBiomechanicalTrajectory(skeleton, poses)

    return {"meshes": meshes, "trajectories": traj, "timestamps": timestamps.tolist()}
