"""
Performs biomechanical analysis on 3D reconstructed keypoints. The results
are stored in DataJoint.
"""

import numpy as np
import datajoint as dj

from pose_pipeline.pipeline import TopDownMethodLookup
from multi_camera.datajoint import sessions
from multi_camera.datajoint.multi_camera_dj import (
    PersonKeypointReconstruction,
    PersonKeypointReconstructionMethodLookup,
)

# TODO: Ugh! for some reason we have to keep this in the same schema or get a foreign key error
schema = dj.schema("mocap_sessions")
# schema = dj.schema("multicamera_tracking_nimblephysics")


reconstruction_settings = {
    "reconstruction_method_name": "Implicit Optimization KP Conf, MaxHuber=10",
    "top_down_method_name": "Bridging_bml_movi_87",
    "model_name": "Rajagopal_Neck_mbl_movi_87_rev15",
    "max_marker_offset": 0.05,
    "regularize_all_body_scales": 0.2,
    "regularize_anatomical_marker_offsets": 100,
    "regularize_tracking_marker_offsets": 10,
    "anthropomorphic_prior": 2,
    "regularize_joint_bounds": 5,
    "set_min_sphere_fit_score": 0.01,
    "set_min_axis_fit_score": 0.001,
    "set_max_joint_weight": 1,
}


def get_height(key):
    """
    Get the height of the subject in meters

    Maps from participant_id to subject_id and pulls from the FirebaseSessions schema,
    which is rather inelegant but works for now
    """

    from sensor_fusion.emgimu_session import Height
    from multi_camera.datajoint.sessions import get_subject_id_from_participant_id

    subject_id = get_subject_id_from_participant_id(key["participant_id"])
    height = (Height & {"subject_id": subject_id}).fetch1("height_mm") / 1000.0

    return height


@schema
class BiomechanicalReconstruction(dj.Computed):
    definition = """
    -> sessions.Session
    ---
    skeleton_definition     : longblob
    """

    class Trial(dj.Part):
        definition = """
        -> BiomechanicalReconstruction
        -> PersonKeypointReconstruction
        -> sessions.Recording
        ---
        timestamps             : longblob
        poses                  : longblob
        joint_centers          : longblob
        average_rmse           : float
        average_max_error      : float
        """

    class JointRmse(dj.Part):
        definition = """
        -> BiomechanicalReconstruction.Trial
        joint_name                : varchar(50)
        ---
        rmse                      : float
        """

    def make(self, key):
        from multi_camera.analysis.biomechanics import bilevel_optimization

        trials = (
            PersonKeypointReconstruction
            * sessions.Recording
            * PersonKeypointReconstructionMethodLookup
            * TopDownMethodLookup
            & key
            & reconstruction_settings
        ).fetch("KEY")

        print(trials, len(trials))
        assert len(sessions.Recording & trials) == len(trials), "Not all trials have been reconstructed"
        assert len(trials) < 20, "WTF"
        if len(trials) == 0:
            print("No trials to process. Skipping" + str(key))
            return
        assert len(trials) > 0, "No trials to process"

        height = get_height(key)
        print(f"Processing {len(trials)} trials with key: {key}")

        use_augmenter = "Augmenter" in reconstruction_settings["model_name"]

        kps = []
        trial_timestamps = []
        for k in trials:
            import numpy as np
            from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording

            kp = bilevel_optimization.fetch_formatted_markers(k, augmenter=use_augmenter)
            dt = (MultiCameraRecording & k).fetch_timestamps()

            kps.append(kp)
            trial_timestamps.append(dt)

        # kps = [bilevel_optimization.fetch_formatted_markers(k) for k in trials]
        results, skeleton = bilevel_optimization.fit_markers(
            kps,
            reconstruction_settings["model_name"],
            max_marker_offset=reconstruction_settings["max_marker_offset"],
            regularize_all_body_scales=reconstruction_settings["regularize_all_body_scales"],
            regularize_anatomical_marker_offsets=reconstruction_settings["regularize_anatomical_marker_offsets"],
            regularize_tracking_marker_offsets=reconstruction_settings["regularize_tracking_marker_offsets"],
            anthropomorphic_prior=reconstruction_settings["anthropomorphic_prior"],
            regularize_joint_bounds=reconstruction_settings["regularize_joint_bounds"],
            set_min_sphere_fit_score=reconstruction_settings["set_min_sphere_fit_score"],
            set_min_axis_fit_score=reconstruction_settings["set_min_axis_fit_score"],
            set_max_joint_weight=reconstruction_settings["set_max_joint_weight"],
            heightM=height,
        )

        print(f"Received {len(results)} results from {len(kps)} trials")

        body_scale_map = {}
        for i in range(skeleton.getNumBodyNodes()):
            bodyNode = skeleton.getBodyNode(i)
            # Now that we adjust the markers BEFORE we rescale the body, we don't want to rescale the marker locations at all
            body_scale_map[bodyNode.getName()] = [
                1.0 / bodyNode.getScale()[0],
                1.0 / bodyNode.getScale()[1],
                1.0 / bodyNode.getScale()[2],
            ]

        fitMarkers = results[0].updatedMarkerMap
        marker_offsets_map = {}
        for k in fitMarkers:
            v = fitMarkers[k]
            marker_offsets_map[k] = (v[0].getName(), v[1])
        del fitMarkers

        skeleton_defintion = {
            "marker_offsets_map": marker_offsets_map,
            "marker_offsets": results[0].markerOffsets,
            "group_scales": skeleton.getGroupScales(),
            "body_scale_map": body_scale_map,
        }
        self.insert1({"skeleton_definition": skeleton_defintion, **key})

        for t, r, kp, dt in zip(trials, results, kps, trial_timestamps):
            t = (PersonKeypointReconstruction & t).fetch1("KEY")
            t.update(key)
            trial_key = t.copy()

            t["timestamps"] = dt
            t["poses"] = r.poses.T
            t["joint_centers"] = r.jointCenters.reshape(-1, 3, r.jointCenters.shape[-1]).T

            metrics = bilevel_optimization.get_trial_performance(r, kp, skeleton)
            t["average_rmse"] = metrics["averageRootMeanSquaredError"]
            t["average_max_error"] = metrics["averageMaxError"]

            print(metrics["getSortedMarkerRMSE"])

            self.Trial.insert1(t)
            for joint_name, rmse in metrics["getSortedMarkerRMSE"]:
                self.JointRmse.insert1({"joint_name": joint_name, "rmse": rmse, **trial_key})


if __name__ == "__main__":
    BiomechanicalReconstruction.populate('participant_id="TF01"')
