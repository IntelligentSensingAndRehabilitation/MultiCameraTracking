"""
Performs biomechanical analysis on 3D reconstructed keypoints. The results
are stored in DataJoint.

Currently using the GaitRite session to organize the groups of trials to
fit together. Ultimately, need to have a general session schema for the
multitrial reconstruction.
"""

import datajoint as dj
from multi_camera.datajoint.gaitrite_comparison import GaitRiteSession, GaitRiteRecording
from multi_camera.datajoint.multi_camera_dj import PersonKeypointReconstruction

schema = dj.schema("multicamera_tracking_biomechanics")


@schema
class BiomechanicalReconstructionLookup(dj.Lookup):
    definition = """
    top_down_method         : int 
    reconstruction_method   : int
    model_name              : varchar(255)
    """
    contents = [(2, 2, "Rajagopal2015_Halpe")]


@schema
class BiomechanicalReconstruction(dj.Computed):
    definition = """
    -> GaitRiteSession
    -> BiomechanicalReconstructionLookup
    ---
    skeleton_definition     : longblob
    """

    class Trial(dj.Part):
        definition = """
        -> BiomechanicalReconstruction
        -> GaitRiteRecording
        -> PersonKeypointReconstruction
        ---
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

        from ..analysis.biomechanics import bilevel_optimization

        trials = (PersonKeypointReconstruction * GaitRiteRecording & key).fetch("KEY")
        assert len(GaitRiteRecording & trials) == len(trials), "Not all trials have been reconstructed"
        assert len(trials) < 20, "WTF"

        print(f"Processing {len(trials)} trials with key: {key}")

        kps = [bilevel_optimization.fetch_formatted_markers(k) for k in trials]
        results, skeleton = bilevel_optimization.fit_markers(kps, key["model_name"])

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

        skeleton_defintion = {
            "marker_offsets_map": marker_offsets_map,
            "marker_offsets": results[0].markerOffsets,
            "group_scales": results[0].groupScales,
            "body_scale_map": body_scale_map,
        }
        self.insert1({"skeleton_definition": skeleton_defintion, **key})

        for t, r, kp in zip(trials, results, kps):
            t = (PersonKeypointReconstruction & t).fetch1("KEY")
            t.update(key)
            trial_key = t.copy()

            t["poses"] = r.poses.T
            t["joint_centers"] = r.jointCenters.reshape(-1, 3, r.jointCenters.shape[-1]).T

            metrics = bilevel_optimization.get_trial_performance(r, kp, skeleton)
            t["average_rmse"] = metrics["averageRootMeanSquaredError"]
            t["average_max_error"] = metrics["averageMaxError"]

            print(metrics["getSortedMarkerRMSE"])

            self.Trial.insert1(t)
            for joint_name, rmse in metrics["getSortedMarkerRMSE"]:
                self.JointRmse.insert1({"joint_name": joint_name, "rmse": rmse, **trial_key})

    def export(self, output_dir):
        import numpy as np
        from ..analysis.biomechanics import bilevel_optimization
        from .multi_camera_dj import SingleCameraVideo, MultiCameraRecording
        from pose_pipeline import VideoInfo

        assert (len(self)) == 1, "Filter for single object"

        model_name, skeleton_def = self.fetch1("model_name", "skeleton_definition")
        skeleton = bilevel_optimization.reload_skeleton(
            model_name, skeleton_def["group_scales"], skeleton_def["marker_offsets"]
        )
        bilevel_optimization.save_model(model_name, skeleton_def, output_dir)

        for key in (self.Trial & self).fetch("KEY"):
            kp = bilevel_optimization.fetch_formatted_markers(key)
            trial = (MultiCameraRecording & key).fetch1("video_base_filename")
            poses = (self.Trial & key).fetch1("poses")

            timestamps = (VideoInfo * SingleCameraVideo & key).fetch("timestamps")[0]
            timestamps = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
            timestamps = timestamps[: len(kp)]

            bilevel_optimization.save_trial(poses, skeleton, kp, timestamps, trial, output_dir)


if __name__ == "__main__":
    import multi_camera.datajoint.biomechanics
    from multi_camera.datajoint.biomechanics import BiomechanicalReconstruction

    BiomechanicalReconstruction.populate("subject_id=108")
