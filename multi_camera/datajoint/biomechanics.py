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
    bilevel_settings        : int
    ---
    max_marker_offset                    : float
    regularize_all_body_scales           : float
    regularize_anatomical_marker_offsets : float
    anthropomorphic_prior                : float
    regularize_joint_bounds              : float
    """
    contents = [
        # (2, 0, "Rajagopal2015_Halpe"),
        # (2, 1, "Rajagopal2015_Halpe"),
        # (2, 2, "Rajagopal2015_Halpe"),
        # (2, 0, "Rajagopal2015_Augmenter"),
        # (2, 1, "Rajagopal2015_Augmenter"),
        # (2, 2, "Rajagopal2015_Augmenter"),
        # (4, 0, "Rajagopal2015_Augmenter"),
        # (4, 2, "Rajagopal2015_Augmenter"),
        # last two should remain zero for now
        (2, 2, "Rajagopal2015_Halpe", 0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (2, 2, "Rajagopal2015_Halpe", 1, 0.1, 0.1, 10.0, 0.0, 0.0),
        (2, 2, "Rajagopal2015_Halpe", 2, 0.05, 1.0, 50.0, 0.0, 0.0),
    ]


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

        from ..analysis.biomechanics import bilevel_optimization

        trials = (PersonKeypointReconstruction * GaitRiteRecording & key).fetch("KEY")
        assert len(GaitRiteRecording & trials) == len(trials), "Not all trials have been reconstructed"
        assert len(trials) < 20, "WTF"

        print(f"Processing {len(trials)} trials with key: {key}")

        kps = []
        trial_timestamps = []
        for k in trials:
            import numpy as np
            from .multi_camera_dj import MultiCameraRecording
            from .gaitrite_comparison import get_walking_time_range

            k2 = k.copy()
            k2["reconstruction_method"] = 0
            k2["top_down_method"] = 2
            trange = get_walking_time_range(k2)

            kp = bilevel_optimization.fetch_formatted_markers(k, augmenter="Augmenter" in key["model_name"])
            dt = (MultiCameraRecording & k).fetch_timestamps()

            valid = np.logical_and(dt >= trange[0], dt <= trange[1])
            kp = [kp for kp, v in zip(kp, valid) if v]
            dt = dt[valid]

            kps.append(kp)
            trial_timestamps.append(dt)

        settings = (BiomechanicalReconstructionLookup & key).fetch1()

        # kps = [bilevel_optimization.fetch_formatted_markers(k) for k in trials]
        results, skeleton = bilevel_optimization.fit_markers(
            kps,
            key["model_name"],
            max_marker_offset=settings["max_marker_offset"],
            regularize_all_body_scales=settings["regularize_all_body_scales"],
            regularize_anatomical_marker_offsets=settings["regularize_anatomical_marker_offsets"],
            anthropomorphic_prior=settings["anthropomorphic_prior"],
            regularize_joint_bounds=settings["regularize_joint_bounds"],
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

        skeleton_defintion = {
            "marker_offsets_map": marker_offsets_map,
            "marker_offsets": results[0].markerOffsets,
            "group_scales": results[0].groupScales,
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
            from .gaitrite_comparison import get_walking_time_range

            trial = (MultiCameraRecording & key).fetch1("video_base_filename")
            poses = (self.Trial & key).fetch1("poses")
            timestamps = (self.Trial & key).fetch1("timestamps")

            # export the same time range to match the fitting filter
            key = key.copy()
            key["reconstruction_method"] = 0
            trange = get_walking_time_range(key)

            kp = bilevel_optimization.fetch_formatted_markers(key)
            dt = (MultiCameraRecording & key).fetch_timestamps()
            valid = np.logical_and(dt >= trange[0], dt <= trange[1])
            kp = [kp for kp, v in zip(kp, valid) if v]

            bilevel_optimization.save_trial(poses, skeleton, kp, timestamps, trial, output_dir)

            if "Augmenter" in model_name:
                print(f"here {key}")
                import nimblephysics as nimble
                import os

                kp = bilevel_optimization.fetch_formatted_markers(
                    (PersonKeypointReconstruction & key).fetch1("KEY"), augmenter=True
                )
                kp = [kp for kp, v in zip(kp, valid) if v]
                nimble.biomechanics.OpenSimParser.saveTRC(
                    os.path.join(output_dir, "MarkerData/" + trial + "_augmenter.trc"),
                    timestamps,
                    kp,
                )


if __name__ == "__main__":
    import multi_camera.datajoint.biomechanics
    from multi_camera.datajoint.biomechanics import BiomechanicalReconstruction

    BiomechanicalReconstruction.populate("subject_id=136 and model_name='Rajagopal2015_Halpe'")
