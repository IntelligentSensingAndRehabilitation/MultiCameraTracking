"""
Performs biomechanical analysis on 3D reconstructed keypoints. The results
are stored in DataJoint.

Currently using the GaitRite session to organize the groups of trials to
fit together. Ultimately, need to have a general session schema for the
multitrial reconstruction.
"""

import numpy as np
import datajoint as dj
from multi_camera.datajoint.gaitrite_comparison import GaitRiteSession, GaitRiteRecording
from multi_camera.datajoint.multi_camera_dj import PersonKeypointReconstruction,  SingleCameraVideo

schema = dj.schema("multicamera_tracking_biomechanics")


@schema
class BilevelLookup(dj.Lookup):
    definition = """
    bilevel_settings        : int
    ---
    max_marker_offset                    : float
    regularize_all_body_scales           : float
    regularize_anatomical_marker_offsets : float
    regularize_tracking_marker_offsets   : float
    anthropomorphic_prior                : float
    regularize_joint_bounds              : float
    set_min_sphere_fit_score             : float
    set_min_axis_fit_score               : float
    set_max_joint_weight                 : float
    """
    contents = [
        (0, 0.05, 0.0, 10.0, 0.05, 0.1, 0.01, 0.01, 0.001, 1.0),
    ]


@schema
class BiomechanicalReconstructionLookup(dj.Lookup):
    definition = """
    -> BilevelLookup
    top_down_method         : int 
    reconstruction_method   : int
    model_name              : varchar(255)
    ---
    """
    contents = [
        (0, 12, 0, "Rajagopal_mbl_movi_87"),
        (0, 12, 1, "Rajagopal_mbl_movi_87"),
        (0, 12, 2, "Rajagopal_mbl_movi_87"),
        (0, 12, 3, "Rajagopal_mbl_movi_87"),
        (0, 12, 11, "Rajagopal_mbl_movi_87"),
    ]
    # (2, 0, "Rajagopal2015_Halpe", 0),),
    # (2, 1, "Rajagopal2015_Halpe", 0),),
    # (2, 2, "Rajagopal2015_Halpe", 0),),
    # (2, 0, "Rajagopal2015_Augmenter", 0),),
    # (2, 1, "Rajagopal2015_Augmenter", 0),),
    # (2, 2, "Rajagopal2015_Augmenter", 0),),
    # (4, 0, "Rajagopal2015_Augmenter", 0),),
    # (4, 2, "Rajagopal2015_Augmenter", 0),),
    # (12, 0, "Rajagopal_mbl_movi_87", 0),
    # (#12, 1, "Rajagopal_mbl_movi_87", 0),
    # (#12, 2, "Rajagopal_mbl_movi_87", 0),
    # (#12, 3, "Rajagopal_mbl_movi_87", 0),
    # (#12, 11, "Rajagopal_mbl_movi_87", 0),
    # ]


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

        if key['bilevel_settings'] > 1:
            from sensor_fusion.emgimu_session import Height
            height = (Height & key).fetch1('height_mm') / 1000.0
        else:
            height = 1.7

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

        settings = (BilevelLookup & key).fetch1()

        # kps = [bilevel_optimization.fetch_formatted_markers(k) for k in trials]
        results, skeleton = bilevel_optimization.fit_markers(
            kps,
            key["model_name"],
            max_marker_offset=settings["max_marker_offset"],
            regularize_all_body_scales=settings["regularize_all_body_scales"],
            regularize_anatomical_marker_offsets=settings["regularize_anatomical_marker_offsets"],
            regularize_tracking_marker_offsets=settings["regularize_tracking_marker_offsets"],
            anthropomorphic_prior=settings["anthropomorphic_prior"],
            regularize_joint_bounds=settings["regularize_joint_bounds"],
            set_min_sphere_fit_score=settings["set_min_sphere_fit_score"],
            set_min_axis_fit_score=settings["set_min_axis_fit_score"],
            set_max_joint_weight=settings["set_max_joint_weight"],
            heightM=height
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

    def export(self, output_dir):
        import numpy as np
        from ..analysis.biomechanics import bilevel_optimization
        from .multi_camera_dj import SingleCameraVideo, MultiCameraRecording
        from pose_pipeline import VideoInfo

        assert (len(self)) == 1, "Filter for single object"

        model_name, skeleton_def = self.fetch1("model_name", "skeleton_definition")
        skeleton = bilevel_optimization.reload_skeleton(model_name, skeleton_def["group_scales"])
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

    def find_incomplete(self):
        """
        Find all the sessions that are missing some of the reconstruction methods

        This shouldn't have occurred but was older code
        """

        incomplete = BiomechanicalReconstruction & (GaitRiteRecording * self - (BiomechanicalReconstruction.Trial * GaitRiteRecording * self).proj()).proj()
        return incomplete

    @property
    def key_source(self):
        """Only calibrate if all the reconstruction methods are computed"""
        possible = GaitRiteSession * BiomechanicalReconstructionLookup
        return possible - (possible - PersonKeypointReconstruction * possible).proj()


@schema
class BiomechanicsGaitCycles(dj.Computed):
    definition = """
    -> BiomechanicalReconstruction.Trial
    ---
    left_gait_cycles  : longblob
    right_gait_cycles : longblob
    """

    def make(self, key):
        from multi_camera.datajoint.gaitrite_comparison import fetch_data, GaitRiteRecordingAlignment
        from jax import vmap, numpy as jnp

        # fetch data
        timestamps, poses = (BiomechanicalReconstruction.Trial & key).fetch1('timestamps', 'poses')
        _, _, df = fetch_data(key)

        t_offset = (GaitRiteRecordingAlignment & key).fetch1("t_offset")

        # get gait event times
        left = df.loc[df['Left Foot']]
        left_times = jnp.stack([left.iloc[:-1]['First Contact Time'].values, left.iloc[1:]['First Contact Time'].values], axis=1)
        left_times = left_times + t_offset

        right = df.loc[~df['Left Foot']]
        right_times = jnp.stack([right.iloc[:-1]['First Contact Time'].values, right.iloc[1:]['First Contact Time'].values], axis=1)
        right_times = right_times + t_offset

        # interpolate poses to gait cycles
        interp_array = vmap(jnp.interp, (None, None, 1), 1)
        interp_windows = vmap(lambda t: interp_array(jnp.linspace(t[0], t[1], 100), timestamps, poses))

        left_cycles = interp_windows(left_times)
        right_cycles = interp_windows(right_times)

        key['left_gait_cycles'] = np.array(left_cycles)
        key['right_gait_cycles'] = np.array(right_cycles)
        self.insert1(key)


@schema
class BiomechanicalTrialMeshOverlay(dj.Computed):
    definition = """
    -> BiomechanicalReconstruction.Trial
    -> SingleCameraVideo
    ---
    output_video      : attach@localattach    # datajoint managed video file
    """

    def make(self, key):
        from multi_camera.datajoint.calibrate_cameras import Calibration
        from multi_camera.analysis.biomechanics.render import create_overlay_video

        camera_name = (SingleCameraVideo & key).fetch1("camera_name")
        camera_names = (Calibration & key).fetch1("camera_names")
        N = camera_names.index(camera_name)
        print(f"Creating overlay for {camera_name} ({N})")
        
        vid = create_overlay_video((BiomechanicalReconstruction.Trial & key).fetch1('KEY'), N)

        key['output_video'] = vid
        self.insert1(key)


@schema
class BiomechanicalReconstructionTrialNoise(dj.Computed):
    definition = """
    -> BiomechanicalReconstruction.Trial
    ---
    pose_noise:  float
    """

    def make(self, key):
        
        poses = (BiomechanicalReconstruction.Trial & key).fetch1('poses')
        poses = np.unwrap(poses, axis=0)

        key['pose_noise'] = np.sqrt(np.mean(np.diff(poses, axis=0) ** 2))
        self.insert1(key)


@schema
class BiomechanicalReconstructionReprojectionQuality(dj.Computed):
    definition = """
    -> BiomechanicalReconstruction.Trial
    ---
    reprojection_pck_5       : float
    reprojection_pck_10      : float
    reprojection_pck_20      : float
    reprojection_pck_50      : float
    reprojection_pck_100     : float
    reprojection_metrics     : longblob
    """

    def make(self, key):

        from pose_pipeline.pipeline import VideoInfo, TopDownPerson, TopDownMethodLookup
        from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording, SingleCameraVideo, PersonKeypointReconstruction
        from multi_camera.datajoint.calibrate_cameras import Calibration
        from multi_camera.datajoint.biomechanics import BiomechanicalReconstruction
        from multi_camera.analysis.biomechanics.opensim import normalize_marker_names
        from multi_camera.analysis.biomechanics.bilevel_optimization import get_markers, reload_skeleton
        from multi_camera.analysis import fit_quality

        # set up the cameras
        camera_params, camera_names = (Calibration & key).fetch1("camera_calibration", "camera_names")

        # find the videos (to get the height and width)
        videos = (TopDownPerson * MultiCameraRecording * PersonKeypointReconstruction * SingleCameraVideo & key).proj()
        video_keys, video_camera_name = (TopDownPerson * SingleCameraVideo * videos).fetch( "KEY", "camera_name")
        assert camera_names == video_camera_name.tolist(), "Videos don't match cameras in calibration"

        fps = np.unique((VideoInfo & video_keys).fetch("fps"))[0]

        # load the skeleton
        model_name, skeleton_def = (BiomechanicalReconstruction & key).fetch1('model_name', 'skeleton_definition')
        skeleton = reload_skeleton(model_name, skeleton_def['group_scales'])
        timestamps, poses = (BiomechanicalReconstruction.Trial & key).fetch1('timestamps', 'poses')

        # get the markers
        markers = get_markers(skeleton, skeleton_def, poses, original_format=True)
        marker_names = list(markers.keys())

        method_name = (TopDownMethodLookup & key).fetch1("top_down_method_name")
        joint_names = TopDownPerson.joint_names(method_name)
        joint_names = normalize_marker_names(joint_names)

        def intersection(lst1, lst2):
            lst3 = [value for value in lst1 if value in lst2]
            return lst3
        common_names = intersection(joint_names, marker_names)

        # 3D markers ##########
        # fetch the 3D markers and make them match the set and order of markers from the model
        markers_ordered = np.array([markers[n] for n in common_names])
        markers_ordered = markers_ordered.transpose([1, 0, 2])
        markers_ordered = markers_ordered * 1000.0 # convert to mm

        # 2D keypoints ##########
        # fetch the 2D keypoints and make them match the set and order of markers from the model
        kp2d, video_camera_name = (TopDownPerson * SingleCameraVideo & key).fetch("keypoints", "camera_name")
        camera_params, camera_names = (Calibration & key).fetch1("camera_calibration", "camera_names")
        assert camera_names == video_camera_name.tolist(), "Videos don't match cameras in calibration"

        # handle cases where there are different numbers of frames
        N = min([k.shape[0] for k in kp2d])
        kp2d = np.stack([k[:N] for k in kp2d], axis=0)

        def map_frame(kp2d):
            return {j: k for j, k in zip(joint_names, kp2d)}

        kp2d_named = [map_frame(k) for k in kp2d.transpose([1, 2, 0, 3])]
        kp2d_ordered = np.array([[kp2d[n] for n in common_names] for kp2d in kp2d_named])
        kp2d_ordered = kp2d_ordered.transpose([2, 0, 1, 3])  # expects camera x time x joint x axis

        # only keep the ones that are in the time range
        fps = int(np.unique((VideoInfo & video_keys).fetch("fps"))[0])
        N = markers_ordered.shape[0]
        frame_0 = int(timestamps[0] * fps)
        kp2d = kp2d_ordered[:, frame_0-1:frame_0-1+N]

        # compute the metrics
        metrics, thresh, confidence = fit_quality.reprojection_quality(markers_ordered, camera_params, kp2d)

        key["reprojection_pck_5"] = metrics[np.argmin(np.abs(thresh - 5)), np.argmin(np.abs(confidence - 0.5))]
        key["reprojection_pck_10"] = metrics[np.argmin(np.abs(thresh - 10)), np.argmin(np.abs(confidence - 0.5))]
        key["reprojection_pck_20"] = metrics[np.argmin(np.abs(thresh - 20)), np.argmin(np.abs(confidence - 0.5))]
        key["reprojection_pck_50"] = metrics[np.argmin(np.abs(thresh - 50)), np.argmin(np.abs(confidence - 0.5))]
        key["reprojection_pck_100"] = metrics[np.argmin(np.abs(thresh - 100)), np.argmin(np.abs(confidence - 0.5))]
        key["reprojection_metrics"] = {
            "metrics": np.array(metrics),
            "thresh": np.array(thresh),
            "confidence": np.array(confidence),
        }
        self.insert1(key)


@schema
class BiomechanicalReconstructionSkeletonOffsets(dj.Computed):
    definition = """
    -> BiomechanicalReconstruction.Trial
    ---
    average_offset       : float
    ankle                : float
    knee                 : float
    hip                  : float
    shoulder             : float
    elbow                : float
    """

    def make(self, key):

        offsets = (BiomechanicalReconstruction & key).fetch1('skeleton_definition')['marker_offsets']
        offsets = {k: np.linalg.norm(v) for k, v in offsets.items()}
        key['ankle'] = (offsets['LAnkle'] + offsets['RAnkle']) / 2
        key['knee'] = (offsets['LKnee'] + offsets['RKnee']) / 2
        key['hip'] = (offsets['LHip'] + offsets['RHip']) / 2
        key['shoulder'] = (offsets['LShoulder'] + offsets['RShoulder']) / 2
        key['elbow'] = (offsets['LElbow'] + offsets['RElbow']) / 2
        key['average_offset'] = offsets = np.mean(list(offsets.values()))
        
        skeleton_definition = BiomechanicalReconstruction.fetch('skeleton_definition')
        # convert list of dicts to dict of lists
        skeleton_definition = {k: [d[k] for d in skeleton_definition] for k in skeleton_definition[0]}

        self.insert1(key)


@schema
class BiomechanicalReconstructionTrialNoise(dj.Computed):
    definition = """
    -> BiomechanicalReconstruction.Trial
    ---
    pose_noise:  float
    """

    def make(self, key):
        
        poses = (BiomechanicalReconstruction.Trial & key).fetch1('poses')
        poses = np.unwrap(poses, axis=0)

        key['pose_noise'] = np.sqrt(np.mean(np.diff(poses, axis=0) ** 2))
        self.insert1(key)


@schema
class BiomechanicalReconstructionJointLimits(dj.Computed):
    definition = """
    -> BiomechanicalReconstruction.Trial
    ---
    violation_fraction:     float
    violation_50_fraction:  float
    violation_100_fraction: float
    under_fraction:         float
    over_fraction:          float
    per_joint:              longblob
    """

    def make(self, key):
        from multi_camera.analysis.biomechanics.bilevel_optimization import reload_skeleton

        # load the skeleton
        model_name, skeleton_def = (BiomechanicalReconstruction & key).fetch1('model_name', 'skeleton_definition')
        skeleton = reload_skeleton(model_name, skeleton_def['group_scales'])
        poses = (BiomechanicalReconstruction.Trial & key).fetch1('poses')

        # get the joint limits
        limits = np.stack([skeleton.getPositionLowerLimits(), skeleton.getPositionUpperLimits()], axis=1)
        limits50 = np.stack([skeleton.getPositionLowerLimits(), skeleton.getPositionUpperLimits()], axis=1) * 1.5
        limits100 = np.stack([skeleton.getPositionLowerLimits(), skeleton.getPositionUpperLimits()], axis=1) * 2.0

        # ignore ones with no allowed range, although really this includes clamped
        # joints which we really should include
        allowed = np.diff(limits, axis=1) > 0

        # compute the violations
        under = (poses < limits[None, :, 0]) * allowed.T
        over = (poses > limits[None, :, 1]) * allowed.T
        under50 = (poses < limits50[None, :, 0]) * allowed.T
        over50 = (poses > limits50[None, :, 1]) * allowed.T
        under100 = (poses < limits100[None, :, 0]) * allowed.T
        over100 = (poses > limits100[None, :, 1]) * allowed.T

        key['violation_fraction'] = np.mean(under | over)
        key['violation_50_fraction'] = np.mean(under50 | over50)
        key['violation_100_fraction'] = np.mean(under100 | over100)
        key['under_fraction'] = np.mean(under)
        key['over_fraction'] = np.mean(over)
        key['per_joint'] = np.mean(under | over, axis=0)

        self.insert1(key)


if __name__ == "__main__":
    import multi_camera.datajoint.biomechanics
    from multi_camera.datajoint.biomechanics import BiomechanicalReconstruction

    BiomechanicalReconstruction.populate("subject_id=136 and model_name='Rajagopal2015_Halpe'")
