import datajoint as dj
import numpy as np

from .calibrate_cameras import Calibration
from pose_pipeline import Video, VideoInfo, TopDownPerson, TopDownMethodLookup, BestDetectedFrames, BlurredVideo

schema = dj.schema("multicamera_tracking")


@schema
class MultiCameraRecording(dj.Manual):
    definition = """
    # Recording from multiple synchronized cameras
    recording_timestamps : timestamp
    camera_config_hash  : varchar(50)    # camera configuration
    video_project       : varchar(50)    # video project, which should match pose pipeline
    ---
    video_base_filename : varchar(100)   # base name for the videos without serial prefix
    """

    def fetch_timestamps(self):
        assert len(self) == 1, "Only fetch timestamps for one recording at a time"
        timestamps = (SingleCameraVideo * VideoInfo & self).fetch("timestamps")
        N = min([len(t) for t in timestamps])
        timestamps = timestamps[0]
        dt = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
        return dt[:N]


@schema
class SingleCameraVideo(dj.Manual):
    definition = """
    # Single view of a multiview recording
    -> MultiCameraRecording
    -> Video
    ---
    camera_name          : varchar(50)
    frame_timestamps      : longblob   # precise timestamps from that camera
    """


@schema
class CalibratedRecording(dj.Manual):
    definition = """
    # Match calibration to a recording
    -> MultiCameraRecording
    -> Calibration
    """


@schema
class PersonKeypointReconstructionMethodLookup(dj.Lookup):
    definition = """
    reconstruction_method      : int
    ---
    reconstruction_method_name : varchar(50)
    """
    contents = [
        {"reconstruction_method": 0, "reconstruction_method_name": "Robust Triangulation"},
        {"reconstruction_method": 1, "reconstruction_method_name": "Explicit Optimization KP Conf, MaxHuber=10"},
        {"reconstruction_method": 2, "reconstruction_method_name": "Implicit Optimization KP Conf, MaxHuber=10"},
        {"reconstruction_method": 3, "reconstruction_method_name": "Implicit Optimization"},
        {"reconstruction_method": 4, "reconstruction_method_name": "Triangulation"},
        {"reconstruction_method": 5, "reconstruction_method_name": r"Robust Triangulation $\\sigma=100$"},
        {"reconstruction_method": 6, "reconstruction_method_name": r"Robust Triangulation $\\sigma=50$"},
        {"reconstruction_method": 7, "reconstruction_method_name": "Explicit Optimization"},
        {"reconstruction_method": 8, "reconstruction_method_name": r"Robust Triangulation $\\gamma=0.3$"},
        {"reconstruction_method": 9, "reconstruction_method_name": "Implicit Optimization KP Conf"},
        {"reconstruction_method": 10, "reconstruction_method_name": r"Implicit Optimization $\\gamma=0.3$"},
        {"reconstruction_method": 11, "reconstruction_method_name": "Implicit Optimization, MaxHuber=10"},
        {"reconstruction_method": 12, "reconstruction_method_name": r"Implicit Optimization $\\sigma=50$"},
    ]


@schema
class PersonKeypointReconstructionMethod(dj.Manual):
    definition = """
    # Use the TopDownKeypoints to reconstruct the 3D joint locations
    -> CalibratedRecording
    -> PersonKeypointReconstructionMethodLookup
    tracking_method     :  int
    top_down_method     :  int
    """


@schema
class PersonKeypointReconstruction(dj.Computed):
    definition = """
    # Use the TopDownKeypoints to reconstruct the 3D joint locations
    -> PersonKeypointReconstructionMethod
    ---
    keypoints3d         : longblob
    camera_weights      : longblob
    reprojection_loss   : float
    skeleton_loss       : float
    smoothness_loss     : float
    """

    def make(self, key):
        import numpy as np
        from ..analysis.camera import robust_triangulate_points, triangulate_point
        from ..analysis.optimize_reconstruction import skeleton_loss, reprojection_loss, smoothness_loss

        calibration_key = (Calibration & key).fetch1("KEY")
        recording_key = (MultiCameraRecording & key).fetch1("KEY")
        top_down_method = key["top_down_method"]
        tracking_method = key["tracking_method"]
        reconstruction_method = key["reconstruction_method"]

        top_down_method_name = (TopDownMethodLookup & key).fetch1("top_down_method_name")

        camera_calibration, camera_names = (Calibration & calibration_key).fetch1("camera_calibration", "camera_names")
        keypoints, camera_name = (
            TopDownPerson * SingleCameraVideo * MultiCameraRecording
            & {
                "top_down_method": top_down_method,
                "tracking_method": tracking_method,
                "reconstruction_method": reconstruction_method,
            }
            & recording_key
        ).fetch("keypoints", "camera_name")

        # need to add zeros for missing frames at the end
        N = max([len(k) for k in keypoints])
        keypoints = np.stack(
            [np.concatenate([k, np.zeros([N - k.shape[0], *k.shape[1:]])], axis=0) for k in keypoints], axis=0
        )

        print(len(camera_names), len(camera_name))
        # work out the order that matches the calibration (should normally match)
        order = [list(camera_name).index(c) for c in camera_names]
        points2d = np.stack([keypoints[o][:, :, :] for o in order], axis=0)

        if top_down_method_name == "MMPoseHalpe":
            joints = TopDownPerson.joint_names("MMPoseHalpe")
            pairs = [
                ("Pelvis", "Right Hip"),
                ("Pelvis", "Left Hip"),
                ("Left Ankle", "Left Knee"),
                ("Right Ankle", "Right Knee"),
                ("Left Knee", "Left Hip"),
                ("Right Knee", "Right Hip"),
                ("Left Hip", "Pelvis"),
                ("Right Hip", "Pelvis"),
                ("Left Shoulder", "Left Elbow"),
                ("Right Shoulder", "Right Elbow"),
                ("Left Elbow", "Left Wrist"),
                ("Right Elbow", "Right Wrist"),
                ("Left Heel", "Left Big Toe"),
                ("Right Heel", "Right Big Toe"),
                ("Right Shoulder", "Left Shoulder"),
                ("Right Shoulder", "Right Elbow"),
                ("Right Elbow", "Right Wrist"),
                ("Left Shoulder", "Left Elbow"),
                ("Left Elbow", "Left Wrist"),
            ]
        else:
            joints = TopDownPerson.joint_names(top_down_method_name)
            pairs = [
                ("Left Hip", "Right Hip"),
                ("Left Ankle", "Left Knee"),
                ("Right Ankle", "Right Knee"),
                ("Left Knee", "Left Hip"),
                ("Right Knee", "Right Hip"),
                ("Left Shoulder", "Left Elbow"),
                ("Right Shoulder", "Right Elbow"),
                ("Left Elbow", "Left Wrist"),
                ("Right Elbow", "Right Wrist"),
                ("Left Heel", "Left Big Toe"),
                ("Right Heel", "Right Big Toe"),
                ("Right Shoulder", "Left Shoulder"),
                ("Right Shoulder", "Right Elbow"),
                ("Right Elbow", "Right Wrist"),
                ("Left Shoulder", "Left Elbow"),
                ("Left Elbow", "Left Wrist"),
            ]
        skeleton = np.array([(joints.index(p[0]), joints.index(p[1])) for p in pairs])

        # select method for reconstruction
        reconstruction_method_name = (PersonKeypointReconstructionMethodLookup & key).fetch1(
            "reconstruction_method_name"
        )

        if reconstruction_method_name == "Triangulation":
            # downweight any views less than 0.5 confidence
            conf = points2d[..., -1]
            conf[conf < 0.5] = 0.0
            points2d[..., -1] = conf
            points3d = triangulate_point(camera_calibration, points2d, return_confidence=True)
            camera_weights = []
            print(points3d.shape)

        elif reconstruction_method_name == "Robust Triangulation":
            points3d, camera_weights = robust_triangulate_points(camera_calibration, points2d, return_weights=True)

        elif reconstruction_method_name == "Robust Triangulation $\sigma=100$":
            points3d, camera_weights = robust_triangulate_points(
                camera_calibration, points2d, return_weights=True, sigma=100
            )

        elif reconstruction_method_name == "Robust Triangulation $\sigma=50$":
            points3d, camera_weights = robust_triangulate_points(
                camera_calibration, points2d, return_weights=True, sigma=50
            )

        elif reconstruction_method_name == "Robust Triangulation $\gamma=0.3$":
            points3d, camera_weights = robust_triangulate_points(
                camera_calibration, points2d, return_weights=True, threshold=0.3
            )

        elif reconstruction_method_name == "Explicit Optimization KP Conf, MaxHuber=10":
            from ..analysis.optimize_reconstruction import optimize_trajectory

            points3d, camera_weights = optimize_trajectory(
                points2d,
                camera_calibration,
                "explicit",
                return_weights=True,
                delta_weight=0.1,
                skeleton_weight=0.1,
                skeleton=skeleton,
                huber_max=10,
                robust_camera_weights=False,
                max_iters=50000,
            )
        elif reconstruction_method_name == "Explicit Optimization":
            from ..analysis.optimize_reconstruction import optimize_trajectory

            points3d, camera_weights = optimize_trajectory(
                points2d,
                camera_calibration,
                "explicit",
                return_weights=True,
                delta_weight=0.1,
                skeleton_weight=0.1,
                skeleton=skeleton,
                huber_max=10000,
                robust_camera_weights=True,
                max_iters=50000,
            )

        elif reconstruction_method_name == "Implicit Optimization $\gamma=0.3$":
            from ..analysis.optimize_reconstruction import optimize_trajectory

            points3d, camera_weights = optimize_trajectory(
                points2d,
                camera_calibration,
                "explicit",
                return_weights=True,
                delta_weight=0.1,
                skeleton_weight=0.1,
                skeleton=skeleton,
                huber_max=10000,
                confidence_threshold=0.3,
                robust_camera_weights=True,
                max_iters=50000,
            )
        elif reconstruction_method_name == "Implicit Optimization $\sigma=50$":
            from ..analysis.optimize_reconstruction import optimize_trajectory

            points3d, camera_weights = optimize_trajectory(
                points2d,
                camera_calibration,
                "explicit",
                return_weights=True,
                delta_weight=0.1,
                skeleton_weight=0.1,
                skeleton=skeleton,
                huber_max=10000,
                sigma=50,
                robust_camera_weights=True,
                max_iters=50000,
            )
        elif reconstruction_method_name == "Implicit Optimization KP Conf, MaxHuber=10":
            from ..analysis.optimize_reconstruction import optimize_trajectory

            points3d, camera_weights = optimize_trajectory(
                points2d,
                camera_calibration,
                "implicit",
                return_weights=True,
                delta_weight=0.1,
                skeleton_weight=0.1,
                skeleton=skeleton,
                huber_max=10,
                robust_camera_weights=False,
                max_iters=50000,
            )

        elif reconstruction_method_name == "Implicit Optimization KP Conf":
            from ..analysis.optimize_reconstruction import optimize_trajectory

            points3d, camera_weights = optimize_trajectory(
                points2d,
                camera_calibration,
                "implicit",
                return_weights=True,
                delta_weight=0.1,
                skeleton_weight=0.1,
                skeleton=skeleton,
                huber_max=10000,
                robust_camera_weights=False,
                max_iters=50000,
            )

        elif reconstruction_method_name == "Implicit Optimization":
            from ..analysis.optimize_reconstruction import optimize_trajectory

            points3d, camera_weights = optimize_trajectory(
                points2d,
                camera_calibration,
                "implicit",
                return_weights=True,
                delta_weight=0.1,
                skeleton_weight=0.1,
                skeleton=skeleton,
                huber_max=10000,
                robust_camera_weights=True,
                max_iters=50000,
            )

        elif reconstruction_method_name == "Implicit Optimization, MaxHuber=10":
            from ..analysis.optimize_reconstruction import optimize_trajectory

            points3d, camera_weights = optimize_trajectory(
                points2d,
                camera_calibration,
                "implicit",
                return_weights=True,
                delta_weight=0.1,
                skeleton_weight=0.1,
                skeleton=skeleton,
                huber_max=10,
                robust_camera_weights=True,
                max_iters=50000,
            )

        else:
            raise ValueError("Unknown reconstruction method")

        key["keypoints3d"] = np.array(points3d)
        key["camera_weights"] = np.array(camera_weights)
        key["reprojection_loss"] = reprojection_loss(camera_calibration, points2d, points3d[:, :, :3], huber_max=100)
        key["skeleton_loss"] = skeleton_loss(points3d[:, :, :3], skeleton)
        key["smoothness_loss"] = smoothness_loss(points3d[:, :, :3])
        if np.isinf(key["smoothness_loss"]):
            key["smoothness_loss"] = 1e10

        self.insert1(key, allow_direct_insert=True)

    @property
    def key_source(self):
        # awkward double negative is to ensure all BlurredVideo views were computed
        return PersonKeypointReconstructionMethod - (SingleCameraVideo - TopDownPerson).proj()

    def plot_joint(self, joint_idx, relative=False):
        from pose_pipeline import VideoInfo, PersonBbox
        from matplotlib import pyplot as plt

        kp3d = self.fetch1("keypoints3d")
        timestamps = (VideoInfo * SingleCameraVideo & self).fetch("timestamps", limit=1)[0]
        present = np.stack((PersonBbox * SingleCameraVideo & self).fetch("present"))
        present = np.sum(present, axis=0) / present.shape[0]
        kp2d = (TopDownPerson * SingleCameraVideo & self).fetch("keypoints")

        if relative:
            relative_idx = TopDownPerson.joint_names("MMPoseHalpe").index("Pelvis")
            kp3d = kp3d - kp3d[:, relative_idx, None, :]

        kp3d = kp3d[:, :27]

        N = min([k.shape[0] for k in kp2d])
        keypoints2d = np.stack([k[:N] for k in kp2d], axis=0)

        keypoints2d = keypoints2d[:, :, : kp3d.shape[1]]
        dt = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
        dt = dt[: kp3d.shape[0]]

        fig, ax = plt.subplots(3, 1, figsize=(5, 4))
        ax[0].plot(dt, kp3d[:, joint_idx, :3])
        ax[1].plot(dt, kp3d[:, joint_idx, 3])
        ax[1].plot(dt, present)
        ax[1].set_ylim(0, 1)
        ax[2].plot(dt, keypoints2d[:, :, joint_idx, 2].T)

        plt.tight_layout()

    def export_trc(self, filename, z_offset=0, start=None, end=None, return_points=False, smooth=False):
        """Export an OpenSim file of marker trajectories

        Params:
            filename (string) : filename to export to
            z_offset (float, optional) : optional vertical offset
            start    (float, optional) : if set, time to start at
            end      (float, optional) : if set, time to end at
            return_points (bool, opt)  : if true, return points
        """

        from pose_pipeline import TopDownPerson, VideoInfo
        from multi_camera.analysis.biomechanics.opensim import normalize_marker_names, points3d_to_trc

        method_name = (TopDownMethodLookup & self).fetch1("top_down_method_name")
        joint_names = TopDownPerson.joint_names(method_name)

        joints3d = self.fetch1("keypoints3d").copy()
        joints3d = joints3d[:, : len(joint_names)]  # discard "unnamed" joints
        joints3d = joints3d / 1000.0  # convert to m
        fps = np.unique((VideoInfo * SingleCameraVideo * MultiCameraRecording & self).fetch("fps"))

        if joints3d.shape[-1] == 4:
            joints3d = joints3d[..., :-1]

        if end is not None:
            joints3d = joints3d[: int(end * fps)]
        if start is not None:
            joints3d = joints3d[int(start * fps) :]

        if smooth:
            import scipy

            for i in range(joints3d.shape[1]):
                for j in range(joints3d.shape[2]):
                    joints3d[:, i, j] = scipy.signal.medfilt(joints3d[:, i, j], 5)

        points3d_to_trc(
            joints3d + np.array([[[0, z_offset, 0]]]), filename, normalize_marker_names(joint_names), fps=fps
        )

        if return_points:
            return joints3d


@schema
class ReprojectionError(dj.Computed):
    definition = """
    # Reprojection error for each camera
    -> PersonKeypointReconstruction
    camera_name          : varchar(10)
    ---
    reprojection_error : float
    reprojection_error_timeseries : longblob
    num_zeros : int
    num_nans : int
    """
    def make(self, key):
        print(key)
        import numpy as np
        from body_models.losses import reprojection_loss
        from multi_camera.datajoint.multi_camera_dj import SingleCameraVideo, MultiCameraRecording, PersonKeypointReconstruction
        from multi_camera.datajoint.calibrate_cameras import Calibration
        from pose_pipeline import TopDownPerson
        
        def fetch_key_data(key):
            """Fetch necessary data for a given key."""
            calibration_key = (Calibration & key).fetch1("KEY")
            recording_key = (MultiCameraRecording & key).fetch1("KEY")
            top_down_method = key["top_down_method"]
            tracking_method = key["tracking_method"]
            reconstruction_method = key["reconstruction_method"]
            k3d = (PersonKeypointReconstruction & key).fetch1("keypoints3d")
            camera_calibration, camera_names = (Calibration & calibration_key).fetch1("camera_calibration", "camera_names")
            keypoints, camera_name = (
                TopDownPerson * SingleCameraVideo * MultiCameraRecording
                & {
                    "top_down_method": top_down_method,
                    "tracking_method": tracking_method,
                    "reconstruction_method": reconstruction_method,
                }
                & recording_key
            ).fetch("keypoints", "camera_name")
            recording_timestamps = (MultiCameraRecording & recording_key).fetch1("recording_timestamps")
            
            return k3d, camera_calibration, camera_names, keypoints, camera_name, recording_timestamps
        
        def pad_keypoints(keypoints):
            """Pad keypoints with zeros for missing frames."""
            N = max(len(k) for k in keypoints)
            return np.stack([np.pad(k, ((0, N - len(k)), (0, 0), (0, 0)), mode='constant') for k in keypoints], axis=0)
        
        def reorder_keypoints(keypoints, camera_names, camera_name):
            """Reorder keypoints to match the calibration order."""
            camera_name_list = camera_name.tolist()  # Convert to list
            order = [camera_name_list.index(c) for c in camera_names]
            return np.stack([keypoints[o] for o in order], axis=0)

        def match_joint_count(points2d, points3d):
            """Ensure the number of joints matches between points2d and points3d."""
            num_joints = min(points2d.shape[2], points3d.shape[1])
            points2d = points2d[:, :, :num_joints, :]
            points3d = points3d[:, :num_joints, :]
            return points2d, points3d
        
        k3d, camera_calibration, camera_names, keypoints, camera_name, recording_timestamps = fetch_key_data(key)
        keypoints = pad_keypoints(keypoints)
        points2d = reorder_keypoints(keypoints, camera_names, camera_name)
        points2d, k3d = match_joint_count(points2d, k3d)
        reprojection_values = np.nanmean(reprojection_loss(camera_calibration, points2d, k3d,huber_max=50, average = False),axis=2)
        
        for camera_name, reprojection_value in zip(camera_names, reprojection_values):
            num_zeros = np.sum(reprojection_value == 0)
            num_nans = np.sum(np.isnan(reprojection_value))
            
            # Replace 0s with NaNs
            reprojection_value_with_nans = np.where(reprojection_value == 0, np.nan, reprojection_value)
            
            # Calculate reprojection error after replacing 0s with NaNs
            reprojection_error_with_nans = float(np.nanmean(reprojection_value_with_nans))
            if np.isnan(reprojection_error_with_nans):
                reprojection_error_with_nans = -1.
            self.insert1({
                **key,
                "camera_name": camera_name,
                "reprojection_error": reprojection_error_with_nans,
                "reprojection_error_timeseries": reprojection_value_with_nans,
                "num_zeros": num_zeros,
                "num_nans": num_nans,
            })

@schema
class PersonKeypointReprojectionQuality(dj.Computed):
    definition = """
    -> PersonKeypointReconstruction
    ---
    reprojection_pck_5       : float
    reprojection_pck_10      : float
    reprojection_pck_20      : float
    reprojection_pck_50      : float
    reprojection_pck_100      : float
    reprojection_metrics     : longblob
    """

    def make(self, key):
        from multi_camera.analysis import fit_quality

        kp3d = (PersonKeypointReconstruction & key).fetch1("keypoints3d")
        kp2d, video_camera_name = (TopDownPerson * SingleCameraVideo & key).fetch("keypoints", "camera_name")
        camera_params, camera_names = (Calibration & key).fetch1("camera_calibration", "camera_names")
        assert camera_names == video_camera_name.tolist(), "Videos don't match cameras in calibration"

        # handle cases where there are different numbers of frames
        N = min([k.shape[0] for k in kp2d])
        kp2d = np.stack([k[:N] for k in kp2d], axis=0)

        # to keep it comparable, only analyzing the first 27 joints, also need to exclude 3D confidence
        # for projection to work properly
        if (TopDownMethodLookup & key).fetch1("top_down_method_name") == "MMPoseHalpe":
            idx = np.setdiff1d(np.arange(26), [17])
            kp3d = kp3d[:, idx, :3]
            kp2d = kp2d[:, :, idx]

        if (TopDownMethodLookup & key).fetch1("top_down_method_name") == "MMPoseWholebody":
            kp3d = kp3d[:, :23, :3]
            kp2d = kp2d[:, :, :23]

        metrics, thresh, confidence = fit_quality.reprojection_quality(kp3d, camera_params, kp2d)

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
class PersonKeypointReconstructionVideo(dj.Computed):
    definition = """
    # Video from reconstruction
    -> PersonKeypointReconstruction
    ---
    output_video      : attach@localattach    # datajoint managed video file
    """

    def make(self, key):
        import os
        import tempfile
        import numpy as np
        from ..utils.visualization import skeleton_video

        method_name = (TopDownMethodLookup & key).fetch1("top_down_method_name")
        fd, out_file_name = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        fps = np.unique((VideoInfo * SingleCameraVideo & key).fetch("fps"))[0]
        # fps = np.round(fps)[0]

        keypoints3d = (PersonKeypointReconstruction & key).fetch1("keypoints3d")
        skeleton_video(keypoints3d, out_file_name, method_name, fps=fps)

        key["output_video"] = out_file_name
        self.insert1(key)


@schema
class PersonKeypointReprojectionVideo(dj.Computed):
    definition = """
    # Video of preprojections
    -> PersonKeypointReconstruction
    ---
    output_video      : attach@localattach    # datajoint managed video file
    """

    def make(self, key):
        from ..utils.visualization import make_reprojection_video

        key["output_video"] = make_reprojection_video(key)
        self.insert1(key)

    @property
    def key_source(self):
        # awkward double negative is to ensure all BlurredVideo views were computed
        return PersonKeypointReconstruction & MultiCameraRecording - (SingleCameraVideo - BlurredVideo).proj()


@schema
class PersonKeypointReprojectionVideos(dj.Computed):
    definition = """
    # Videos of reconstruction preprojections
    -> PersonKeypointReconstruction
    ---
    """

    class Video(dj.Part):
        definition = """
        -> PersonKeypointReprojectionVideos
        -> SingleCameraVideo
        ---
        output_video      : attach@localattach    # datajoint managed video file
        """

    def make(self, key):
        import cv2
        import os
        import tempfile
        from ..analysis.camera import project_distortion
        from pose_pipeline.utils.visualization import video_overlay, draw_keypoints

        self.insert1(key)

        videos = Video * MultiCameraRecording * PersonKeypointReconstruction * SingleCameraVideo & key
        video_keys, video_camera_name = (SingleCameraVideo.proj() * videos).fetch("KEY", "camera_name")
        keypoints3d = (PersonKeypointReconstruction & key).fetch1("keypoints3d")
        camera_params, camera_names = (Calibration & key).fetch1("camera_calibration", "camera_names")
        assert camera_names == video_camera_name.tolist(), "Videos don't match cameras in calibration"

        # get video parameters
        width = np.unique((VideoInfo & video_keys).fetch("width"))[0]
        height = np.unique((VideoInfo & video_keys).fetch("height"))[0]
        fps = np.unique((VideoInfo & video_keys).fetch("fps"))[0]

        # compute keypoints from reprojection of SMPL fit
        kp3d = keypoints3d[..., :-1]
        conf3d = keypoints3d[..., -1]
        keypoints2d = np.array(
            [project_distortion(camera_params, i, kp3d) for i in range(camera_params["mtx"].shape[0])]
        )

        print(f"Height: {height}. Width: {width}. FPS: {fps}")

        # handle any bad projections
        valid_kp = np.tile((conf3d < 0.5)[None, ...], [keypoints2d.shape[0], 1, 1])
        clipped = np.logical_or.reduce(
            (
                keypoints2d[..., 0] <= 0,
                keypoints2d[..., 0] >= width,
                keypoints2d[..., 1] <= 0,
                keypoints2d[..., 1] >= height,
                np.isnan(keypoints2d[..., 0]),
                np.isnan(keypoints2d[..., 1]),
                valid_kp,
            )
        )
        keypoints2d[clipped, 0] = 0
        keypoints2d[clipped, 1] = 0
        # add low confidence when clipped
        keypoints2d = np.concatenate([keypoints2d, ~clipped[..., None] * 1.0], axis=-1)

        for i, video_key in enumerate(video_keys):

            def render_overlay(frame, idx):
                if idx >= keypoints2d.shape[1]:
                    return frame

                frame = draw_keypoints(frame, keypoints2d[i, idx], radius=6, color=(125, 125, 255), threshold=0.75)

                return frame

            fd, out_file_name = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)

            video = (BlurredVideo & video_key).fetch1("output_video")
            video_overlay(video, out_file_name, render_overlay, max_frames=None, downsample=2, compress=True)

            single_video_key = (SingleCameraVideo * PersonKeypointReconstruction & key & video_key).fetch1("KEY")
            single_video_key["output_video"] = out_file_name

            PersonKeypointReprojectionVideos.Video.insert1(single_video_key)

            os.remove(video)
            os.remove(out_file_name)


@schema
class SMPLReconstruction(dj.Computed):
    definition = """
    # Use the TopDownKeypoints to reconstruct the 3D joint locations
    -> PersonKeypointReconstruction
    ---
    poses               : longblob
    shape               : longblob
    orientation         : longblob
    translation         : longblob
    joints3d            : longblob
    vertices            : longblob
    faces               : longblob
    """

    def make(self, key):
        from ..analysis.easymocap import easymocap_fit_smpl_3d, get_joint_openpose, get_vertices, get_faces
        from easymocap.dataset import CONFIG as config

        # get triangulated points and convert to meters
        points3d = (PersonKeypointReconstruction & key).fetch1("keypoints3d").copy()
        points3d[..., :3] = points3d[..., :3] / 1000.0  # convert coordinates to m, but leave confidence untouched

        if key["top_down_method"] == 2:
            # Convert the HALPE coordinate order to the expected order

            def joint_renamer(j):
                j = j.replace("Sternum", "Neck")
                j = j.replace("Right ", "R")
                j = j.replace("Left ", "L")
                j = j.replace("Little", "Small")
                j = j.replace("Pelvis", "MidHip")
                j = j.replace(" ", "")
                return j

            def normalize_marker_names(joints):
                """Convert joint names to those expected by OpenSim model"""
                return [joint_renamer(j) for j in joints]

            joint_names = normalize_marker_names(TopDownPerson.joint_names("MMPoseHalpe"))

            # move these to where COCO would put them (which is what Body25 uses)
            points3d[:, joint_names.index("Neck")] = (
                points3d[:, joint_names.index("RShoulder")] + points3d[:, joint_names.index("LShoulder")]
            ) / 2
            points3d[:, joint_names.index("MidHip")] = (
                points3d[:, joint_names.index("RHip")] + points3d[:, joint_names.index("LHip")]
            ) / 2
            # reduce confidence on little toes as it seems to lock onto values from big toe (quick with MMPose model)
            points3d[:, joint_names.index("RSmallToe"), -1] = points3d[:, joint_names.index("RSmallToe"), -1] * 0.5
            points3d[:, joint_names.index("LSmallToe"), -1] = points3d[:, joint_names.index("LSmallToe"), -1] * 0.5

            joint_reorder = np.array([joint_names.index(j) for j in config["body25"]["joint_names"]])
            points3d = points3d[:, joint_reorder]

        elif key["top_down_method"] == 4:
            # for OpenPose the keypoint order can be preserved
            pass

        else:
            raise NotImplementedError(f'Top down method {key["top_down_method"]} not supported.')

        res = easymocap_fit_smpl_3d(points3d, verbose=True)
        key["poses"] = res["poses"]
        key["shape"] = res["shapes"]
        key["orientation"] = res["Rh"]
        key["translation"] = res["Th"]
        key["joints3d"] = get_joint_openpose(res)
        key["vertices"] = get_vertices(res)
        key["faces"] = get_faces()
        self.insert1(key)

    def get_result(self):
        poses, shapes, Rh, Th = self.fetch1("poses", "shape", "orientation", "translation")
        return {"poses": poses, "shapes": shapes, "Rh": Rh, "Th": Th}

    def export_trc(self, filename, z_offset=0, start=None, end=None, return_points=False):
        """Export an OpenSim file of marker trajectories

        Params:
            filename (string) : filename to export to
            z_offset (float, optional) : optional vertical offset
            start    (float, optional) : if set, time to start at
            end      (float, optional) : if set, time to end at
            return_points (bool, opt)  : if true, return points
        """

        from pose_pipeline import TopDownPerson, VideoInfo
        from multi_camera.analysis.biomechanics.opensim import normalize_marker_names, points3d_to_trc

        joint_names = TopDownPerson.joint_names("OpenPose")
        joints3d = self.fetch1("joints3d")
        fps = np.unique((VideoInfo * SingleCameraVideo * MultiCameraRecording & self).fetch("fps"))

        if end is not None:
            joints3d = joints3d[: int(end * fps)]
        if start is not None:
            joints3d = joints3d[int(start * fps) :]

        points3d_to_trc(
            joints3d + np.array([[[0, z_offset, 0]]]), filename, normalize_marker_names(joint_names), fps=fps
        )

        if return_points:
            return joints3d


@schema
class SMPLReconstructionVideos(dj.Computed):
    definition = """
    # Videos of SMPL reconstruction from multiview
    -> SMPLReconstruction
    ---
    """

    class Video(dj.Part):
        definition = """
        -> SMPLReconstructionVideos
        -> SingleCameraVideo
        ---
        output_video      : attach@localattach    # datajoint managed video file
        """

    def make(self, key):
        import cv2
        import os
        import tempfile
        from einops import rearrange
        from ..analysis.camera import project_distortion, get_intrinsic, get_extrinsic, distort_3d
        from pose_pipeline.utils.visualization import video_overlay, draw_keypoints
        from easymocap.visualize.renderer import Renderer

        self.insert1(key)

        videos = Video * TopDownPerson * MultiCameraRecording * PersonKeypointReconstruction * SingleCameraVideo & key
        video_keys, camera_names, keypoints2d = videos.fetch("KEY", "camera_name", "keypoints")
        camera_params = (Calibration & key).fetch1("camera_calibration")

        # get video parameters
        width = np.unique((VideoInfo & video_keys).fetch("width"))[0]
        height = np.unique((VideoInfo & video_keys).fetch("height"))[0]
        fps = np.unique((VideoInfo & video_keys).fetch("fps"))[0]

        # get vertices, in world coordintes
        faces, vertices, joints3d = (SMPLReconstruction & key).fetch1("faces", "vertices", "joints3d")

        # convert from meter to the mm that the camera model expects
        joints3d = joints3d * 1000.0

        # compute keypoints from reprojection of SMPL fit
        keypoints2d = np.array(
            [project_distortion(camera_params, i, joints3d) for i in range(camera_params["mtx"].shape[0])]
        )

        render = Renderer(height=height, width=width, down_scale=2, bg_color=[0, 0, 0, 0.0])

        for i, video_key in enumerate(video_keys):
            # get camera parameters
            K = np.array(get_intrinsic(camera_params, i))

            # don't use real extrinsic since we apply distortion which does this
            R = np.eye(3)
            T = np.zeros((3,))
            cameras = {"K": [K], "R": [R], "T": [T]}

            # account for camera distortion. convert vertices to mm first.
            vertices_distorted = np.array(distort_3d(camera_params, i, vertices * 1000.0))
            # then back to meters
            vertices_distorted = vertices_distorted / 1000.0

            def render_overlay(frame, idx, vertices=vertices_distorted, faces=faces, cameras=cameras):
                if idx >= vertices.shape[0]:
                    return frame

                render_data = {3: {"vertices": vertices[idx], "faces": faces, "name": "human"}}

                frame = render.render(render_data, cameras, [frame], add_back=True)[0].copy()
                frame = draw_keypoints(frame, keypoints2d[i][idx] / render.down_scale, radius=2, color=(125, 125, 255))

                return frame

            fd, out_file_name = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)

            video = (Video & video_key).fetch1("video")
            video_overlay(video, out_file_name, render_overlay, max_frames=None, downsample=2, compress=True)

            single_video_key = (SingleCameraVideo * SMPLReconstruction & key & video_key).fetch1("KEY")
            single_video_key["output_video"] = out_file_name

            SMPLReconstructionVideos.Video.insert1(single_video_key)

            os.remove(video)
            os.remove(out_file_name)


@schema
class SMPLXReconstruction(dj.Computed):
    definition = """
    # Use the TopDownKeypoints to reconstruct the 3D joint locations
    -> PersonKeypointReconstruction
    ---
    poses               : longblob
    shape               : longblob
    orientation         : longblob
    translation         : longblob
    expression          : longblob
    joints3d            : longblob
    vertices            : longblob
    faces               : longblob
    """

    def make(self, key):
        from ..analysis.easymocap import easymocap_fit_smpl_3d, get_joint_openpose, get_vertices, get_faces
        from easymocap.dataset import CONFIG as config

        # get triangulated points and convert to meters
        points3d = (PersonKeypointReconstruction & key).fetch1("keypoints3d").copy()
        points3d[..., :3] = points3d[..., :3] / 1000.0  # convert coordinates to m, but leave confidence untouched

        if key["top_down_method"] == 2:
            # Convert the HALPE coordinate order to the expected order

            def joint_renamer(j):
                j = j.replace("Sternum", "Neck")
                j = j.replace("Right ", "R")
                j = j.replace("Left ", "L")
                j = j.replace("Little", "Small")
                j = j.replace("Pelvis", "MidHip")
                j = j.replace(" ", "")
                return j

            def normalize_marker_names(joints):
                """Convert joint names to those expected by OpenSim model"""
                return [joint_renamer(j) for j in joints]

            joint_names = normalize_marker_names(TopDownPerson.joint_names("MMPoseHalpe"))

            # move these to where COCO would put them (which is what Body25 uses)
            points3d[:, joint_names.index("Neck")] = (
                points3d[:, joint_names.index("RShoulder")] + points3d[:, joint_names.index("LShoulder")]
            ) / 2
            points3d[:, joint_names.index("MidHip")] = (
                points3d[:, joint_names.index("RHip")] + points3d[:, joint_names.index("LHip")]
            ) / 2
            # reduce confidence on little toes as it seems to lock onto values from big toe (quick with MMPose model)
            points3d[:, joint_names.index("RSmallToe"), -1] = points3d[:, joint_names.index("RSmallToe"), -1] * 0.1
            points3d[:, joint_names.index("LSmallToe"), -1] = points3d[:, joint_names.index("LSmallToe"), -1] * 0.1

            joint_reorder = np.array([joint_names.index(j) for j in config["body25"]["joint_names"]])
            points3d_body25 = points3d[:, joint_reorder]

            # from https://github.com/Fang-Haoshu/Halpe-FullBody
            left_hand = points3d[:, np.arange(94, 115)]
            right_hand = points3d[:, np.arange(115, 136)]
            # leave the first 17 points off. doesn't use outline. add 2 points at
            # end that halpe is missing
            face = points3d[:, np.arange(26 + 17, 94)]

            points3d = np.concatenate([points3d_body25, left_hand, right_hand, face], axis=1)

        elif key["top_down_method"] == 4:
            # for OpenPose the keypoint order can be preserved
            pass

        else:
            raise NotImplementedError(f'Top down method {key["top_down_method"]} not supported.')

        res = easymocap_fit_smpl_3d(points3d, verbose=True, body_model="smplx", skel_type="facebodyhand")
        key["poses"] = res["poses"]
        key["shape"] = res["shapes"]
        key["expression"] = res["expression"]
        key["orientation"] = res["Rh"]
        key["translation"] = res["Th"]
        key["joints3d"] = get_joint_openpose(res, body_model="smplx")
        key["vertices"] = get_vertices(res, body_model="smplx")
        key["faces"] = get_faces(body_model="smplx")
        self.insert1(key)


@schema
class SMPLXReconstructionVideos(dj.Computed):
    definition = """
    # Videos of SMPL reconstruction from multiview
    -> SMPLXReconstruction
    ---
    """

    class Video(dj.Part):
        definition = """
        -> SMPLXReconstructionVideos
        -> SingleCameraVideo
        ---
        output_video      : attach@localattach    # datajoint managed video file
        """

    def make(self, key):
        import cv2
        import os
        import tempfile
        from einops import rearrange
        from ..analysis.camera import project_distortion, get_intrinsic, get_extrinsic, distort_3d
        from pose_pipeline.utils.visualization import video_overlay, draw_keypoints
        from easymocap.visualize.renderer import Renderer

        self.insert1(key)

        videos = Video * TopDownPerson * MultiCameraRecording * PersonKeypointReconstruction * SingleCameraVideo & key
        video_keys, camera_names, keypoints2d = videos.fetch("KEY", "camera_name", "keypoints")
        camera_params = (Calibration & key).fetch1("camera_calibration")

        # get video parameters
        width = np.unique((VideoInfo & video_keys).fetch("width"))[0]
        height = np.unique((VideoInfo & video_keys).fetch("height"))[0]
        fps = np.unique((VideoInfo & video_keys).fetch("fps"))[0]

        # get vertices, in world coordintes
        faces, vertices, joints3d = (SMPLXReconstruction & key).fetch1("faces", "vertices", "joints3d")

        # convert from meter to the mm that the camera model expects
        joints3d = joints3d * 1000.0

        # compute keypoints from reprojection of SMPL fit
        keypoints2d = np.array(
            [project_distortion(camera_params, i, joints3d) for i in range(camera_params["mtx"].shape[0])]
        )

        render = Renderer(height=height, width=width, down_scale=2, bg_color=[0, 0, 0, 0.0])

        for i, video_key in enumerate(video_keys):
            # get camera parameters
            K = np.array(get_intrinsic(camera_params, i))

            # don't use real extrinsic since we apply distortion which does this
            R = np.eye(3)
            T = np.zeros((3,))
            cameras = {"K": [K], "R": [R], "T": [T]}

            # account for camera distortion. convert vertices to mm first.
            vertices_distorted = np.array(distort_3d(camera_params, i, vertices * 1000.0))
            # then back to meters
            vertices_distorted = vertices_distorted / 1000.0

            def render_overlay(frame, idx, vertices=vertices_distorted, faces=faces, cameras=cameras):
                if idx >= vertices.shape[0]:
                    return frame

                render_data = {3: {"vertices": vertices[idx], "faces": faces, "name": "human"}}

                frame = render.render(render_data, cameras, [frame], add_back=True)[0].copy()
                # frame = draw_keypoints(frame, keypoints2d[i][idx] / render.down_scale, radius=1, color=(125, 125, 255))

                return frame

            fd, out_file_name = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)

            video = (Video & video_key).fetch1("video")
            video_overlay(video, out_file_name, render_overlay, max_frames=None, downsample=2, compress=True)

            single_video_key = (SingleCameraVideo * SMPLReconstruction & key & video_key).fetch1("KEY")
            single_video_key["output_video"] = out_file_name

            SMPLXReconstructionVideos.Video.insert1(single_video_key)

            os.remove(video)
            os.remove(out_file_name)


def import_recording(vid_base, vid_path=".", video_project="MULTICAMERA_TEST", legacy_flip=None, skip_connection=False):
    import os
    import json
    import numpy as np
    import datajoint as dj
    from datetime import datetime

    from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording, SingleCameraVideo
    from pose_pipeline import Video

    # search for files. expects them to be in the format vid_base.serial_number.mp4
    vids = []
    camera_names = []
    for v in os.listdir(vid_path):
        base, ext = os.path.splitext(v)
        if ext == ".mp4" and len(base.split(".")) == 2 and base.split(".")[0] == vid_base:
            vids.append(os.path.join(vid_path, v))

    print(f"Found {len(vids)} videos.")

    def mysplit(x):
        splits = x.split("_")
        base = "_".join(splits[:-2])
        date = "_".join(splits[-2:])

        return base, date

    camera_names = [os.path.split(v)[1].split(".")[1] for v in vids]

    # Loading the JSON file corresponding to the calibration to get the hash
    with open(os.path.join(vid_path,f"{vid_base}.json"),'r') as f:
        output_json = json.load(f)

    camera_hash = output_json["camera_config_hash"]

    _, timestamp = mysplit(vid_base)
    timestamp = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")

    parent = {
        "recording_timestamps": timestamp,
        "camera_config_hash": camera_hash,
        "video_project": video_project,
        "video_base_filename": vid_base,
    }

    frame_timestamps = np.array(output_json["timestamps"])
    if "serials" in output_json.keys():
        serials = output_json["serials"]
    else:
        assert legacy_flip is not None, "Please specify flip direction for videos without serial numbers"
        if legacy_flip:
            serials = ["UnknownLeft", "UnknownRight"]
        else:
            serials = ["UnknownRight", "UnknownLeft"]

    assert all(np.sort(serials) == np.sort(camera_names))

    vid_structs = []
    single_structs = []
    for v, serial in zip(vids, camera_names):
        vid_filename = os.path.split(v)[1]
        vid_filename = os.path.splitext(vid_filename)[0]

        vid_struct = {"video_project": video_project, "filename": vid_filename, "start_time": timestamp, "video": v}

        ts_idx = serials.index(serial)
        single_struct = {
            "recording_timestamps": timestamp,
            "camera_config_hash": camera_hash,
            "camera_name": serial,
            "video_project": video_project,
            "filename": vid_filename,
            "frame_timestamps": list(frame_timestamps[:, ts_idx]),
        }

        vid_structs.append(vid_struct)
        single_structs.append(single_struct)

    if MultiCameraRecording & parent:
        print("Recording already exists. Skipping.")
        return parent

    if skip_connection:
        MultiCameraRecording.insert1(parent)
        Video.insert(vid_structs, skip_duplicates=True)
        SingleCameraVideo.insert(single_structs)

    else:
        dj.conn().start_transaction()
        try:
            MultiCameraRecording.insert1(parent)
            Video.insert(vid_structs, skip_duplicates=True)
            SingleCameraVideo.insert(single_structs)
        except Exception as e:
            dj.conn().cancel_transaction()
            raise e
        else:
            dj.conn().commit_transaction()

    return parent
