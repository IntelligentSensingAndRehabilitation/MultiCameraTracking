import os

import yaml
import numpy as np
import datajoint as dj

from pose_pipeline import Video, VideoInfo, BottomUpPeople
from .multi_camera_dj import (
    schema,
    MultiCameraRecording,
    SingleCameraVideo,
    Calibration,
    CalibratedRecording,
)

_analysis_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "analysis"
)

TRACKING_CONFIGS = [
    os.path.join(_analysis_dir, "mvmp1f_default.yml"),
    os.path.join(_analysis_dir, "mvmp1f_fallback1.yml"),
    os.path.join(_analysis_dir, "mvmp1f_fallback3.yml"),
    os.path.join(_analysis_dir, "mvmp1f_fallback4.yml"),
    os.path.join(_analysis_dir, "mvmp1f_fallback2.yml"),
]


def _load_config_settings(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


# Hardcoding a selected method
bottom_up = BottomUpPeople & {"bottom_up_method_name": "Bridging_OpenPose"}
# bottom_up = (BottomUpPeople & {'bottom_up_method_name': 'OpenPose_HR'})

try:
    # NOTE: we do not want to require EasyMocap to access the data. This is a workaround although
    # really this could all migrate into the analysis module, but would then introduce DJ dependencies
    # into the analysis, which is a practice we attempt to avoid.

    from easymocap.dataset.base import MVBase
    from easymocap.dataset import CONFIG
    from easymocap.mytools.file_utils import get_bbox_from_pose

    def _build_camera(params, index=0):
        from multi_camera.analysis.camera import (
            get_intrinsic,
            get_extrinsic,
            get_projection,
        )

        cam = {}

        params = params.copy()
        params["tvec"] = (
            params["tvec"] / 1000.0
        )  # have the final matrices end up in meters

        # cam['K'] = np.array(params[index]['matrix'])
        cam["K"] = np.array(get_intrinsic(params, index))
        cam["RT"] = np.array(get_extrinsic(params, index)[:3])
        cam["P"] = np.array(get_projection(params, index))

        cam["invK"] = np.linalg.inv(cam["K"])
        cam["Rvec"] = params["rvec"][index, :, None]
        cam["T"] = cam["RT"][:3, -1, None]
        cam["R"] = cam["RT"][:, :3]
        cam["center"] = -params["rvec"][index].T @ cam["T"]
        cam["dist"] = params["dist"][index]
        return cam

    class MCTDataset(MVBase):
        """
        Provides an EasyMocap compatible interface to MultiCamera / PosePipe
        """

        def __init__(self, key, filter2d=None, images=False):
            self.images = images

            assert len(SingleCameraVideo & key) == len(
                (bottom_up * SingleCameraVideo & key)
            ), "Missing BottomUpPeople OpenPose computations for some cameras"
            assert len(SingleCameraVideo & key) > 0, (
                "No cameras found for this recording"
            )
            # TODO: should support general bottom up class and thus other types of keypoints
            self.keypoints, self.cams = (bottom_up * SingleCameraVideo & key).fetch(
                "keypoints", "camera_name"
            )
            self.calibration, calibration_cameras = (Calibration & key).fetch1(
                "camera_calibration", "camera_names"
            )

            n_frames = [len(k) for k in self.keypoints]
            zero_frames = [self.cams[i] for i, n in enumerate(n_frames) if n == 0]
            if len(zero_frames) > 0:
                missing = (
                    Video
                    & (
                        SingleCameraVideo
                        & key
                        & f"camera_name IN ({','.join(zero_frames)})"
                    )
                ).fetch("KEY")
                raise ValueError(f"Missing keypoints for {zero_frames}. {missing}")

            calibration_idx = np.array(
                [calibration_cameras.index(c) for c in self.cams]
            )
            if len(calibration_idx) != len(self.cams):
                print(f"Keeping cameras: {calibration_idx}")
            for k in self.calibration.keys():
                self.calibration[k] = self.calibration[k][calibration_idx]
            calibration_cameras = self.cams

            if self.images:
                import pims

                videos = (Video * bottom_up * SingleCameraVideo & key).fetch("video")
                self.caps = [pims.Video(v) for v in videos]

            self.frames = np.unique(
                (VideoInfo * SingleCameraVideo & key).fetch("num_frames")
            )
            # assert len(self.frames) == 1
            self.frames = np.min(self.frames)
            self.width = np.unique(
                (VideoInfo * SingleCameraVideo & key).fetch("width")
            )[0]
            self.height = np.unique(
                (VideoInfo * SingleCameraVideo & key).fetch("height")
            )[0]

            # set some default values. these fields are expected and would be created by
            # calling the parent class constructor, if it wouldn't break for us
            self.kpts_type = "body25"  # openpose
            self.config = CONFIG[self.kpts_type]
            self.ret_crop = False
            self.undis = True
            self.nViews = len(self.cams)

            # set up cameras
            self.cameras = {
                c: _build_camera(self.calibration, i)
                for i, c in enumerate(calibration_cameras)
            }
            self.cameras["basenames"] = calibration_cameras
            self.Pall = np.stack(
                [
                    self.cameras[cam]["K"]
                    @ np.hstack((self.cameras[cam]["R"], self.cameras[cam]["T"]))
                    for cam in self.cams
                ]
            )

            # some additional variables normally set in parent to reproduce complete behavior
            self.filter2d = filter2d
            if filter2d is not None:
                self.filter2d = make_filter(filter2d)

        def __len__(self) -> int:
            return self.frames

        def __getitem__(self, index: int):
            if self.images:
                images = [np.array(c[index]) for c in self.caps]
            else:
                images = []

            def _parse_people(keypoints):
                # split into dictionary format needed downstream
                if keypoints is None:
                    return []
                return [
                    {"keypoints": k.copy(), "bbox": get_bbox_from_pose(k)}
                    for k in keypoints
                ]  # iterate over first axis

            # reformat all the multi
            annots = [_parse_people(k[index]) for k in self.keypoints]

            if self.undis:
                if self.images:
                    images = self.undistort(images)
                annots = self.undis_det(annots)

            return images, annots

except ImportError:
    print("EasyMocap not installed. Will not be able to populate new entries")


@schema
class SkippedRecording(dj.Manual):
    definition = """
    # Recordings that failed all tracking configs
    participant_id         : varchar(50)
    session_date           : date
    recording_timestamps   : timestamp
    camera_config_hash     : varchar(50)
    cal_timestamp          : timestamp
    ---
    video_project          : varchar(50)
    video_base_filename    : varchar(100)
    recording_comment      : varchar(255)
    skip_reason            : varchar(255)
    configs_tried          : longblob
    insertion_time = CURRENT_TIMESTAMP : timestamp
    """


@schema
class EasymocapTracking(dj.Computed):
    definition = """
    # Use EasyMocap to track and associate people in the view
    -> CalibratedRecording
    ---
    tracking_results     : longblob
    num_tracks           : int
    tracking_config = NULL : longblob   # config settings used for tracking
    """

    def make(self, key):
        import gc
        import ctypes
        from multi_camera.analysis.easymocap import mvmp_association_and_tracking

        assert len((SingleCameraVideo & key) - bottom_up) == 0, (
            f"Missing OpenPose computations for {key}"
        )

        dataset = MCTDataset(key)
        libc = ctypes.CDLL("libc.so.6")

        for i, config_path in enumerate(TRACKING_CONFIGS):
            try:
                results = mvmp_association_and_tracking(
                    dataset, config_file=config_path
                )
                key["tracking_results"] = results
                key["num_tracks"] = len(
                    np.unique([k["id"] for r in results for k in r])
                )
                key["tracking_config"] = _load_config_settings(config_path)
                self.insert1(key)
                return
            except np.linalg.LinAlgError:
                print(
                    f"Config {i + 1}/{len(TRACKING_CONFIGS)} failed with LinAlgError, trying next..."
                )
                gc.collect()
                libc.malloc_trim(0)
                continue

        self._log_skipped(key)

    def _log_skipped(self, key):
        from .sessions import Recording

        recording = (Recording & key).fetch1()
        mcr = (MultiCameraRecording & key).fetch1()

        SkippedRecording.insert1(
            {
                **key,
                "participant_id": recording["participant_id"],
                "session_date": recording["session_date"],
                "recording_comment": recording["comment"],
                "video_project": mcr["video_project"],
                "video_base_filename": mcr["video_base_filename"],
                "skip_reason": "All tracking configs failed with LinAlgError (SVD did not converge)",
                "configs_tried": [_load_config_settings(p) for p in TRACKING_CONFIGS],
            }
        )

        print(f"All configs failed for {key}. Inserted into SkippedRecording.")

    @property
    def key_source(self):
        return (
            CalibratedRecording
            & MultiCameraRecording
            - (SingleCameraVideo - bottom_up).proj()
            - SkippedRecording
        )

    def create_bounding_boxes(self, subject_ids):
        """Create bounding boxes based on Easymocap tracks

        Will insert the entries into TrackingBboxMethod and populates PersonBbox. Using
        python apps/visualize_easymocap.py filename --filter subject_ids can allow
        checking for the person of interest.

        Params:
            self (EasymocapTracking) : should be restricted to a single entry
            subjects_ids : List[int] : list of subjects
        """

        from pose_pipeline import (
            TrackingBbox,
            TrackingBboxMethod,
            TrackingBboxMethodLookup,
            PersonBbox,
            PersonBboxValid,
        )

        # Add Easymocap as a tracking method if it is not already in TrackingBboxMethodLookup
        if not TrackingBboxMethodLookup & {"tracking_method": 21}:
            TrackingBboxMethodLookup.insert1(
                {"tracking_method": 21, "tracking_method_name": "Easymocap"}
            )

        results = self.fetch1("tracking_results")

        camera_names = (SingleCameraVideo * MultiCameraRecording & self).fetch(
            "camera_name"
        )
        camera_names.sort()
        N = len(SingleCameraVideo & self)

        def parse_frame(r):
            person = [p for p in r if p["id"] in subject_ids]
            if len(person) == 0 or "bbox" not in person[0].keys():
                return np.zeros((N, 5))
            else:
                return person[0]["bbox"]

        bbox = np.array([parse_frame(r) for r in results])

        output_bboxes = {i: [] for i in range(N)}
        for i, b in enumerate(bbox):
            for j in range(N):
                if b[j, -1]:
                    t = {
                        "track_id": 1,
                        "tlbr": b[j, :-1],
                        "tlhw": np.array(
                            [b[j, 0], b[j, 1], b[j, 2] - b[j, 0], b[j, 3] - b[j, 1]]
                        ),
                    }
                    output_bboxes[j].append([t])
                    # print(i,j,len(output_bboxes[j]))
                else:
                    output_bboxes[j].append([])

        for i, c in enumerate(camera_names):
            vid_key = (
                Video
                & (SingleCameraVideo * MultiCameraRecording & self & {"camera_name": c})
            ).fetch1("KEY")
            vid_key["tracking_method"] = 21

            print(vid_key)

            TrackingBboxMethod.insert1(vid_key, skip_duplicates=True)

            track_key = vid_key.copy()
            track_key["tracks"] = output_bboxes[i]
            track_key["num_tracks"] = 1
            TrackingBbox.insert1(
                track_key, skip_duplicates=True, allow_direct_insert=True
            )

            valid_key = vid_key.copy()
            valid_key["keep_tracks"] = [1]
            valid_key["video_subject_id"] = 0
            PersonBboxValid.insert1(valid_key, skip_duplicates=True)

            PersonBbox.populate(vid_key)


@schema
class EasymocapSmpl(dj.Computed):
    definition = """
    # Use EasyMocap to track and associate people in the view
    -> EasymocapTracking
    ---
    smpl_results         : longblob
    """

    def make(self, key):
        from ..analysis.easymocap import fit_multiple_smpl

        results = (EasymocapTracking & key).fetch1("tracking_results")
        key["smpl_results"] = fit_multiple_smpl(results)

        self.insert1(key)
