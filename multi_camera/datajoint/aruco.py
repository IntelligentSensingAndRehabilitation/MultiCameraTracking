"""ArUco-marker DataJoint tables for multi-camera calibration.

``CalibrationArucoDetection`` stores per-camera pixel detections of ArUco
markers in the calibration videos. Detection is generic — every marker the
detector finds is kept, no protocol-specific filtering. Detection
auto-populates for any calibration whose recording comment contains the
substring ``"aruco"`` (set in the acquisition GUI and propagated through
``MultiCameraCalibration.comment``).

Protocol-specific interpretation (e.g. 10MWT walkway goalposts) lives
downstream in consumer packages, which own their own per-protocol flag tables
keyed off ``Calibration``.
"""

import cv2
import datajoint as dj

from .calibrate_cameras import Calibration
from .multi_camera_dj import CalibrationVideos, MultiCameraCalibration
from pose_pipeline import Video

schema = dj.schema("multicamera_tracking")


DEFAULT_FRAME_STEP = 10
DEFAULT_DICTIONARY_ID = cv2.aruco.DICT_4X4_250


@schema
class CalibrationArucoDetection(dj.Computed):
    definition = """
    # Per-camera pixel detections of ArUco markers in calibration videos
    -> Calibration
    ---
    pixel_detections           : longblob    # dict[cam_name -> CameraDetectionResult.to_dict()]
    num_markers_found          : int         # count of unique marker IDs detected
    marker_ids_found           : longblob    # list[int] sorted
    aruco_dictionary           : varchar(50) # cv2.aruco predefined-dict name, e.g. "DICT_4X4_250"
    frame_step                 : int         # frames sampled (every Nth)
    marker_position_spread_mm  : longblob    # dict[marker_id -> median absolute deviation (mm) of triangulated positions across frames]
    marker_n_frames_detected   : longblob    # dict[marker_id -> int] frames each marker was triangulatable in (>= min_cameras visible)
    """

    @property
    def key_source(self):
        # Only attempt detection on calibrations whose recording comment
        # mentions aruco markers — set in the acquisition GUI at record time
        # and propagated through MultiCameraCalibration on push to DataJoint.
        return Calibration & (MultiCameraCalibration & 'comment LIKE "%aruco%"')

    def make(self, key):
        from multi_camera.analysis.aruco import (
            aruco_dictionary_name,
            detect_markers_multi_camera,
            triangulate_detected_markers_detailed,
        )
        from multi_camera.analysis.calibration import recreate_cgroup_from_entry

        frame_step = DEFAULT_FRAME_STEP
        dictionary_id = DEFAULT_DICTIONARY_ID
        dictionary_name = aruco_dictionary_name(dictionary_id)

        # Fetch per-camera video paths via CalibrationVideos → Video
        video_query = (Video & (CalibrationVideos & key)).proj("video", "filename")
        video_rows = video_query.fetch(as_dict=True)
        if not video_rows:
            raise FileNotFoundError(
                f"No CalibrationVideos rows linked to {key} — push calibration videos first."
            )

        video_paths: dict[str, str] = {}
        for row in video_rows:
            # Filename is "{recording_base}.{camera_serial}" — last dot-segment is the serial
            cam_serial = row["filename"].rsplit(".", 1)[-1]
            video_paths[cam_serial] = row["video"]

        cal_entry = (Calibration & key).fetch1()
        cgroup = recreate_cgroup_from_entry(cal_entry)

        print(f"[CalibrationArucoDetection] Detecting ArUco markers for {key}")
        print(f"  {len(video_paths)} cameras, dictionary={dictionary_name}, frame_step={frame_step}")

        pixel_detections = detect_markers_multi_camera(
            video_paths,
            expected_ids=None,  # generic: keep every marker the detector finds
            dictionary_id=dictionary_id,
            frame_step=frame_step,
        )

        # Triangulate every detected marker, recording per-marker spatial spread.
        # Stable markers (physically fixed) have low spread; transient markers
        # (e.g. someone walking past with a marker, equipment moving in/out of
        # frame) have high spread. Downstream protocol-specific tables apply
        # their own spread threshold over the markers they care about.
        detailed = triangulate_detected_markers_detailed(pixel_detections, cgroup)

        marker_ids_found = sorted(detailed.keys())
        marker_position_spread_mm = {mid: info["spread_mm"] for mid, info in detailed.items()}
        marker_n_frames_detected = {mid: info["n_frames"] for mid, info in detailed.items()}

        print(f"  Found {len(marker_ids_found)} markers: {marker_ids_found}")
        print(f"  Per-marker spatial spread (mm), worst → best:")
        for mid, spread in sorted(marker_position_spread_mm.items(), key=lambda kv: -kv[1]):
            n = marker_n_frames_detected[mid]
            print(f"    Marker {mid:>4d}: spread={spread:7.2f}mm  (n_frames={n})")

        serialized = {cam: result.to_dict() for cam, result in pixel_detections.items()}

        self.insert1(
            {
                **key,
                "pixel_detections": serialized,
                "num_markers_found": len(marker_ids_found),
                "marker_ids_found": marker_ids_found,
                "aruco_dictionary": dictionary_name,
                "frame_step": frame_step,
                "marker_position_spread_mm": marker_position_spread_mm,
                "marker_n_frames_detected": marker_n_frames_detected,
            }
        )
