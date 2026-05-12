"""Generic ArUco marker detection and triangulation in multi-camera videos."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def aruco_dictionary_name(dictionary_id: int) -> str:
    """Return the cv2.aruco predefined-dictionary name for an integer ID.

    cv2.aruco exposes constants like ``cv2.aruco.DICT_4X4_250`` whose values are
    plain integers; there's no built-in inverse lookup. This walks the module
    once and returns the first matching ``DICT_*`` attribute name. Falls back
    to ``"DICT_UNKNOWN_{n}"`` if the id doesn't match a known constant.
    """
    for attr in dir(cv2.aruco):
        if attr.startswith("DICT_") and getattr(cv2.aruco, attr) == dictionary_id:
            return attr
    return f"DICT_UNKNOWN_{dictionary_id}"


@dataclass
class MarkerDetection:
    marker_id: int
    corners: np.ndarray
    center: np.ndarray


@dataclass
class FrameDetections:
    frame_idx: int
    detections: list[MarkerDetection]


@dataclass
class CameraDetectionResult:
    camera_name: str
    video_path: str
    total_frames: int
    fps: float
    width: int
    height: int
    frame_detections: list[FrameDetections]

    def detection_rate(self, marker_id: int) -> float:
        frames_processed = len(self.frame_detections)
        if frames_processed == 0:
            return 0.0
        frames_with_marker = sum(
            1 for fd in self.frame_detections if any(d.marker_id == marker_id for d in fd.detections)
        )
        return frames_with_marker / frames_processed

    def to_dict(self) -> dict:
        return {
            "camera_name": self.camera_name,
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "frame_detections": [
                {
                    "frame_idx": fd.frame_idx,
                    "detections": [
                        {"marker_id": d.marker_id, "corners": d.corners, "center": d.center} for d in fd.detections
                    ],
                }
                for fd in self.frame_detections
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraDetectionResult":
        return cls(
            camera_name=data["camera_name"],
            video_path=data["video_path"],
            total_frames=data["total_frames"],
            fps=data["fps"],
            width=data["width"],
            height=data["height"],
            frame_detections=[
                FrameDetections(
                    frame_idx=fd["frame_idx"],
                    detections=[
                        MarkerDetection(marker_id=d["marker_id"], corners=d["corners"], center=d["center"])
                        for d in fd["detections"]
                    ],
                )
                for fd in data["frame_detections"]
            ],
        )


def create_aruco_detector(
    dictionary_id: int = cv2.aruco.DICT_4X4_250,
    aggressive: bool = True,
) -> cv2.aruco.ArucoDetector:
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    parameters = cv2.aruco.DetectorParameters()
    if aggressive:
        # Smaller adaptive threshold windows help detect small/distant markers
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 53
        parameters.adaptiveThreshWinSizeStep = 4
        # Lower perimeter threshold to detect markers that appear small at oblique angles
        parameters.minMarkerPerimeterRate = 0.01
        # More bits per cell for perspective-distorted markers viewed at grazing angles
        parameters.perspectiveRemovePixelPerCell = 8
        # Allow more error bits for partially occluded or glare-affected markers
        parameters.maxErroneousBitsInBorderRate = 0.5
        # Subpixel corner refinement
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def detect_markers_in_frame(
    frame: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    expected_ids: set[int] | None = None,
) -> list[MarkerDetection]:
    """Detect ArUco markers in a frame.

    If ``expected_ids`` is provided, only markers with matching IDs are returned;
    pass ``None`` to return every marker the detector finds.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    corners, ids, _ = detector.detectMarkers(enhanced)
    detections: list[MarkerDetection] = []

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            mid = int(marker_id)
            if expected_ids is None or mid in expected_ids:
                corner_pts = corners[i][0]
                center = np.mean(corner_pts, axis=0)
                detections.append(MarkerDetection(marker_id=mid, corners=corner_pts, center=center))

    return detections


def get_marker_centers_by_frame(
    result: CameraDetectionResult,
    marker_id: int,
) -> dict[int, np.ndarray]:
    centers: dict[int, np.ndarray] = {}
    for fd in result.frame_detections:
        for det in fd.detections:
            if det.marker_id == marker_id:
                centers[fd.frame_idx] = det.center
                break
    return centers


def detect_markers_in_video(
    video_path: str,
    detector: cv2.aruco.ArucoDetector,
    expected_ids: list[int] | None = None,
    camera_name: str = "",
    frame_step: int = 1,
) -> CameraDetectionResult:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    expected_set = set(expected_ids) if expected_ids is not None else None
    frame_detections: list[FrameDetections] = []

    frame_idx = 0
    while frame_idx < total_frames:
        if frame_step > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        detections = detect_markers_in_frame(frame, detector, expected_set)
        frame_detections.append(FrameDetections(frame_idx=frame_idx, detections=detections))
        frame_idx += frame_step

    cap.release()

    return CameraDetectionResult(
        camera_name=camera_name,
        video_path=video_path,
        total_frames=total_frames,
        fps=fps,
        width=width,
        height=height,
        frame_detections=frame_detections,
    )


def detect_markers_multi_camera(
    video_paths: dict[str, str],
    expected_ids: list[int] | None = None,
    dictionary_id: int = cv2.aruco.DICT_4X4_250,
    frame_step: int = 1,
    max_workers: int | None = 4,
    cv2_threads_per_worker: int | None = 1,
) -> dict[str, CameraDetectionResult]:
    """Run ArUco detection across multiple synchronized camera videos in parallel.

    ``expected_ids=None`` returns every marker the detector finds.

    ``cv2_threads_per_worker`` caps OpenCV's internal TBB threading per detect
    call. Without it, each ``cv2.aruco.detectMarkers`` call fans out across
    every CPU and the outer ``max_workers`` pool stops being a meaningful
    concurrency limit. Effective core count ≈ ``max_workers * cv2_threads_per_worker``.
    The previous global value is restored on return so unrelated cv2 work in
    the same process is unaffected. Pass ``None`` to leave OpenCV's thread
    setting untouched.
    """
    prev_cv2_threads: int | None = None
    if cv2_threads_per_worker is not None:
        prev_cv2_threads = cv2.getNumThreads()
        cv2.setNumThreads(cv2_threads_per_worker)

    def _process_camera(cam_name: str, video_path: str) -> tuple[str, CameraDetectionResult]:
        detector = create_aruco_detector(dictionary_id)
        result = detect_markers_in_video(
            video_path, detector, expected_ids, camera_name=cam_name, frame_step=frame_step
        )
        return cam_name, result

    try:
        results: dict[str, CameraDetectionResult] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_camera, name, path): name for name, path in video_paths.items()}
            for future in concurrent.futures.as_completed(futures):
                cam_name, result = future.result()
                results[cam_name] = result
        return results
    finally:
        if prev_cv2_threads is not None:
            cv2.setNumThreads(prev_cv2_threads)


def triangulate_detected_markers_detailed(
    detection_results: dict[str, CameraDetectionResult],
    cgroup,
    marker_ids: list[int] | None = None,
    min_cameras: int = 2,
) -> dict[int, dict]:
    """Triangulate per-marker 3D positions and per-marker spatial spread.

    Returns ``{marker_id: {"position": median_xyz_mm,
                           "spread_mm": float,
                           "n_frames": int}}``.

    ``spread_mm`` is the median Euclidean distance from each per-frame
    triangulation to the consensus median position — a robust scalar measure of
    how stable the marker's triangulated location is across frames. Physically
    fixed markers have low spread (a few mm, dominated by detection noise);
    moving or transient markers have high spread.

    ``cgroup`` is an aniposelib ``CameraGroup`` constructed from the calibration.
    If ``marker_ids`` is ``None``, every marker ID seen in any camera is
    triangulated. Markers seen by fewer than ``min_cameras`` on every frame are
    omitted from the result.
    """
    camera_names = sorted(detection_results.keys())
    num_cams = len(camera_names)

    if marker_ids is None:
        seen: set[int] = set()
        for result in detection_results.values():
            for fd in result.frame_detections:
                for d in fd.detections:
                    seen.add(d.marker_id)
        marker_ids = sorted(seen)

    out: dict[int, dict] = {}

    for marker_id in marker_ids:
        cam_frame_centers: list[dict[int, np.ndarray]] = []
        for cam_name in camera_names:
            centers = get_marker_centers_by_frame(detection_results[cam_name], marker_id)
            cam_frame_centers.append(centers)

        all_frames: set[int] = set()
        for centers in cam_frame_centers:
            all_frames.update(centers.keys())

        valid_frames = []
        for frame_idx in sorted(all_frames):
            n_visible = sum(1 for centers in cam_frame_centers if frame_idx in centers)
            if n_visible >= min_cameras:
                valid_frames.append(frame_idx)

        if not valid_frames:
            continue

        pts2d = np.full((num_cams, len(valid_frames), 2), np.nan, dtype=np.float64)
        for fi, frame_idx in enumerate(valid_frames):
            for cam_i, centers in enumerate(cam_frame_centers):
                if frame_idx in centers:
                    pts2d[cam_i, fi, :] = centers[frame_idx]

        pts3d = cgroup.triangulate(pts2d)  # (N, 3)
        median_pos = np.median(pts3d, axis=0)
        # Median absolute deviation from the consensus position — robust to outliers
        spread_mm = float(np.median(np.linalg.norm(pts3d - median_pos, axis=1)))
        out[marker_id] = {
            "position": median_pos,
            "spread_mm": spread_mm,
            "n_frames": len(valid_frames),
        }

    return out


def triangulate_detected_markers(
    detection_results: dict[str, CameraDetectionResult],
    cgroup,
    marker_ids: list[int] | None = None,
    min_cameras: int = 2,
) -> dict[int, np.ndarray]:
    """Position-only convenience wrapper around ``triangulate_detected_markers_detailed``.

    Returns ``{marker_id: median_xyz_mm}``.
    """
    detailed = triangulate_detected_markers_detailed(
        detection_results, cgroup, marker_ids=marker_ids, min_cameras=min_cameras
    )
    return {mid: info["position"] for mid, info in detailed.items()}


def positions_converged(
    prev: dict[int, np.ndarray],
    current: dict[int, np.ndarray],
    threshold_mm: float,
) -> bool:
    """Check if all marker positions shifted less than threshold between iterations."""
    if not prev:
        return False
    for marker_id, pos in current.items():
        if marker_id not in prev:
            return False
        if np.linalg.norm(pos - prev[marker_id]) >= threshold_mm:
            return False
    return True


def discover_camera_videos(video_base: str, video_dir: str) -> dict[str, str]:
    """Find all per-camera videos for a given recording base.

    Returns a mapping from camera serial → full path. Filenames are expected
    in the format ``{video_base}.{camera_serial}.mp4``.
    """
    video_dir_path = Path(video_dir)
    pattern = f"{video_base}.*.mp4"
    found: dict[str, str] = {}

    for path in sorted(video_dir_path.glob(pattern)):
        filename = path.name
        # Extract serial: everything between base. and .mp4
        suffix = filename[len(video_base) + 1 : -4]  # strip "{base}." prefix and ".mp4" suffix
        if suffix:
            found[suffix] = str(path)

    return found


def draw_aruco_overlay(
    frame: np.ndarray,
    detections: list[MarkerDetection],
    color_for_id: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Draw detected ArUco markers on a frame.

    ``color_for_id`` optionally maps marker_id → BGR color. Markers without
    a mapping are drawn white.
    """
    color_for_id = color_for_id or {}
    _, w = frame.shape[:2]
    scale = max(1, w / 640)

    for det in detections:
        color = color_for_id.get(det.marker_id, (255, 255, 255))

        corners_int = det.corners.astype(int)
        center = det.center.astype(int)

        padding_factor = 0.5
        padded = center + (corners_int - center) * (1 + padding_factor)
        padded = padded.astype(int)

        cv2.polylines(
            frame,
            [padded],
            isClosed=True,
            color=color,
            thickness=max(1, int(3 * scale)),
        )

        font_scale = 2.0 * scale
        thickness = max(2, int(3 * scale))
        text = str(det.marker_id)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_pos = (
            int(center[0] - text_size[0] / 2),
            int(center[1] + text_size[1] / 2),
        )
        cv2.putText(
            frame,
            text,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
        )

    return frame


def make_aruco_grid_video(
    video_paths: dict[str, str],
    results: dict[str, CameraDetectionResult],
    output_path: str,
    color_for_id: dict[int, tuple[int, int, int]] | None = None,
    downsample: int = 2,
    n_cols: int = 5,
) -> str:
    """Render a grid video showing all cameras side-by-side with detection overlays."""
    camera_names = sorted(video_paths.keys())
    caps = {name: cv2.VideoCapture(video_paths[name]) for name in camera_names}

    first_result = results[camera_names[0]]
    total_frames = min(r.total_frames for r in results.values())
    fps = first_result.fps

    sample_width = int(first_result.width / downsample)
    sample_height = int(first_result.height / downsample)

    # Build detection lookup: camera -> frame_idx -> list[MarkerDetection]
    detection_lookup: dict[str, dict[int, list[MarkerDetection]]] = {}
    for cam_name, result in results.items():
        cam_lookup: dict[int, list[MarkerDetection]] = {}
        for fd in result.frame_detections:
            cam_lookup[fd.frame_idx] = fd.detections
        detection_lookup[cam_name] = cam_lookup

    def images_to_grid(images: list[np.ndarray], n_cols: int = n_cols) -> np.ndarray:
        n_rows = int(np.ceil(len(images) / n_cols))
        grid = np.zeros(
            (n_rows * images[0].shape[0], n_cols * images[0].shape[1], 3),
            dtype=np.uint8,
        )
        for i, img in enumerate(images):
            row = i // n_cols
            col = i % n_cols
            grid[
                row * img.shape[0] : (row + 1) * img.shape[0],
                col * img.shape[1] : (col + 1) * img.shape[1],
                :,
            ] = img
        return grid

    # Compute grid dimensions for VideoWriter
    actual_cols = min(n_cols, len(camera_names))
    n_rows = int(np.ceil(len(camera_names) / actual_cols))
    grid_width = actual_cols * sample_width
    grid_height = n_rows * sample_height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (grid_width, grid_height))

    label_scale = max(0.3, sample_width / 640)
    label_thickness = max(1, int(label_scale * 2))

    for frame_idx in range(total_frames):
        frames: list[np.ndarray] = []
        for cam_name in camera_names:
            ret, frame = caps[cam_name].read()
            if not ret or frame is None:
                frame = np.zeros((sample_height, sample_width, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (sample_width, sample_height))

            detections = detection_lookup.get(cam_name, {}).get(frame_idx, [])
            if detections:
                # Scale detections to match downsampled frame
                scaled_detections = []
                for d in detections:
                    scale_factor = 1.0 / downsample
                    scaled = MarkerDetection(
                        marker_id=d.marker_id,
                        corners=d.corners * scale_factor,
                        center=d.center * scale_factor,
                    )
                    scaled_detections.append(scaled)
                frame = draw_aruco_overlay(frame, scaled_detections, color_for_id)

            cv2.putText(
                frame,
                cam_name,
                (5, int(20 * label_scale + 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_scale,
                (255, 255, 255),
                label_thickness,
            )
            frames.append(frame)

        grid = images_to_grid(frames, n_cols=actual_cols)
        writer.write(grid)

    writer.release()
    for cap in caps.values():
        cap.release()

    return output_path
