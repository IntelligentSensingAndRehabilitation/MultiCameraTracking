from __future__ import annotations

import concurrent.futures
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class WalkwayMarkerConfig:
    """Configuration for walkway ArUco marker layout."""

    dictionary_id: int = cv2.aruco.DICT_4X4_250
    marker_pairs: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "overall_start": (1, 2),
            "timing_start": (3, 4),
            "timing_stop": (5, 6),
            "cooldown_end": (7, 8),
        }
    )
    min_cameras: int = 2
    detection_rate_threshold: float = 0.8

    @property
    def all_marker_ids(self) -> list[int]:
        ids: list[int] = []
        for left, right in self.marker_pairs.values():
            ids.extend([left, right])
        return sorted(set(ids))

    @property
    def pair_for_id(self) -> dict[int, str]:
        result: dict[int, str] = {}
        for name, (left, right) in self.marker_pairs.items():
            result[left] = name
            result[right] = name
        return result

    @property
    def pair_colors(self) -> dict[str, tuple[int, int, int]]:
        return {
            "overall_start": (0, 255, 0),
            "timing_start": (255, 0, 0),
            "timing_stop": (0, 0, 255),
            "cooldown_end": (0, 255, 255),
        }


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
        if self.total_frames == 0:
            return 0.0
        frames_with_marker = sum(
            1
            for fd in self.frame_detections
            if any(d.marker_id == marker_id for d in fd.detections)
        )
        return frames_with_marker / self.total_frames


@dataclass
class CoverageMatrix:
    rates: np.ndarray
    marker_ids: list[int]
    camera_names: list[str]

    def to_dict(self) -> dict:
        return {
            "marker_ids": self.marker_ids,
            "camera_names": self.camera_names,
            "rates": self.rates.tolist(),
        }

    def print_table(self) -> str:
        header = f"{'Marker':>10}" + "".join(f"{cam:>15}" for cam in self.camera_names)
        lines = [header, "-" * len(header)]
        for i, mid in enumerate(self.marker_ids):
            row = f"{mid:>10}"
            for j in range(len(self.camera_names)):
                rate = self.rates[i, j]
                row += f"{rate:>14.1%} "
            lines.append(row)
        return "\n".join(lines)


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
    expected_ids: set[int],
) -> list[MarkerDetection]:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(
        lab[:, :, 0]
    )
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    corners, ids, _ = detector.detectMarkers(enhanced)
    detections: list[MarkerDetection] = []

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if int(marker_id) in expected_ids:
                corner_pts = corners[i][0]
                center = np.mean(corner_pts, axis=0)
                detections.append(
                    MarkerDetection(
                        marker_id=int(marker_id), corners=corner_pts, center=center
                    )
                )

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
    expected_ids: list[int],
    camera_name: str = "",
) -> CameraDetectionResult:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    expected_set = set(expected_ids)
    frame_detections: list[FrameDetections] = []

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        detections = detect_markers_in_frame(frame, detector, expected_set)
        frame_detections.append(
            FrameDetections(frame_idx=frame_idx, detections=detections)
        )

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
    config: WalkwayMarkerConfig | None = None,
) -> dict[str, CameraDetectionResult]:
    if config is None:
        config = WalkwayMarkerConfig()

    expected_ids = config.all_marker_ids

    def _process_camera(
        cam_name: str, video_path: str
    ) -> tuple[str, CameraDetectionResult]:
        detector = create_aruco_detector(config.dictionary_id)
        result = detect_markers_in_video(
            video_path, detector, expected_ids, camera_name=cam_name
        )
        return cam_name, result

    results: dict[str, CameraDetectionResult] = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_process_camera, name, path): name
            for name, path in video_paths.items()
        }
        for future in concurrent.futures.as_completed(futures):
            cam_name, result = future.result()
            results[cam_name] = result

    return results


def compute_coverage_matrix(
    results: dict[str, CameraDetectionResult],
    marker_ids: list[int],
) -> CoverageMatrix:
    camera_names = sorted(results.keys())
    rates = np.zeros((len(marker_ids), len(camera_names)))

    for j, cam_name in enumerate(camera_names):
        for i, mid in enumerate(marker_ids):
            rates[i, j] = results[cam_name].detection_rate(mid)

    return CoverageMatrix(rates=rates, marker_ids=marker_ids, camera_names=camera_names)


def generate_summary_report(
    coverage: CoverageMatrix, config: WalkwayMarkerConfig
) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("ArUco Marker Coverage Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append(coverage.print_table())
    lines.append("")

    all_pass = True

    for pair_name, (left, right) in config.marker_pairs.items():
        left_idx = coverage.marker_ids.index(left)
        right_idx = coverage.marker_ids.index(right)

        left_cameras = sum(
            1
            for rate in coverage.rates[left_idx]
            if rate >= config.detection_rate_threshold
        )
        right_cameras = sum(
            1
            for rate in coverage.rates[right_idx]
            if rate >= config.detection_rate_threshold
        )

        left_ok = left_cameras >= config.min_cameras
        right_ok = right_cameras >= config.min_cameras
        pair_ok = left_ok and right_ok

        status = "PASS" if pair_ok else "FAIL"
        if not pair_ok:
            all_pass = False

        lines.append(
            f"  {pair_name}: [{status}] marker {left} seen by {left_cameras} cams, "
            f"marker {right} seen by {right_cameras} cams"
        )

    lines.append("")
    overall = "PASS" if all_pass else "FAIL"
    lines.append(
        f"Overall: {overall} (min {config.min_cameras} cameras required per marker, "
        f"threshold {config.detection_rate_threshold:.0%})"
    )
    lines.append("=" * 60)

    return "\n".join(lines)


def discover_camera_videos(video_base: str, video_dir: str) -> dict[str, str]:
    video_dir_path = Path(video_dir)
    pattern = f"{video_base}.*.mp4"
    found: dict[str, str] = {}

    for path in sorted(video_dir_path.glob(pattern)):
        filename = path.name
        # Extract serial: everything between base. and .mp4
        suffix = filename[
            len(video_base) + 1 : -4
        ]  # strip "{base}." prefix and ".mp4" suffix
        if suffix:
            found[suffix] = str(path)

    return found


def draw_aruco_overlay(
    frame: np.ndarray,
    detections: list[MarkerDetection],
    config: WalkwayMarkerConfig,
) -> np.ndarray:
    pair_for_id = config.pair_for_id
    pair_colors = config.pair_colors
    _, w = frame.shape[:2]
    scale = max(1, w / 640)

    for det in detections:
        pair_name = pair_for_id.get(det.marker_id)
        color = (
            pair_colors.get(pair_name, (255, 255, 255))
            if pair_name
            else (255, 255, 255)
        )

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
        text_size = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )[0]
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
    config: WalkwayMarkerConfig,
    output_path: str,
    downsample: int = 2,
    n_cols: int = 5,
) -> str:
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
                frame = draw_aruco_overlay(frame, scaled_detections, config)

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


def save_results_json(
    results: dict[str, CameraDetectionResult],
    coverage: CoverageMatrix,
    config: WalkwayMarkerConfig,
    output_path: str,
) -> None:
    data: dict = {
        "config": {
            "dictionary_id": config.dictionary_id,
            "marker_pairs": {k: list(v) for k, v in config.marker_pairs.items()},
            "min_cameras": config.min_cameras,
            "detection_rate_threshold": config.detection_rate_threshold,
        },
        "coverage": coverage.to_dict(),
        "cameras": {},
    }

    for cam_name, result in results.items():
        cam_data: dict = {
            "video_path": result.video_path,
            "total_frames": result.total_frames,
            "fps": result.fps,
            "width": result.width,
            "height": result.height,
            "detection_rates": {
                mid: result.detection_rate(mid) for mid in config.all_marker_ids
            },
        }
        data["cameras"][cam_name] = cam_data

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
