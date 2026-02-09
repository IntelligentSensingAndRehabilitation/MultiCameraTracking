import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from multi_camera.analysis.aruco_validation import (
    CameraDetectionResult,
    MarkerDetection,
    WalkwayMarkerConfig,
    compute_coverage_matrix,
    create_aruco_detector,
    detect_markers_in_video,
    detect_markers_multi_camera,
    discover_camera_videos,
    draw_aruco_overlay,
    generate_summary_report,
    make_aruco_grid_video,
)


# ── Task 1: Config dataclass ──


def test_default_config_marker_ids():
    config = WalkwayMarkerConfig()
    assert config.all_marker_ids == [1, 2, 3, 4, 5, 6, 7, 8]


def test_default_config_pair_lookup():
    config = WalkwayMarkerConfig()
    assert config.pair_for_id[1] == "overall_start"
    assert config.pair_for_id[4] == "timing_start"
    assert config.pair_for_id[5] == "timing_stop"
    assert config.pair_for_id[8] == "cooldown_end"


def test_default_config_dictionary():
    config = WalkwayMarkerConfig()
    assert config.dictionary_id == cv2.aruco.DICT_4X4_250


def test_custom_config():
    config = WalkwayMarkerConfig(
        marker_pairs={"start": (10, 11), "end": (20, 21)},
        min_cameras=3,
    )
    assert config.all_marker_ids == [10, 11, 20, 21]
    assert config.min_cameras == 3


# ── Helpers ──


def _make_test_video(
    marker_ids: list[int], num_frames: int = 10, fps: float = 30.0
) -> str:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255

    for i, mid in enumerate(marker_ids):
        marker_img = cv2.aruco.generateImageMarker(dictionary, mid, 80)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        x_offset = 50 + i * 120
        y_offset = 200
        canvas[y_offset : y_offset + 80, x_offset : x_offset + 80] = marker_bgr

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
    for _ in range(num_frames):
        writer.write(canvas)
    writer.release()
    return path


# ── Task 2: Single-video detection ──


def test_detect_markers_in_video():
    video_path = _make_test_video([1, 3, 5])
    try:
        detector = create_aruco_detector()
        result = detect_markers_in_video(
            video_path, detector, expected_ids=[1, 2, 3, 4, 5, 6, 7, 8]
        )
        assert isinstance(result, CameraDetectionResult)
        assert result.total_frames == 10
        assert result.detection_rate(1) == 1.0
        assert result.detection_rate(3) == 1.0
        assert result.detection_rate(5) == 1.0
        assert result.detection_rate(2) == 0.0
    finally:
        os.remove(video_path)


# ── Task 3: Multi-camera parallel detection ──


def test_detect_markers_multi_camera():
    vid1 = _make_test_video([1, 2, 3])
    vid2 = _make_test_video([3, 5, 7])
    try:
        video_paths = {"cam_a": vid1, "cam_b": vid2}
        results = detect_markers_multi_camera(video_paths)
        assert set(results.keys()) == {"cam_a", "cam_b"}
        assert results["cam_a"].detection_rate(1) == 1.0
        assert results["cam_a"].detection_rate(5) == 0.0
        assert results["cam_b"].detection_rate(5) == 1.0
    finally:
        os.remove(vid1)
        os.remove(vid2)


# ── Task 4: Coverage matrix + summary report ──


def test_coverage_matrix():
    vid1 = _make_test_video([1, 2, 3, 4, 5, 6, 7, 8])
    vid2 = _make_test_video([1, 2])
    try:
        results = detect_markers_multi_camera({"cam_a": vid1, "cam_b": vid2})
        config = WalkwayMarkerConfig()
        coverage = compute_coverage_matrix(results, config.all_marker_ids)
        assert coverage.rates.shape == (8, 2)
        assert coverage.rates[0, 0] == 1.0  # marker 1, cam_a
        assert coverage.rates[0, 1] == 1.0  # marker 1, cam_b
        assert coverage.rates[4, 1] == 0.0  # marker 5, cam_b
    finally:
        os.remove(vid1)
        os.remove(vid2)


def test_summary_report_pass():
    vid1 = _make_test_video([1, 2, 3, 4, 5, 6, 7, 8])
    vid2 = _make_test_video([1, 2, 3, 4, 5, 6, 7, 8])
    try:
        results = detect_markers_multi_camera({"cam_a": vid1, "cam_b": vid2})
        config = WalkwayMarkerConfig()
        coverage = compute_coverage_matrix(results, config.all_marker_ids)
        report = generate_summary_report(coverage, config)
        assert "PASS" in report
        assert "overall_start" in report
    finally:
        os.remove(vid1)
        os.remove(vid2)


# ── Task 5: Video discovery ──


def test_discover_camera_videos(tmp_path: Path):
    (tmp_path / "rec_20260113_120000.23106516.mp4").touch()
    (tmp_path / "rec_20260113_120000.23106529.mp4").touch()
    (tmp_path / "rec_20260113_120000.23106530.mp4").touch()
    (tmp_path / "unrelated.mp4").touch()

    found = discover_camera_videos("rec_20260113_120000", str(tmp_path))
    assert len(found) == 3
    assert "23106516" in found
    assert "unrelated" not in str(found.values())


# ── Task 6: ArUco overlay drawing ──


def test_draw_aruco_overlay():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
    marker_img = cv2.aruco.generateImageMarker(dictionary, 1, 80)
    marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    canvas[200:280, 50:130] = marker_bgr

    detector = create_aruco_detector()
    corners, ids, _ = detector.detectMarkers(canvas)
    detections = [
        MarkerDetection(
            marker_id=int(ids[0][0]),
            corners=corners[0][0],
            center=np.mean(corners[0][0], axis=0),
        )
    ]

    config = WalkwayMarkerConfig()
    result = draw_aruco_overlay(canvas.copy(), detections, config)
    assert result.shape == canvas.shape
    assert not np.array_equal(result, canvas)


# ── Task 7: Multi-view grid overlay video ──


def test_make_aruco_grid_video(tmp_path: Path):
    vid1 = _make_test_video([1, 2], num_frames=5)
    vid2 = _make_test_video([3, 4], num_frames=5)
    try:
        video_paths = {"cam_a": vid1, "cam_b": vid2}
        results = detect_markers_multi_camera(video_paths)
        output = str(tmp_path / "grid.mp4")
        make_aruco_grid_video(
            video_paths, results, WalkwayMarkerConfig(), output, downsample=2
        )
        assert Path(output).exists()
        cap = cv2.VideoCapture(output)
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 5
        cap.release()
    finally:
        os.remove(vid1)
        os.remove(vid2)
