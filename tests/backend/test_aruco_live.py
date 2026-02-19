from __future__ import annotations

import cv2
import numpy as np

from multi_camera.analysis.aruco_validation import (
    MarkerDetection,
    create_aruco_detector,
    detect_markers_in_frame,
)


def _make_marker_frame(
    marker_ids: list[int], width: int = 640, height: int = 480
) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    for i, mid in enumerate(marker_ids):
        marker_img = cv2.aruco.generateImageMarker(dictionary, mid, 80)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        x_offset = 50 + i * 120
        y_offset = 200
        canvas[y_offset : y_offset + 80, x_offset : x_offset + 80] = marker_bgr
    return canvas


def test_detect_markers_in_frame():
    frame = _make_marker_frame([1, 3])
    detector = create_aruco_detector()
    detections = detect_markers_in_frame(frame, detector, expected_ids={1, 2, 3, 4})

    assert len(detections) == 2
    detected_ids = {d.marker_id for d in detections}
    assert detected_ids == {1, 3}

    for det in detections:
        assert isinstance(det, MarkerDetection)
        assert det.corners.shape == (4, 2)
        assert det.center.shape == (2,)


def test_detect_markers_in_frame_empty():
    blank = np.ones((480, 640, 3), dtype=np.uint8) * 255
    detector = create_aruco_detector()
    detections = detect_markers_in_frame(blank, detector, expected_ids={1, 2, 3, 4})

    assert detections == []


def test_detect_markers_in_frame_filters_unexpected():
    frame = _make_marker_frame([1, 3, 5])
    detector = create_aruco_detector()
    detections = detect_markers_in_frame(frame, detector, expected_ids={1, 3})

    assert len(detections) == 2
    detected_ids = {d.marker_id for d in detections}
    assert detected_ids == {1, 3}
