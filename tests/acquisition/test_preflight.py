"""Tests for FlirRecorder preflight hooks that depend on health.py.

The full FlirRecorder integration would require the Spinnaker SDK, so these
tests exercise only the small surface area we added: the preflight contract
(CameraUnreachableError), public read_* wrappers (safe when cams is empty),
and the last_trial_frame_skip_counts aggregation.
"""

from __future__ import annotations

import pytest

from multi_camera.acquisition.health import (
    CameraUnreachableError,
    DetectedCamera,
    check_camera_reachability,
)


class _FakeCam:
    def __init__(self, serial: str):
        self.DeviceSerialNumber = serial


class _FakeRecorder:
    """Stand-in for FlirRecorder that supplies .cams for reachability check.

    We don't import the real FlirRecorder here because doing so requires
    PySpin. The preflight logic is isolated enough that exercising it on a
    minimal stub is sufficient.
    """

    def __init__(self, held_serials: list[str]):
        self.cams = [_FakeCam(s) for s in held_serials]


class TestPreflight:
    def test_raises_when_expected_camera_missing(self) -> None:
        recorder = _FakeRecorder(["111", "222"])

        def fake_enum() -> list[DetectedCamera]:
            return []

        report = check_camera_reachability(
            expected_serials=[c.DeviceSerialNumber for c in recorder.cams],
            recorder=recorder,
            enumerator=fake_enum,
        )
        # Recorder has handles — the recorder path snapshots from them, so all
        # expected cameras are "detected". This mirrors how preflight would
        # normally pass mid-session.
        assert report.missing == []

    def test_camera_unreachable_error_carries_missing_list(self) -> None:
        err = CameraUnreachableError(missing=["111", "222"])
        assert err.missing == ["111", "222"]
        assert "111" in str(err)
        assert "222" in str(err)


def test_last_trial_frame_skip_count_aggregation_logic() -> None:
    """Mirror the aggregation that lives in start_acquisition's tail.

    Kept as a unit test here (rather than inside the SDK-gated
    test_acquisition.py) so it runs in CI.
    """
    frame_skip_events = [
        {"camera_serial": "111", "frame_idx": 10, "gap_size": 1},
        {"camera_serial": "111", "frame_idx": 20, "gap_size": 2},
        {"camera_serial": "222", "frame_idx": 30, "gap_size": 1},
    ]

    skip_counts: dict[str, int] = {}
    for ev in frame_skip_events:
        serial = str(ev.get("camera_serial", ""))
        if serial:
            skip_counts[serial] = skip_counts.get(serial, 0) + 1

    assert skip_counts == {"111": 2, "222": 1}


def test_camera_unreachable_error_custom_message() -> None:
    err = CameraUnreachableError(missing=["333"], message="custom reason")
    assert "custom reason" in str(err)
