"""Tests for acquisition diagnostic instrumentation in json_parser.py.

All tests use synthetic JSON via tmp_path — no FLIR hardware needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from multi_camera.acquisition.diagnostics.json_parser import (
    create_camera_error_figure,
    create_queue_depth_figure,
    create_sync_wait_figure,
    diagnose_sync_issues,
    load_session,
    load_trial,
)

CAMERA_IDS = ["CAM_A", "CAM_B", "CAM_C"]
N_FRAMES = 20
BASE_TIMESTAMP_NS = 1_000_000_000_000
FRAME_PERIOD_NS = 34_400_000  # ~29.07 fps → ~34.4 ms


def _make_json_data(
    n_frames: int = N_FRAMES,
    camera_ids: list[str] | None = None,
    include_diagnostics: bool = False,
    diagnostics_version: int = 1,
    sync_wait_cycles: list[int] | None = None,
    sync_bottleneck_cameras: list[list[str]] | None = None,
    queue_depths: list[dict[str, int]] | None = None,
    frame_id_cross_delta: list[dict[str, int]] | None = None,
    camera_error_summary: dict[str, dict[str, int]] | None = None,
    camera_stream_stats: dict[str, dict[str, int]] | None = None,
    system_snapshots: list[dict] | None = None,
    ptp_offset_start: dict[str, int] | None = None,
    ptp_offset_end: dict[str, int] | None = None,
    camera_temperature_start: dict[str, float] | None = None,
    camera_temperature_end: dict[str, float] | None = None,
    timespread_alerts: dict | None = None,
    sync_timeout_events: list[dict] | None = None,
    frame_skip_events: list[dict] | None = None,
) -> dict:
    cams = camera_ids or CAMERA_IDS
    n_cams = len(cams)

    timestamps = []
    frame_ids = []
    for i in range(n_frames):
        row_ts = [
            BASE_TIMESTAMP_NS + i * FRAME_PERIOD_NS + cam_idx * 100
            for cam_idx in range(n_cams)
        ]
        row_fid = [1000 + i for _ in range(n_cams)]
        timestamps.append(row_ts)
        frame_ids.append(row_fid)

    data: dict = {
        "serials": cams,
        "timestamps": timestamps,
        "frame_id": frame_ids,
        "real_times": [[0.0] * n_cams] * n_frames,
        "exposure_times": [15000] * n_cams,
        "frame_rates_requested": [30] * n_cams,
        "frame_rates_binning": [30] * n_cams,
        "camera_config_hash": "abc123",
        "camera_info": {},
        "meta_info": {},
        "system_info": {},
    }

    if include_diagnostics:
        data["diagnostics_version"] = diagnostics_version
        data["sync_wait_cycles"] = sync_wait_cycles or [0] * n_frames
        data["sync_bottleneck_cameras"] = sync_bottleneck_cameras or [[]] * n_frames
        data["queue_depths"] = queue_depths or [{c: 1 for c in cams}] * n_frames
        data["frame_id_cross_delta"] = (
            frame_id_cross_delta or [{c: 0 for c in cams}] * n_frames
        )

    if camera_error_summary is not None:
        data["camera_error_summary"] = camera_error_summary

    if camera_stream_stats is not None:
        data["camera_stream_stats"] = camera_stream_stats

    if system_snapshots is not None:
        data["system_snapshots"] = system_snapshots

    if ptp_offset_start is not None:
        data["ptp_offset_start"] = ptp_offset_start

    if ptp_offset_end is not None:
        data["ptp_offset_end"] = ptp_offset_end

    if camera_temperature_start is not None:
        data["camera_temperature_start"] = camera_temperature_start

    if camera_temperature_end is not None:
        data["camera_temperature_end"] = camera_temperature_end

    if timespread_alerts is not None:
        data["timespread_alerts"] = timespread_alerts

    if sync_timeout_events is not None:
        data["sync_timeout_events"] = sync_timeout_events

    if frame_skip_events is not None:
        data["frame_skip_events"] = frame_skip_events

    return data


def _write_json(
    tmp_path: Path, data: dict, filename: str = "test_20250101_120000.json"
) -> Path:
    path = tmp_path / filename
    path.write_text(json.dumps(data))
    return path


class TestLegacyBackwardCompat:
    def test_legacy_json_loads_with_none_diagnostics(self, tmp_path: Path) -> None:
        data = _make_json_data(include_diagnostics=False)
        json_path = _write_json(tmp_path, data)

        trial = load_trial(json_path, trial_index=0)

        assert trial.sync_wait_cycles is None
        assert trial.sync_bottleneck_cameras is None
        assert trial.queue_depths is None
        assert trial.frame_id_cross_delta is None
        assert trial.camera_error_summary is None
        assert trial.camera_stream_stats is None
        assert trial.diagnostics_version is None
        assert trial.has_diagnostics is False

    def test_legacy_max_mean_sync_wait(self, tmp_path: Path) -> None:
        data = _make_json_data(include_diagnostics=False)
        json_path = _write_json(tmp_path, data)
        trial = load_trial(json_path, trial_index=0)

        assert trial.max_sync_wait_cycles == 0
        assert trial.mean_sync_wait_cycles == 0.0


class TestDiagnosticFieldParsing:
    def test_all_fields_parsed(self, tmp_path: Path) -> None:
        wait_cycles = [0, 2, 0, 1] + [0] * 16
        bottleneck = [[], ["CAM_C"], [], ["CAM_B"]] + [[]] * 16
        q_depths = [{c: i + 1 for c in CAMERA_IDS} for i in range(N_FRAMES)]
        cross_delta = [{"CAM_A": 0, "CAM_B": 1, "CAM_C": 0}] + [
            {c: 0 for c in CAMERA_IDS}
        ] * 19
        error_summary = {
            "CAM_A": {
                "incomplete_frames": 2,
                "exceptions": 1,
                "total_acquired": 100,
                "frame_id_gaps": 0,
            },
            "CAM_B": {
                "incomplete_frames": 0,
                "exceptions": 0,
                "total_acquired": 100,
                "frame_id_gaps": 3,
            },
        }
        stream_stats = {
            "CAM_A": {"StreamDroppedFrameCount": 5, "TransferQueueOverflowCount": 0},
            "CAM_B": {"StreamDroppedFrameCount": 0, "TransferQueueOverflowCount": 2},
        }

        data = _make_json_data(
            include_diagnostics=True,
            sync_wait_cycles=wait_cycles,
            sync_bottleneck_cameras=bottleneck,
            queue_depths=q_depths,
            frame_id_cross_delta=cross_delta,
            camera_error_summary=error_summary,
            camera_stream_stats=stream_stats,
        )
        json_path = _write_json(tmp_path, data)
        trial = load_trial(json_path, trial_index=0)

        assert trial.has_diagnostics is True
        assert trial.diagnostics_version == 1
        assert isinstance(trial.sync_wait_cycles, np.ndarray)
        assert trial.sync_wait_cycles.dtype == np.int64
        assert len(trial.sync_wait_cycles) == N_FRAMES
        assert trial.max_sync_wait_cycles == 2
        assert trial.mean_sync_wait_cycles == pytest.approx(3 / 20)
        assert trial.sync_bottleneck_cameras == bottleneck
        assert trial.queue_depths == q_depths
        assert trial.frame_id_cross_delta == cross_delta
        assert trial.camera_error_summary == error_summary
        assert trial.camera_stream_stats == stream_stats


class TestSyncBottleneckInsight:
    def test_bottleneck_camera_detected(self, tmp_path: Path) -> None:
        n = 20
        wait_cycles = [1] * 10 + [0] * 10
        bottleneck = [["CAM_C"]] * 8 + [["CAM_A"]] * 2 + [[]] * 10

        data = _make_json_data(
            n_frames=n,
            include_diagnostics=True,
            sync_wait_cycles=wait_cycles,
            sync_bottleneck_cameras=bottleneck,
        )
        _write_json(tmp_path, data)
        report = load_session(tmp_path)
        insights = diagnose_sync_issues(report)

        bottleneck_insights = [i for i in insights if "sync bottleneck" in i]
        assert len(bottleneck_insights) >= 1
        assert "CAM_C" in bottleneck_insights[0]
        assert "8/10" in bottleneck_insights[0]

    def test_uniform_distribution_suppressed(self, tmp_path: Path) -> None:
        n = 20
        wait_cycles = [1] * 10 + [0] * 10
        bottleneck = [["CAM_A", "CAM_B", "CAM_C"]] * 10 + [[]] * 10

        data = _make_json_data(
            n_frames=n,
            include_diagnostics=True,
            sync_wait_cycles=wait_cycles,
            sync_bottleneck_cameras=bottleneck,
        )
        _write_json(tmp_path, data)
        report = load_session(tmp_path)
        insights = diagnose_sync_issues(report)

        bottleneck_insights = [i for i in insights if "sync bottleneck" in i]
        assert len(bottleneck_insights) == 0


class TestFrameIdMisalignmentPersistence:
    def test_consecutive_misalignment_detected(self, tmp_path: Path) -> None:
        n = 20
        cross_delta = [{c: 0 for c in CAMERA_IDS} for _ in range(n)]
        for i in range(5, 12):
            cross_delta[i] = {"CAM_A": 0, "CAM_B": 1, "CAM_C": 0}

        data = _make_json_data(
            n_frames=n,
            include_diagnostics=True,
            frame_id_cross_delta=cross_delta,
        )
        _write_json(tmp_path, data)
        report = load_session(tmp_path)
        insights = diagnose_sync_issues(report)

        misalign_insights = [i for i in insights if "misalignment" in i.lower()]
        assert len(misalign_insights) >= 1
        assert "frame 5" in misalign_insights[0]
        assert "7 frames" in misalign_insights[0]


class TestPerCameraErrorInsight:
    def test_error_rate_reported(self, tmp_path: Path) -> None:
        error_summary = {
            "CAM_A": {
                "incomplete_frames": 5,
                "exceptions": 3,
                "total_acquired": 200,
                "frame_id_gaps": 2,
            },
            "CAM_B": {
                "incomplete_frames": 0,
                "exceptions": 0,
                "total_acquired": 200,
                "frame_id_gaps": 0,
            },
        }

        data = _make_json_data(
            include_diagnostics=True,
            camera_error_summary=error_summary,
        )
        _write_json(tmp_path, data)
        report = load_session(tmp_path)
        insights = diagnose_sync_issues(report)

        error_insights = [i for i in insights if "error rate" in i]
        assert len(error_insights) >= 1
        cam_a_insight = [i for i in error_insights if "CAM_A" in i]
        assert len(cam_a_insight) == 1
        assert "5 incomplete" in cam_a_insight[0]
        assert "3 exceptions" in cam_a_insight[0]
        cam_b_insights = [i for i in error_insights if "CAM_B" in i]
        assert len(cam_b_insights) == 0


class TestFigureReturnsNoneForLegacy:
    def test_sync_wait_figure_none(self, tmp_path: Path) -> None:
        data = _make_json_data(include_diagnostics=False)
        _write_json(tmp_path, data)
        report = load_session(tmp_path)

        assert create_sync_wait_figure(report) is None
        assert create_queue_depth_figure(report) is None
        assert create_camera_error_figure(report) is None


class TestMixedSession:
    def test_mixed_diagnostic_and_legacy(self, tmp_path: Path) -> None:
        legacy_data = _make_json_data(include_diagnostics=False)
        _write_json(tmp_path, legacy_data, filename="test_20250101_120000.json")

        diag_data = _make_json_data(
            include_diagnostics=True,
            sync_wait_cycles=[1] * 10 + [0] * 10,
            sync_bottleneck_cameras=[["CAM_A"]] * 8 + [["CAM_B"]] * 2 + [[]] * 10,
            camera_error_summary={
                "CAM_A": {
                    "incomplete_frames": 1,
                    "exceptions": 0,
                    "total_acquired": 20,
                    "frame_id_gaps": 0,
                },
            },
        )
        _write_json(tmp_path, diag_data, filename="test_20250101_120100.json")

        report = load_session(tmp_path)

        assert report.n_trials == 2
        assert report.trials[0].has_diagnostics is False
        assert report.trials[1].has_diagnostics is True

        insights = diagnose_sync_issues(report)
        bottleneck_insights = [i for i in insights if "sync bottleneck" in i]
        assert len(bottleneck_insights) >= 1
        assert "Trial 1" in bottleneck_insights[0]

        fig = create_sync_wait_figure(report)
        assert fig is not None


class TestCameraStreamStatsInsight:
    def test_dropped_frames_reported(self, tmp_path: Path) -> None:
        stream_stats = {
            "CAM_A": {"StreamDroppedFrameCount": 12, "TransferQueueOverflowCount": 0},
            "CAM_B": {"StreamDroppedFrameCount": 0, "TransferQueueOverflowCount": 3},
            "CAM_C": {"StreamDroppedFrameCount": 0, "TransferQueueOverflowCount": 0},
        }
        data = _make_json_data(
            include_diagnostics=True,
            camera_stream_stats=stream_stats,
        )
        _write_json(tmp_path, data)
        report = load_session(tmp_path)
        insights = diagnose_sync_issues(report)

        device_insights = [i for i in insights if "device/network" in i]
        assert len(device_insights) == 2
        cam_a = [i for i in device_insights if "CAM_A" in i]
        assert "dropped 12 frames" in cam_a[0]
        cam_b = [i for i in device_insights if "CAM_B" in i]
        assert "queue overflow 3" in cam_b[0]


class TestV2DiagnosticFieldParsing:
    def test_v2_fields_parsed(self, tmp_path: Path) -> None:
        ptp_start = {"CAM_A": 100, "CAM_B": -50, "CAM_C": 200}
        ptp_end = {"CAM_A": 150, "CAM_B": -30, "CAM_C": 220}
        temp_start = {"CAM_A": 45.0, "CAM_B": 42.0, "CAM_C": 44.5}
        temp_end = {"CAM_A": 50.0, "CAM_B": 43.0, "CAM_C": 45.0}
        snapshots = [
            {"wall_clock": 1000.0, "nic": {"rx_dropped": 0, "rx_packets": 500}},
            {"wall_clock": 1010.0, "nic": {"rx_dropped": 2, "rx_packets": 1000}},
        ]
        alerts = {"count": 3, "first_frame": 5, "last_frame": 15, "max_spread_ms": 7.5}
        timeout_events = [
            {
                "frame_idx": 10,
                "elapsed_s": 5.5,
                "empty_cameras": ["CAM_B"],
                "queue_depths": {"CAM_A": 3, "CAM_B": 0},
            },
        ]

        data = _make_json_data(
            include_diagnostics=True,
            diagnostics_version=2,
            system_snapshots=snapshots,
            ptp_offset_start=ptp_start,
            ptp_offset_end=ptp_end,
            camera_temperature_start=temp_start,
            camera_temperature_end=temp_end,
            timespread_alerts=alerts,
            sync_timeout_events=timeout_events,
        )
        json_path = _write_json(tmp_path, data)
        trial = load_trial(json_path, trial_index=0)

        assert trial.diagnostics_version == 2
        assert trial.ptp_offset_start == ptp_start
        assert trial.ptp_offset_end == ptp_end
        assert trial.camera_temperature_start == temp_start
        assert trial.camera_temperature_end == temp_end
        assert trial.system_snapshots == snapshots
        assert trial.timespread_alerts == alerts
        assert trial.sync_timeout_events == timeout_events

    def test_v2_fields_none_for_legacy(self, tmp_path: Path) -> None:
        data = _make_json_data(include_diagnostics=False)
        json_path = _write_json(tmp_path, data)
        trial = load_trial(json_path, trial_index=0)

        assert trial.system_snapshots is None
        assert trial.ptp_offset_start is None
        assert trial.ptp_offset_end is None
        assert trial.camera_temperature_start is None
        assert trial.camera_temperature_end is None
        assert trial.timespread_alerts is None
        assert trial.sync_timeout_events is None


class TestPtpOffsetDriftInsight:
    def test_significant_drift_detected(self, tmp_path: Path) -> None:
        data = _make_json_data(
            include_diagnostics=True,
            diagnostics_version=2,
            ptp_offset_start={"CAM_A": 0, "CAM_B": 100},
            ptp_offset_end={"CAM_A": 200_000, "CAM_B": 100},
        )
        _write_json(tmp_path, data)
        report = load_session(tmp_path)
        insights = diagnose_sync_issues(report)

        ptp_insights = [i for i in insights if "PTP offset drifted" in i]
        assert len(ptp_insights) == 1
        assert "CAM_A" in ptp_insights[0]
        assert "200" in ptp_insights[0]

    def test_small_drift_not_flagged(self, tmp_path: Path) -> None:
        data = _make_json_data(
            include_diagnostics=True,
            diagnostics_version=2,
            ptp_offset_start={"CAM_A": 0, "CAM_B": 100},
            ptp_offset_end={"CAM_A": 50, "CAM_B": 110},
        )
        _write_json(tmp_path, data)
        report = load_session(tmp_path)
        insights = diagnose_sync_issues(report)

        ptp_insights = [i for i in insights if "PTP offset drifted" in i]
        assert len(ptp_insights) == 0


class TestCameraTemperatureInsight:
    def test_temperature_rise_detected(self, tmp_path: Path) -> None:
        data = _make_json_data(
            include_diagnostics=True,
            diagnostics_version=2,
            camera_temperature_start={"CAM_A": 40.0, "CAM_B": 42.0},
            camera_temperature_end={"CAM_A": 46.0, "CAM_B": 42.5},
        )
        _write_json(tmp_path, data)
        report = load_session(tmp_path)
        insights = diagnose_sync_issues(report)

        temp_insights = [i for i in insights if "temperature rose" in i]
        assert len(temp_insights) == 1
        assert "CAM_A" in temp_insights[0]
        assert "6.0" in temp_insights[0]


class TestSyncTimeoutInsight:
    def test_timeout_events_reported(self, tmp_path: Path) -> None:
        timeout_events = [
            {"frame_idx": 10, "elapsed_s": 5.5, "empty_cameras": ["CAM_B", "CAM_C"]},
            {"frame_idx": 50, "elapsed_s": 6.0, "empty_cameras": ["CAM_B"]},
        ]

        data = _make_json_data(
            include_diagnostics=True,
            diagnostics_version=2,
            sync_timeout_events=timeout_events,
        )
        _write_json(tmp_path, data)
        report = load_session(tmp_path)
        insights = diagnose_sync_issues(report)

        timeout_insights = [i for i in insights if "sync timeout" in i]
        assert len(timeout_insights) == 1
        assert "2 sync timeout" in timeout_insights[0]
        assert "CAM_B" in timeout_insights[0]
        assert "CAM_C" in timeout_insights[0]


class TestTimespreadAlertInsight:
    def test_alerts_reported(self, tmp_path: Path) -> None:
        alerts = {
            "count": 5,
            "first_frame": 10,
            "last_frame": 45,
            "max_spread_ms": 8.123,
        }

        data = _make_json_data(
            include_diagnostics=True,
            diagnostics_version=2,
            timespread_alerts=alerts,
        )
        _write_json(tmp_path, data)
        report = load_session(tmp_path)
        insights = diagnose_sync_issues(report)

        alert_insights = [i for i in insights if "timespread alert" in i]
        assert len(alert_insights) == 1
        assert "5 timespread alert" in alert_insights[0]
        assert "8.123" in alert_insights[0]
        assert "frames 10-45" in alert_insights[0]

    def test_zero_alerts_not_reported(self, tmp_path: Path) -> None:
        alerts = {"count": 0, "first_frame": -1, "last_frame": -1, "max_spread_ms": 0.0}

        data = _make_json_data(
            include_diagnostics=True,
            diagnostics_version=2,
            timespread_alerts=alerts,
        )
        _write_json(tmp_path, data)
        report = load_session(tmp_path)
        insights = diagnose_sync_issues(report)

        alert_insights = [i for i in insights if "timespread alert" in i]
        assert len(alert_insights) == 0


class TestNicRxDroppedInsight:
    def test_rx_dropped_increase_flagged(self, tmp_path: Path) -> None:
        snapshots = [
            {"wall_clock": 1000.0, "nic": {"rx_dropped": 5, "rx_packets": 500}},
            {"wall_clock": 1010.0, "nic": {"rx_dropped": 5, "rx_packets": 1000}},
            {"wall_clock": 1020.0, "nic": {"rx_dropped": 12, "rx_packets": 1500}},
        ]

        data = _make_json_data(
            include_diagnostics=True,
            diagnostics_version=2,
            system_snapshots=snapshots,
        )
        _write_json(tmp_path, data)
        report = load_session(tmp_path)
        insights = diagnose_sync_issues(report)

        nic_insights = [i for i in insights if "rx_dropped" in i]
        assert len(nic_insights) == 1
        assert "7" in nic_insights[0]

    def test_no_increase_not_flagged(self, tmp_path: Path) -> None:
        snapshots = [
            {"wall_clock": 1000.0, "nic": {"rx_dropped": 5}},
            {"wall_clock": 1010.0, "nic": {"rx_dropped": 5}},
        ]

        data = _make_json_data(
            include_diagnostics=True,
            diagnostics_version=2,
            system_snapshots=snapshots,
        )
        _write_json(tmp_path, data)
        report = load_session(tmp_path)
        insights = diagnose_sync_issues(report)

        nic_insights = [i for i in insights if "rx_dropped" in i]
        assert len(nic_insights) == 0


class TestJsonCompaction:
    def test_compacted_json_loads_correctly(self, tmp_path: Path) -> None:
        """JSON with placeholder fields stripped should still parse correctly."""
        data = _make_json_data(n_frames=10, include_diagnostics=True)
        # Remove fields that _compact_json_data would strip (all-zero/empty defaults)
        del data["frame_id_cross_delta"]
        del data["sync_wait_cycles"]
        del data["sync_bottleneck_cameras"]

        _write_json(tmp_path, data)
        report = load_session(tmp_path)

        assert report.trials[0].max_timestamp_spread_ms > 0
        assert report.trials[0].sync_wait_cycles is None
        assert report.trials[0].sync_bottleneck_cameras is None
        assert report.trials[0].frame_id_cross_delta is None
        assert report.trials[0].has_diagnostics is True


class TestFrameSkipEventsParsing:
    def test_skip_events_parsed(self, tmp_path: Path) -> None:
        events = [
            {
                "frame_idx": 42,
                "camera_serial": "CAM_B",
                "expected_frame_id": 1042,
                "actual_frame_id": 1043,
                "gap_size": 1,
                "recovered": True,
            },
        ]
        data = _make_json_data(
            include_diagnostics=True,
            diagnostics_version=2,
            frame_skip_events=events,
        )
        json_path = _write_json(tmp_path, data)
        trial = load_trial(json_path, trial_index=0)

        assert trial.frame_skip_events is not None
        assert len(trial.frame_skip_events) == 1
        assert trial.frame_skip_events[0]["camera_serial"] == "CAM_B"
        assert trial.frame_skip_events[0]["gap_size"] == 1
        assert trial.frame_skip_events[0]["recovered"] is True

    def test_no_skip_events_returns_none(self, tmp_path: Path) -> None:
        data = _make_json_data(include_diagnostics=True, diagnostics_version=2)
        json_path = _write_json(tmp_path, data)
        trial = load_trial(json_path, trial_index=0)

        assert trial.frame_skip_events is None

    def test_empty_skip_events_returns_none(self, tmp_path: Path) -> None:
        data = _make_json_data(
            include_diagnostics=True,
            diagnostics_version=2,
            frame_skip_events=[],
        )
        json_path = _write_json(tmp_path, data)
        trial = load_trial(json_path, trial_index=0)

        assert trial.frame_skip_events is None


class TestFrameSkipInsight:
    def test_recovered_skip_insight(self, tmp_path: Path) -> None:
        events = [
            {
                "frame_idx": 100,
                "camera_serial": "CAM_A",
                "expected_frame_id": 1100,
                "actual_frame_id": 1101,
                "gap_size": 1,
                "recovered": True,
            },
        ]
        data = _make_json_data(
            include_diagnostics=True,
            diagnostics_version=2,
            frame_skip_events=events,
        )
        _write_json(tmp_path, data)
        report = load_session(tmp_path)
        insights = diagnose_sync_issues(report)

        skip_insights = [i for i in insights if "skipped" in i]
        assert len(skip_insights) == 1
        assert "CAM_A" in skip_insights[0]
        assert "placeholder inserted" in skip_insights[0]
        assert "frame 100" in skip_insights[0]

    def test_unrecovered_skip_insight(self, tmp_path: Path) -> None:
        events = [
            {
                "frame_idx": 50,
                "camera_serial": "CAM_C",
                "expected_frame_id": 1050,
                "actual_frame_id": 1052,
                "gap_size": 2,
                "recovered": False,
            },
        ]
        data = _make_json_data(
            include_diagnostics=True,
            diagnostics_version=2,
            frame_skip_events=events,
        )
        _write_json(tmp_path, data)
        report = load_session(tmp_path)
        insights = diagnose_sync_issues(report)

        skip_insights = [i for i in insights if "skipped" in i]
        assert len(skip_insights) == 1
        assert "CAM_C" in skip_insights[0]
        assert "NOT recovered" in skip_insights[0]
        assert "2 frames" in skip_insights[0]

    def test_no_skip_events_no_insight(self, tmp_path: Path) -> None:
        data = _make_json_data(include_diagnostics=True, diagnostics_version=2)
        _write_json(tmp_path, data)
        report = load_session(tmp_path)
        insights = diagnose_sync_issues(report)

        skip_insights = [i for i in insights if "skipped" in i]
        assert len(skip_insights) == 0


class TestMergedDriftFrequencyInsight:
    """Verify that checks 2 and 5 are merged: a single pass produces both
    the repeat-offender insight and the per-camera drift frequency table."""

    def _make_drift_data(self, tmp_path: Path) -> None:
        """Create two trials where CAM_B drifts in both, CAM_C in one."""
        for trial_idx, (filename, drift_cam_c) in enumerate(
            [
                ("test_20250101_120000.json", True),
                ("test_20250101_120100.json", False),
            ]
        ):
            n = N_FRAMES
            timestamps = []
            frame_ids = []
            for i in range(n):
                row_ts = [
                    BASE_TIMESTAMP_NS + i * FRAME_PERIOD_NS + c * 100 for c in range(3)
                ]
                fids = [1000 + i, 1000 + i, 1000 + i]
                if i == 5:
                    fids[1] += 1  # CAM_B drifts
                    if drift_cam_c:
                        fids[2] += 1  # CAM_C drifts in first trial only
                timestamps.append(row_ts)
                frame_ids.append(fids)

            data = {
                "serials": CAMERA_IDS,
                "timestamps": timestamps,
                "frame_id": frame_ids,
                "real_times": [[0.0] * 3] * n,
                "exposure_times": [15000] * 3,
                "frame_rates_requested": [30] * 3,
                "frame_rates_binning": [30] * 3,
                "camera_config_hash": "abc",
                "camera_info": {},
                "meta_info": {},
                "system_info": {},
            }
            _write_json(tmp_path, data, filename=filename)

    def test_drift_frequency_table_emitted(self, tmp_path: Path) -> None:
        self._make_drift_data(tmp_path)
        report = load_session(tmp_path)
        insights = diagnose_sync_issues(report)

        freq_insights = [i for i in insights if "Per-camera drift frequency" in i]
        assert len(freq_insights) == 1
        assert "CAM_B" in freq_insights[0]

    def test_repeat_offender_emitted(self, tmp_path: Path) -> None:
        self._make_drift_data(tmp_path)
        report = load_session(tmp_path)
        insights = diagnose_sync_issues(report)

        offender_insights = [i for i in insights if "Repeat offender" in i]
        assert len(offender_insights) == 1
        assert "CAM_B" in offender_insights[0]

    def test_no_duplicate_drift_computation(self, tmp_path: Path) -> None:
        """Frequency table and offender insight should not duplicate counts."""
        self._make_drift_data(tmp_path)
        report = load_session(tmp_path)
        insights = diagnose_sync_issues(report)

        freq_insights = [i for i in insights if "Per-camera drift frequency" in i]
        assert len(freq_insights) == 1
        assert "CAM_B: 2 / 2" in freq_insights[0]
        assert "CAM_C: 1 / 2" in freq_insights[0]


class TestModuleConstants:
    """Verify that module-level diagnostic constants are importable and have expected values."""

    def test_constants_importable(self) -> None:
        from multi_camera.acquisition.diagnostics.json_parser import (
            BOTTLENECK_DISPROPORTION_RATIO,
            BOTTLENECK_MIN_FRACTION,
            BURST_MIN_CONSECUTIVE,
            FRAME_PERIOD_TOLERANCE,
            PTP_DRIFT_THRESHOLD_US,
            QUEUE_DEPTH_SPIKE_THRESHOLD,
            REPEAT_OFFENDER_FRACTION,
            TEMPERATURE_RISE_THRESHOLD_C,
        )

        assert FRAME_PERIOD_TOLERANCE == 0.15
        assert REPEAT_OFFENDER_FRACTION == 0.3
        assert BURST_MIN_CONSECUTIVE == 3
        assert BOTTLENECK_DISPROPORTION_RATIO == 1.5
        assert BOTTLENECK_MIN_FRACTION == 0.3
        assert QUEUE_DEPTH_SPIKE_THRESHOLD == 10
        assert PTP_DRIFT_THRESHOLD_US == 100
        assert TEMPERATURE_RISE_THRESHOLD_C == 5.0
