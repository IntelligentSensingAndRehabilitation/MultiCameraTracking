"""Diagnostics for multi-camera sync quality from FLIR GigE acquisition JSON metadata.

Analyzes timestamp synchronization, frame ID alignment, and serial data integrity
across cameras from JSON files written by flir_recording_api.write_metadata_queue().
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray


@dataclass(frozen=True)
class TrialSyncMetrics:
    """Per-JSON-file synchronization results."""

    file_path: Path
    trial_index: int
    recording_timestamp: str
    camera_ids: list[str]
    n_frames: int
    timestamp_delta_ms: NDArray[np.float64]
    frame_id_delta: NDArray[np.int64]
    serial_data_delta: NDArray[np.int64] | None
    frame_period_ms: float
    sync_wait_cycles: NDArray[np.int64] | None = None
    sync_bottleneck_cameras: list[list[str]] | None = None
    queue_depths: list[dict[str, int]] | None = None
    frame_id_cross_delta: list[dict[str, int]] | None = None
    camera_error_summary: dict[str, dict[str, int]] | None = None
    camera_stream_stats: dict[str, dict[str, int]] | None = None
    diagnostics_version: int | None = None
    system_snapshots: list[dict] | None = None
    ptp_offset_start: dict[str, int] | None = None
    ptp_offset_end: dict[str, int] | None = None
    camera_temperature_start: dict[str, float] | None = None
    camera_temperature_end: dict[str, float] | None = None
    timespread_alerts: dict | None = None
    sync_timeout_events: list[dict] | None = None

    @property
    def reference_camera(self) -> str:
        return self.camera_ids[0]

    @property
    def delta_camera_ids(self) -> list[str]:
        return self.camera_ids[1:]

    @property
    def max_timestamp_spread_ms(self) -> float:
        return float(np.max(np.abs(self.timestamp_delta_ms)))

    @property
    def mean_timestamp_spread_ms(self) -> float:
        return float(np.mean(np.abs(self.timestamp_delta_ms)))

    @property
    def has_frame_id_drift(self) -> bool:
        return bool(np.any(self.frame_id_delta != 0))

    @property
    def has_diagnostics(self) -> bool:
        return self.diagnostics_version is not None

    @property
    def max_sync_wait_cycles(self) -> int:
        if self.sync_wait_cycles is None:
            return 0
        return int(np.max(self.sync_wait_cycles))

    @property
    def mean_sync_wait_cycles(self) -> float:
        if self.sync_wait_cycles is None:
            return 0.0
        return float(np.mean(self.sync_wait_cycles))


@dataclass(frozen=True)
class SessionSyncReport:
    """Aggregate synchronization report across all trials in a session."""

    data_dir: Path
    trials: list[TrialSyncMetrics]
    camera_ids: list[str]
    camera_set_consistent: bool

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def total_frames(self) -> int:
        return sum(t.n_frames for t in self.trials)

    @property
    def worst_timestamp_spread_ms(self) -> float:
        if not self.trials:
            return 0.0
        return max(t.max_timestamp_spread_ms for t in self.trials)


# --- Loading ---


def extract_recording_timestamp(json_path: Path) -> str:
    """Extract YYYYMMDD_HHMMSS from filename following {root}_{YYYYMMDD}_{HHMMSS}.json convention."""
    parts = json_path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Filename does not match expected pattern {{root}}_{{YYYYMMDD}}_{{HHMMSS}}: {json_path.name}"
        )
    candidate = f"{parts[-2]}_{parts[-1]}"
    datetime.strptime(candidate, "%Y%m%d_%H%M%S")
    return candidate


def load_trial(json_path: Path, trial_index: int) -> TrialSyncMetrics:
    """Load a single JSON metadata file and compute sync deltas."""
    with open(json_path) as f:
        data = json.load(f)

    camera_ids: list[str] = data["serials"]
    ts = np.array(data["timestamps"], dtype=np.float64)
    frame_id = np.array(data["frame_id"], dtype=np.int64)

    timestamp_delta_ms = (ts[:, 1:] - ts[:, :1]) / 1e6

    frame_id_delta = frame_id[:, 1:] - frame_id[:, :1]

    serial_data_delta: NDArray[np.int64] | None = None
    if "chunk_serial_data" in data and data["chunk_serial_data"]:
        raw = np.array(data["chunk_serial_data"], dtype=np.int64)
        if raw.size > 0:
            serial_data_delta = raw[:, 1:] - raw[:, :1]

    recording_timestamp = extract_recording_timestamp(json_path)

    if ts.shape[0] > 1:
        inter_frame_ms = np.diff(ts[:, 0]) / 1e6
        frame_period_ms = float(np.median(inter_frame_ms))
    else:
        frame_period_ms = 0.0

    sync_wait_cycles: NDArray[np.int64] | None = None
    if "sync_wait_cycles" in data and data["sync_wait_cycles"]:
        sync_wait_cycles = np.array(data["sync_wait_cycles"], dtype=np.int64)

    sync_bottleneck_cameras: list[list[str]] | None = None
    if "sync_bottleneck_cameras" in data and data["sync_bottleneck_cameras"]:
        sync_bottleneck_cameras = data["sync_bottleneck_cameras"]

    queue_depths: list[dict[str, int]] | None = None
    if "queue_depths" in data and data["queue_depths"]:
        queue_depths = data["queue_depths"]

    frame_id_cross_delta: list[dict[str, int]] | None = None
    if "frame_id_cross_delta" in data and data["frame_id_cross_delta"]:
        frame_id_cross_delta = data["frame_id_cross_delta"]

    camera_error_summary: dict[str, dict[str, int]] | None = data.get(
        "camera_error_summary"
    )
    camera_stream_stats: dict[str, dict[str, int]] | None = data.get(
        "camera_stream_stats"
    )
    diagnostics_version: int | None = data.get("diagnostics_version")

    system_snapshots: list[dict] | None = data.get("system_snapshots")
    ptp_offset_start: dict[str, int] | None = data.get("ptp_offset_start")
    ptp_offset_end: dict[str, int] | None = data.get("ptp_offset_end")
    camera_temperature_start: dict[str, float] | None = data.get(
        "camera_temperature_start"
    )
    camera_temperature_end: dict[str, float] | None = data.get("camera_temperature_end")
    timespread_alerts: dict | None = data.get("timespread_alerts")
    sync_timeout_events: list[dict] | None = data.get("sync_timeout_events")

    return TrialSyncMetrics(
        file_path=json_path,
        trial_index=trial_index,
        recording_timestamp=recording_timestamp,
        camera_ids=camera_ids,
        n_frames=ts.shape[0],
        timestamp_delta_ms=timestamp_delta_ms,
        frame_id_delta=frame_id_delta,
        serial_data_delta=serial_data_delta,
        frame_period_ms=frame_period_ms,
        sync_wait_cycles=sync_wait_cycles,
        sync_bottleneck_cameras=sync_bottleneck_cameras,
        queue_depths=queue_depths,
        frame_id_cross_delta=frame_id_cross_delta,
        camera_error_summary=camera_error_summary,
        camera_stream_stats=camera_stream_stats,
        diagnostics_version=diagnostics_version,
        system_snapshots=system_snapshots,
        ptp_offset_start=ptp_offset_start,
        ptp_offset_end=ptp_offset_end,
        camera_temperature_start=camera_temperature_start,
        camera_temperature_end=camera_temperature_end,
        timespread_alerts=timespread_alerts,
        sync_timeout_events=sync_timeout_events,
    )


def _sort_key_for_json(json_path: Path) -> tuple[str, float]:
    """Return (timestamp_str, mtime) so we sort by timestamp first, mtime as fallback."""
    try:
        ts = extract_recording_timestamp(json_path)
        return (ts, 0.0)
    except ValueError:
        print(
            f"  Warning: cannot parse timestamp from {json_path.name}, falling back to mtime",
            file=sys.stderr,
        )
        return ("", json_path.stat().st_mtime)


def load_session(data_dir: Path) -> SessionSyncReport:
    """Load all JSON metadata files in a directory and build a session report."""
    json_files = sorted(data_dir.glob("*.json"), key=_sort_key_for_json)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    trials: list[TrialSyncMetrics] = []
    for i, json_path in enumerate(json_files):
        trials.append(load_trial(json_path, trial_index=i))

    first_camera_set = trials[0].camera_ids
    camera_set_consistent = all(t.camera_ids == first_camera_set for t in trials)

    return SessionSyncReport(
        data_dir=data_dir,
        trials=trials,
        camera_ids=first_camera_set,
        camera_set_consistent=camera_set_consistent,
    )


# --- Terminal Report ---


def print_report(report: SessionSyncReport) -> None:
    """Print structured sync diagnostics to stdout."""
    width = 80
    print(f"\nSession Sync Diagnostics: {report.data_dir}")
    print("=" * width)
    print()
    print(f"  Trials:    {report.n_trials} files, sorted by recording timestamp")
    print(f"  Cameras:   {len(report.camera_ids)} (reference: {report.camera_ids[0]})")
    print(f"  Frames:    {report.total_frames:,} total")
    consistent_str = "YES" if report.camera_set_consistent else "NO"
    print(f"  Camera set consistent: {consistent_str}")

    if not report.camera_set_consistent:
        seen: set[tuple[str, ...]] = set()
        for t in report.trials:
            key = tuple(t.camera_ids)
            if key not in seen:
                seen.add(key)
                print(f"    Trial {t.trial_index}: {t.camera_ids}")

    print()
    print("Per-Trial Summary")
    print("-" * width)
    print(
        f"  {'Trial':>5}  {'Timestamp':<17}  {'Frames':>6}  {'Max dT (ms)':>11}  {'Mean |dT| (ms)':>14}  Frame ID Drift"
    )

    for t in report.trials:
        drift_cameras = []
        if t.has_frame_id_drift:
            for col_idx, cam_id in enumerate(t.delta_camera_ids):
                col_deltas = t.frame_id_delta[:, col_idx]
                unique_deltas = np.unique(col_deltas[col_deltas != 0])
                if len(unique_deltas) > 0:
                    drift_str = ",".join(f"{d:+d}" for d in unique_deltas)
                    drift_cameras.append(f"cam {cam_id}: {drift_str}")
        drift_text = "; ".join(drift_cameras) if drift_cameras else "none"

        print(
            f"  {t.trial_index:>5}  {t.recording_timestamp:<17}  {t.n_frames:>6}"
            f"  {t.max_timestamp_spread_ms:>11.3f}  {t.mean_timestamp_spread_ms:>14.3f}  {drift_text}"
        )

    all_ts_deltas = np.concatenate(
        [t.timestamp_delta_ms.ravel() for t in report.trials]
    )
    abs_deltas = np.abs(all_ts_deltas)
    drift_trials = sum(1 for t in report.trials if t.has_frame_id_drift)
    serial_trials = sum(
        1
        for t in report.trials
        if t.serial_data_delta is not None and np.any(t.serial_data_delta != 0)
    )
    has_serial = any(t.serial_data_delta is not None for t in report.trials)

    print()
    print("Aggregate Statistics")
    print("-" * width)
    print(
        f"  Timestamp delta (ms):  max={np.max(abs_deltas):.3f}  mean={np.mean(abs_deltas):.3f}  std={np.std(abs_deltas):.3f}"
    )
    print(
        f"  Frame ID drift:        {drift_trials} / {report.n_trials} trials affected"
    )
    if has_serial:
        print(
            f"  Serial data drift:     {serial_trials} / {report.n_trials} trials affected"
        )

    frame_period = _estimate_session_frame_period_ms(report)
    if frame_period > 0:
        print(
            f"  Est. frame period:     {frame_period:.2f} ms ({1000.0 / frame_period:.1f} fps)"
        )

    _print_acquisition_diagnostics(report, width)

    insights = diagnose_sync_issues(report)
    if insights:
        print()
        print("Diagnostic Insights")
        print("-" * width)
        for insight in insights:
            for line in insight.split("\n"):
                print(f"  {line}")
    print()


def _print_acquisition_diagnostics(report: SessionSyncReport, width: int) -> None:
    """Print acquisition diagnostics section if any trials have diagnostic data."""
    diag_trials = [t for t in report.trials if t.has_diagnostics]
    if not diag_trials:
        return

    print()
    print("Acquisition Diagnostics")
    print("-" * width)

    for t in diag_trials:
        waited_frames = (
            int(np.sum(t.sync_wait_cycles > 0)) if t.sync_wait_cycles is not None else 0
        )
        print(
            f"  Trial {t.trial_index}: waited frames={waited_frames}, "
            f"max wait cycles={t.max_sync_wait_cycles}, "
            f"mean wait cycles={t.mean_sync_wait_cycles:.1f}"
        )

    error_summaries = [
        t.camera_error_summary for t in diag_trials if t.camera_error_summary
    ]
    if error_summaries:
        print()
        print("  Per-Camera Error Summary:")
        print(
            f"    {'Camera':<20} {'Total':>8} {'Incomplete':>12} {'Exceptions':>12} {'Gaps':>8}"
        )
        all_cameras: set[str] = set()
        for es in error_summaries:
            all_cameras.update(es.keys())
        for cam in sorted(all_cameras):
            totals = {
                "total_acquired": 0,
                "incomplete_frames": 0,
                "exceptions": 0,
                "frame_id_gaps": 0,
            }
            for es in error_summaries:
                if cam in es:
                    for k in totals:
                        totals[k] += es[cam].get(k, 0)
            print(
                f"    {cam:<20} {totals['total_acquired']:>8} "
                f"{totals['incomplete_frames']:>12} {totals['exceptions']:>12} {totals['frame_id_gaps']:>8}"
            )

    stream_stats = [t.camera_stream_stats for t in diag_trials if t.camera_stream_stats]
    if stream_stats:
        print()
        print("  Camera Stream Stats:")
        all_cameras_ss: set[str] = set()
        for ss in stream_stats:
            all_cameras_ss.update(ss.keys())
        for cam in sorted(all_cameras_ss):
            parts = []
            for ss in stream_stats:
                if cam in ss:
                    for attr, val in ss[cam].items():
                        if val > 0:
                            parts.append(f"{attr}={val}")
            if parts:
                print(f"    {cam}: {', '.join(parts)}")

    v2_trials = [t for t in diag_trials if (t.diagnostics_version or 0) >= 2]
    if v2_trials:
        print()
        print("  System Monitor & Camera Health (v2):")
        for t in v2_trials:
            if t.ptp_offset_start:
                print(
                    f"    Trial {t.trial_index} PTP offsets (start): {t.ptp_offset_start}"
                )
            if t.ptp_offset_end:
                print(
                    f"    Trial {t.trial_index} PTP offsets (end):   {t.ptp_offset_end}"
                )
            if t.camera_temperature_start:
                temps = {c: f"{v:.1f}°C" for c, v in t.camera_temperature_start.items()}
                print(f"    Trial {t.trial_index} temperatures (start): {temps}")
            if t.camera_temperature_end:
                temps = {c: f"{v:.1f}°C" for c, v in t.camera_temperature_end.items()}
                print(f"    Trial {t.trial_index} temperatures (end):   {temps}")
            if t.system_snapshots:
                print(
                    f"    Trial {t.trial_index}: {len(t.system_snapshots)} system snapshots recorded"
                )
            if t.sync_timeout_events:
                print(
                    f"    Trial {t.trial_index}: {len(t.sync_timeout_events)} sync timeout event(s)"
                )
            if t.timespread_alerts and t.timespread_alerts.get("count", 0) > 0:
                a = t.timespread_alerts
                print(
                    f"    Trial {t.trial_index}: {a['count']} timespread alert(s), "
                    f"max {a['max_spread_ms']:.3f} ms"
                )


# --- Diagnostics ---


def _estimate_session_frame_period_ms(report: SessionSyncReport) -> float:
    """Median frame period across all trials from reference camera timestamps."""
    periods = [t.frame_period_ms for t in report.trials if t.frame_period_ms > 0]
    if not periods:
        return 0.0
    return float(np.median(periods))


def diagnose_sync_issues(report: SessionSyncReport) -> list[str]:
    """Detect known sync failure patterns and return human-readable insight strings."""
    insights: list[str] = []
    frame_period = _estimate_session_frame_period_ms(report)
    if frame_period <= 0:
        return insights

    fps_estimate = 1000.0 / frame_period
    tolerance = 0.15

    # 1. Frame-period spikes: max dT ≈ frame period → alignment error, not clock drift
    frame_period_trials: list[TrialSyncMetrics] = []
    for t in report.trials:
        max_dt = t.max_timestamp_spread_ms
        if abs(max_dt - frame_period) / frame_period < tolerance:
            frame_period_trials.append(t)

    if frame_period_trials:
        indices = [str(t.trial_index) for t in frame_period_trials]
        insights.append(
            f"{len(frame_period_trials)} / {report.n_trials} trials have max dT ≈ {frame_period:.1f} ms "
            f"(1 frame @ {fps_estimate:.1f} fps) → frame alignment error, not clock drift. "
            f"Trials: {', '.join(indices)}"
        )

    # 2. Repeat offender cameras: camera appearing in >30% of affected drift trials
    drift_trials = [t for t in report.trials if t.has_frame_id_drift]
    if drift_trials:
        camera_drift_counts: dict[str, int] = {}
        for t in drift_trials:
            for col_idx, cam_id in enumerate(t.delta_camera_ids):
                if np.any(t.frame_id_delta[:, col_idx] != 0):
                    camera_drift_counts[cam_id] = camera_drift_counts.get(cam_id, 0) + 1

        threshold = max(1, int(len(drift_trials) * 0.3))
        offenders = {cam: n for cam, n in camera_drift_counts.items() if n >= threshold}
        if offenders:
            parts = [
                f"cam {cam}: {n}/{len(drift_trials)} trials"
                for cam, n in sorted(offenders.items(), key=lambda x: -x[1])
            ]
            insights.append(f"Repeat offender cameras: {'; '.join(parts)}")

    # 3. Burst patterns: 3+ consecutive affected trials → sustained issue
    if drift_trials:
        drift_indices = {t.trial_index for t in drift_trials}
        runs: list[list[int]] = []
        current_run: list[int] = []
        for i in range(report.n_trials):
            if i in drift_indices:
                current_run.append(i)
            else:
                if len(current_run) >= 3:
                    runs.append(current_run)
                current_run = []
        if len(current_run) >= 3:
            runs.append(current_run)
        for run in runs:
            insights.append(
                f"Burst: trials {run[0]}-{run[-1]} ({len(run)} consecutive) had frame ID drift → sustained issue during that window"
            )

    # 4. Reference camera jumps: ALL delta cameras show same ±1 drift → reference skipped
    for t in frame_period_trials:
        if not t.has_frame_id_drift:
            continue
        per_frame_deltas = t.frame_id_delta
        for row_idx in range(per_frame_deltas.shape[0]):
            row = per_frame_deltas[row_idx, :]
            nonzero = row[row != 0]
            if len(nonzero) == per_frame_deltas.shape[1] and len(set(nonzero)) == 1:
                insights.append(
                    f"Trial {t.trial_index}: all delta cameras show {nonzero[0]:+d} at frame {row_idx} "
                    f"→ reference camera ({t.reference_camera}) skipped, not the others"
                )
                break

    # 5. Per-camera drift frequency table
    if drift_trials:
        all_camera_counts: dict[str, int] = {}
        for t in drift_trials:
            for col_idx, cam_id in enumerate(t.delta_camera_ids):
                if np.any(t.frame_id_delta[:, col_idx] != 0):
                    all_camera_counts[cam_id] = all_camera_counts.get(cam_id, 0) + 1
        if all_camera_counts:
            header = "Per-camera drift frequency:"
            lines = [header]
            for cam, n in sorted(all_camera_counts.items(), key=lambda x: -x[1]):
                lines.append(f"    cam {cam}: {n} / {len(drift_trials)} drift trials")
            insights.append("\n".join(lines))

    # 6. Sync bottleneck camera: only report cameras that are disproportionately slow
    for t in report.trials:
        if (
            not t.has_diagnostics
            or t.sync_bottleneck_cameras is None
            or t.sync_wait_cycles is None
        ):
            continue
        waited_indices = [i for i, w in enumerate(t.sync_wait_cycles) if w > 0]
        if not waited_indices:
            continue
        bottleneck_counts: dict[str, int] = {}
        for i in waited_indices:
            if i < len(t.sync_bottleneck_cameras):
                for cam in t.sync_bottleneck_cameras[i]:
                    bottleneck_counts[cam] = bottleneck_counts.get(cam, 0) + 1
        if not bottleneck_counts:
            continue
        n_waited = len(waited_indices)
        counts = list(bottleneck_counts.values())
        min_count = min(counts)
        max_count = max(counts)
        if min_count > 0 and max_count / min_count < 1.5:
            continue
        mean_count = sum(counts) / len(counts)
        for cam, count in sorted(bottleneck_counts.items(), key=lambda x: -x[1]):
            if count > mean_count * 1.5 and count > n_waited * 0.3:
                insights.append(
                    f"Trial {t.trial_index}: Camera {cam} was sync bottleneck on {count}/{n_waited} waited frames"
                )

    # 7. Queue depth spikes: any camera queue exceeding depth 10
    for t in report.trials:
        if not t.has_diagnostics or t.queue_depths is None:
            continue
        max_depths: dict[str, int] = {}
        for qd in t.queue_depths:
            for cam, depth in qd.items():
                if depth > max_depths.get(cam, 0):
                    max_depths[cam] = depth
        spiked = {cam: d for cam, d in max_depths.items() if d > 10}
        if spiked:
            parts = [
                f"{cam}={d}" for cam, d in sorted(spiked.items(), key=lambda x: -x[1])
            ]
            insights.append(
                f"Trial {t.trial_index}: Queue depth spikes ({', '.join(parts)}) "
                f"→ metadata thread falling behind acquisition"
            )

    # 8. Frame-ID misalignment persistence: consecutive nonzero cross-delta
    for t in report.trials:
        if not t.has_diagnostics or t.frame_id_cross_delta is None:
            continue
        nonzero_frames = [
            i
            for i, fcd in enumerate(t.frame_id_cross_delta)
            if any(v != 0 for v in fcd.values())
        ]
        if not nonzero_frames:
            continue
        first_nonzero = nonzero_frames[0]
        longest_run = 0
        longest_run_start = nonzero_frames[0]
        current_run_len = 1
        current_run_start = nonzero_frames[0]
        for j in range(1, len(nonzero_frames)):
            if nonzero_frames[j] == nonzero_frames[j - 1] + 1:
                current_run_len += 1
            else:
                if current_run_len > longest_run:
                    longest_run = current_run_len
                    longest_run_start = current_run_start
                current_run_len = 1
                current_run_start = nonzero_frames[j]
        if current_run_len > longest_run:
            longest_run = current_run_len
            longest_run_start = current_run_start
        insights.append(
            f"Trial {t.trial_index}: Frame-ID cross-camera misalignment first at frame {first_nonzero}, "
            f"longest consecutive run of {longest_run} frames starting at frame {longest_run_start}"
        )

    # 9. Per-camera error rates
    for t in report.trials:
        if not t.has_diagnostics or t.camera_error_summary is None:
            continue
        for cam, counters in t.camera_error_summary.items():
            total = counters.get("total_acquired", 0)
            incomplete = counters.get("incomplete_frames", 0)
            exceptions = counters.get("exceptions", 0)
            gaps = counters.get("frame_id_gaps", 0)
            errors = incomplete + exceptions + gaps
            if errors > 0:
                rate = (errors / total * 100) if total > 0 else 0.0
                insights.append(
                    f"Trial {t.trial_index}: Camera {cam}: "
                    f"{incomplete} incomplete, {exceptions} exceptions, {gaps} gaps ({rate:.1f}% error rate)"
                )

    # 10. Camera stream stats: dropped frames at device/network level
    for t in report.trials:
        if not t.has_diagnostics or t.camera_stream_stats is None:
            continue
        for cam, stats in t.camera_stream_stats.items():
            dropped = stats.get("StreamDroppedFrameCount", 0)
            overflow = stats.get("TransferQueueOverflowCount", 0)
            if dropped > 0 or overflow > 0:
                parts = []
                if dropped > 0:
                    parts.append(f"dropped {dropped} frames")
                if overflow > 0:
                    parts.append(f"queue overflow {overflow}")
                insights.append(
                    f"Trial {t.trial_index}: Camera {cam} {', '.join(parts)} at device/network level"
                )

    # 11. PTP offset drift: significant change between recording start and end
    for t in report.trials:
        if t.ptp_offset_start is None or t.ptp_offset_end is None:
            continue
        for cam in t.ptp_offset_start:
            if cam not in t.ptp_offset_end:
                continue
            start_ns = t.ptp_offset_start[cam]
            end_ns = t.ptp_offset_end[cam]
            drift_us = abs(end_ns - start_ns) / 1000
            if drift_us > 100:
                insights.append(
                    f"Trial {t.trial_index}: Camera {cam} PTP offset drifted {drift_us:.0f} µs during recording "
                    f"(start={start_ns} ns, end={end_ns} ns)"
                )

    # 12. Camera temperature increase: flag if temperature rose >5°C
    for t in report.trials:
        if t.camera_temperature_start is None or t.camera_temperature_end is None:
            continue
        for cam in t.camera_temperature_start:
            if cam not in t.camera_temperature_end:
                continue
            start_temp = t.camera_temperature_start[cam]
            end_temp = t.camera_temperature_end[cam]
            delta = end_temp - start_temp
            if delta > 5.0:
                insights.append(
                    f"Trial {t.trial_index}: Camera {cam} temperature rose {delta:.1f}°C "
                    f"({start_temp:.1f}→{end_temp:.1f}°C)"
                )

    # 13. Sync timeout events
    for t in report.trials:
        if not t.sync_timeout_events:
            continue
        n_events = len(t.sync_timeout_events)
        cameras_involved: set[str] = set()
        for evt in t.sync_timeout_events:
            cameras_involved.update(evt.get("empty_cameras", []))
        insights.append(
            f"Trial {t.trial_index}: {n_events} sync timeout event(s), "
            f"stalled cameras: {', '.join(sorted(cameras_involved))}"
        )

    # 14. Timespread alerts summary
    for t in report.trials:
        if t.timespread_alerts is None or t.timespread_alerts.get("count", 0) == 0:
            continue
        alerts = t.timespread_alerts
        insights.append(
            f"Trial {t.trial_index}: {alerts['count']} timespread alert(s), "
            f"max {alerts['max_spread_ms']:.3f} ms, "
            f"frames {alerts['first_frame']}-{alerts['last_frame']}"
        )

    # 15. NIC rx_dropped increase during recording
    for t in report.trials:
        if not t.system_snapshots or len(t.system_snapshots) < 2:
            continue
        first = t.system_snapshots[0].get("nic", {})
        last = t.system_snapshots[-1].get("nic", {})
        first_dropped = first.get("rx_dropped", 0)
        last_dropped = last.get("rx_dropped", 0)
        delta_dropped = last_dropped - first_dropped
        if delta_dropped > 0:
            insights.append(
                f"Trial {t.trial_index}: NIC rx_dropped increased by {delta_dropped} during recording"
            )

    return insights


# --- Plotly Figures ---


def _build_trial_boundary_shapes(
    report: SessionSyncReport,
) -> tuple[list[dict], list[dict], list[int]]:
    """Build vertical line shapes and annotations at trial boundaries.

    Returns (shapes, annotations, cumulative_frame_offsets).
    """
    shapes: list[dict] = []
    annotations: list[dict] = []
    offsets: list[int] = []
    cumulative = 0
    for t in report.trials:
        offsets.append(cumulative)
        if cumulative > 0:
            shapes.append(
                dict(
                    type="line",
                    x0=cumulative,
                    x1=cumulative,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(dash="dash", color="gray", width=1),
                )
            )
            annotations.append(
                dict(
                    x=cumulative,
                    y=1.02,
                    yref="paper",
                    text=str(t.trial_index),
                    showarrow=False,
                    font=dict(size=9, color="gray"),
                )
            )
        cumulative += t.n_frames
    return shapes, annotations, offsets


def create_timestamp_delta_figure(report: SessionSyncReport) -> go.Figure:
    """Interactive plot of per-frame timestamp deltas from reference camera (ms)."""
    shapes, annotations, offsets = _build_trial_boundary_shapes(report)
    fig = go.Figure()

    legend_shown: set[str] = set()

    for t, offset in zip(report.trials, offsets):
        x = np.arange(t.n_frames) + offset
        for col_idx, cam_id in enumerate(t.delta_camera_ids):
            show = cam_id not in legend_shown
            legend_shown.add(cam_id)
            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=t.timestamp_delta_ms[:, col_idx],
                    mode="markers",
                    marker=dict(size=2),
                    name=cam_id,
                    legendgroup=cam_id,
                    showlegend=show,
                    hovertemplate=f"Trial {t.trial_index}, Frame %{{x}}, Camera {cam_id}: %{{y:.3f}} ms<extra></extra>",
                )
            )

    fig.update_layout(
        title="Timestamp Delta from Reference Camera",
        xaxis_title="Global Frame Index",
        yaxis_title="Delta (ms)",
        shapes=shapes,
        annotations=annotations,
        height=500,
        template="plotly_white",
    )
    return fig


def create_frame_id_delta_figure(report: SessionSyncReport) -> go.Figure:
    """Interactive plot of per-frame frame ID deltas from reference camera."""
    shapes, annotations, offsets = _build_trial_boundary_shapes(report)
    fig = go.Figure()

    legend_shown: set[str] = set()

    for t, offset in zip(report.trials, offsets):
        x = np.arange(t.n_frames) + offset
        for col_idx, cam_id in enumerate(t.delta_camera_ids):
            show = cam_id not in legend_shown
            legend_shown.add(cam_id)
            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=t.frame_id_delta[:, col_idx],
                    mode="markers",
                    marker=dict(size=2),
                    name=cam_id,
                    legendgroup=cam_id,
                    showlegend=show,
                    hovertemplate=f"Trial {t.trial_index}, Frame %{{x}}, Camera {cam_id}: %{{y:d}}<extra></extra>",
                )
            )

    fig.update_layout(
        title="Frame ID Delta from Reference Camera",
        xaxis_title="Global Frame Index",
        yaxis_title="Frame ID Delta",
        shapes=shapes,
        annotations=annotations,
        height=500,
        template="plotly_white",
        yaxis=dict(dtick=1),
    )
    return fig


def create_serial_data_delta_figure(report: SessionSyncReport) -> go.Figure | None:
    """Interactive plot of serial data deltas. Returns None if no serial data present."""
    if not any(t.serial_data_delta is not None for t in report.trials):
        return None

    shapes, annotations, offsets = _build_trial_boundary_shapes(report)
    fig = go.Figure()

    legend_shown: set[str] = set()

    for t, offset in zip(report.trials, offsets):
        if t.serial_data_delta is None:
            continue
        x = np.arange(t.n_frames) + offset
        for col_idx, cam_id in enumerate(t.delta_camera_ids):
            show = cam_id not in legend_shown
            legend_shown.add(cam_id)
            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=t.serial_data_delta[:, col_idx],
                    mode="markers",
                    marker=dict(size=2),
                    name=cam_id,
                    legendgroup=cam_id,
                    showlegend=show,
                    hovertemplate=f"Trial {t.trial_index}, Frame %{{x}}, Camera {cam_id}: %{{y:d}}<extra></extra>",
                )
            )

    fig.update_layout(
        title="Serial Data Delta from Reference Camera",
        xaxis_title="Global Frame Index",
        yaxis_title="Serial Data Delta",
        shapes=shapes,
        annotations=annotations,
        height=500,
        template="plotly_white",
    )
    return fig


def create_sync_wait_figure(report: SessionSyncReport) -> go.Figure | None:
    """Scatter of sync_wait_cycles per frame. Returns None for legacy data."""
    if not any(t.has_diagnostics for t in report.trials):
        return None

    shapes, annotations, offsets = _build_trial_boundary_shapes(report)
    fig = go.Figure()

    for t, offset in zip(report.trials, offsets):
        if t.sync_wait_cycles is None:
            continue
        x = np.arange(t.n_frames) + offset
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=t.sync_wait_cycles,
                mode="markers",
                marker=dict(size=3),
                name=f"Trial {t.trial_index}",
                hovertemplate=f"Trial {t.trial_index}, Frame %{{x}}: %{{y}} wait cycles<extra></extra>",
            )
        )

    fig.update_layout(
        title="Sync Wait Cycles per Frame",
        xaxis_title="Global Frame Index",
        yaxis_title="Wait Cycles",
        shapes=shapes,
        annotations=annotations,
        height=500,
        template="plotly_white",
    )
    return fig


def create_queue_depth_figure(report: SessionSyncReport) -> go.Figure | None:
    """Per-camera scatter of queue depths over time. Returns None for legacy data."""
    if not any(t.has_diagnostics and t.queue_depths for t in report.trials):
        return None

    shapes, annotations, offsets = _build_trial_boundary_shapes(report)
    fig = go.Figure()
    legend_shown: set[str] = set()

    for t, offset in zip(report.trials, offsets):
        if t.queue_depths is None:
            continue
        x = np.arange(len(t.queue_depths)) + offset
        all_cams = sorted({cam for qd in t.queue_depths for cam in qd})
        for cam in all_cams:
            depths = [qd.get(cam, 0) for qd in t.queue_depths]
            show = cam not in legend_shown
            legend_shown.add(cam)
            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=depths,
                    mode="markers",
                    marker=dict(size=2),
                    name=cam,
                    legendgroup=cam,
                    showlegend=show,
                    hovertemplate=f"Trial {t.trial_index}, Frame %{{x}}, Camera {cam}: depth %{{y}}<extra></extra>",
                )
            )

    fig.update_layout(
        title="Acquisition Queue Depth per Camera",
        xaxis_title="Global Frame Index",
        yaxis_title="Queue Depth",
        shapes=shapes,
        annotations=annotations,
        height=500,
        template="plotly_white",
    )
    return fig


def create_camera_error_figure(report: SessionSyncReport) -> go.Figure | None:
    """Grouped bar chart of per-camera error counts. Returns None for legacy data."""
    error_summaries = [
        t.camera_error_summary
        for t in report.trials
        if t.has_diagnostics and t.camera_error_summary
    ]
    if not error_summaries:
        return None

    aggregated: dict[str, dict[str, int]] = {}
    for es in error_summaries:
        for cam, counters in es.items():
            if cam not in aggregated:
                aggregated[cam] = {
                    "incomplete_frames": 0,
                    "exceptions": 0,
                    "frame_id_gaps": 0,
                }
            for k in aggregated[cam]:
                aggregated[cam][k] += counters.get(k, 0)

    cameras = sorted(aggregated.keys())
    error_types = ["incomplete_frames", "exceptions", "frame_id_gaps"]
    labels = ["Incomplete", "Exceptions", "Frame ID Gaps"]

    fig = go.Figure()
    for error_type, label in zip(error_types, labels):
        values = [aggregated[cam][error_type] for cam in cameras]
        fig.add_trace(go.Bar(name=label, x=cameras, y=values))

    fig.update_layout(
        title="Per-Camera Error Counts",
        xaxis_title="Camera Serial",
        yaxis_title="Count",
        barmode="group",
        height=500,
        template="plotly_white",
    )
    return fig


def save_figures(
    report: SessionSyncReport,
    output_dir: Path | None = None,
    include_serial: bool = False,
) -> list[Path]:
    """Generate and save all diagnostic plots as standalone HTML files."""
    out = output_dir or (report.data_dir / "diagnostics")
    out.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    figures: list[tuple[str, go.Figure | None]] = [
        ("timestamp_delta", create_timestamp_delta_figure(report)),
        ("frame_id_delta", create_frame_id_delta_figure(report)),
        ("sync_wait_cycles", create_sync_wait_figure(report)),
        ("queue_depth", create_queue_depth_figure(report)),
        ("camera_errors", create_camera_error_figure(report)),
    ]
    if include_serial:
        figures.append(("serial_data_delta", create_serial_data_delta_figure(report)))

    for name, fig in figures:
        if fig is None:
            continue
        path = out / f"{name}.html"
        fig.write_html(str(path), include_plotlyjs=True)
        saved.append(path)

    return saved


# --- CLI ---


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze multi-camera sync quality from FLIR GigE acquisition JSON metadata.",
    )
    parser.add_argument(
        "data_path", type=Path, help="Directory containing JSON metadata files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory for plots",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation (terminal report only)",
    )
    parser.add_argument(
        "--serial-data",
        action="store_true",
        help="Include serial data delta plot (off by default)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> SessionSyncReport:
    args = parse_args(argv)
    data_dir = args.data_path.resolve()

    report = load_session(data_dir)
    print_report(report)

    if not args.no_plots:
        saved = save_figures(
            report, output_dir=args.output_dir, include_serial=args.serial_data
        )
        for p in saved:
            print(f"  Saved: {p}")
        print()

    return report


if __name__ == "__main__":
    main()
