"""Tests for ``HealthIdlePoller``.

Uses a short poll interval + a ``threading.Event`` to synchronize with the
daemon thread without sleeping arbitrary amounts.
"""

from __future__ import annotations

import datetime
import threading


from multi_camera.acquisition.health import (
    CameraReachabilityReport,
    DhcpServerStatus,
    Finding,
    HealthCheckReport,
    HealthConfig,
    HealthIdlePoller,
    HostNetworkStatus,
    severity_changed,
)


UTC = datetime.timezone.utc


def _make_report(
    severity: str = "ok",
    missing_cameras: list[str] | None = None,
) -> HealthCheckReport:
    missing_cameras = missing_cameras or []
    return HealthCheckReport(
        generated_at=datetime.datetime.now(UTC),
        deployment_mode="laptop",
        overall=severity,  # type: ignore[arg-type]
        dhcp=DhcpServerStatus(applicable=True, findings=[]),
        cameras=CameraReachabilityReport(
            expected=[],
            detected=[],
            missing=missing_cameras,
            extra=[],
            cameras=[],
            enumerated=True,
            findings=[],
        ),
        host_network=HostNetworkStatus(
            interface="enp5s0", interface_present=True, findings=[]
        ),
        recording_state="Idle",
        findings=[Finding(level=severity, code="x", message="x")],  # type: ignore[arg-type]
    )


class TestSeverityChanged:
    def test_new_report_with_ok_and_no_previous_is_unchanged(self) -> None:
        assert severity_changed(_make_report("ok"), None) is False

    def test_new_report_non_ok_and_no_previous_is_changed(self) -> None:
        assert severity_changed(_make_report("error"), None) is True

    def test_same_severity_unchanged(self) -> None:
        prev = _make_report("warn")
        new = _make_report("warn")
        assert severity_changed(new, prev) is False

    def test_different_severity_changed(self) -> None:
        prev = _make_report("ok")
        new = _make_report("error")
        assert severity_changed(new, prev) is True


class TestHealthIdlePoller:
    def test_runs_when_not_recording(self) -> None:
        config = HealthConfig(idle_poll_s=0.05)
        polled = threading.Event()

        reports = [_make_report("ok"), _make_report("error")]

        def run_check():
            return reports.pop(0)

        def on_poll(new, previous):
            polled.set()

        poller = HealthIdlePoller(
            config=config,
            run_check=run_check,
            on_poll=on_poll,
        )
        poller.start()
        try:
            assert polled.wait(timeout=1.0), "poller never ran a check"
        finally:
            poller.stop()

    def test_runs_unconditionally(self) -> None:
        """The poller no longer gates on recording state — adapting is the
        caller's job (run_check decides whether to skip camera enumeration,
        etc.). This test pins that contract.
        """
        config = HealthConfig(idle_poll_s=0.05)
        call_count = {"n": 0}
        done = threading.Event()

        def run_check():
            call_count["n"] += 1
            if call_count["n"] >= 2:
                done.set()
            return _make_report("ok")

        poller = HealthIdlePoller(
            config=config,
            run_check=run_check,
            on_poll=lambda new, prev: None,
        )
        poller.start()
        try:
            assert done.wait(timeout=1.0), "poller stopped firing"
        finally:
            poller.stop()

    def test_tracks_previous_report_between_polls(self) -> None:
        config = HealthConfig(idle_poll_s=0.05)
        reports = [_make_report("ok"), _make_report("error"), _make_report("error")]
        observed: list[tuple[str, str | None]] = []
        done = threading.Event()

        def run_check():
            return reports.pop(0) if reports else _make_report("ok")

        def on_poll(new, previous):
            observed.append((new.overall, previous.overall if previous else None))
            if len(observed) >= 3:
                done.set()

        poller = HealthIdlePoller(
            config=config,
            run_check=run_check,
            on_poll=on_poll,
        )
        poller.start()
        try:
            assert done.wait(timeout=2.0)
        finally:
            poller.stop()

        assert observed[0] == ("ok", None)
        assert observed[1] == ("error", "ok")
        assert observed[2] == ("error", "error")

    def test_survives_run_check_exception(self) -> None:
        config = HealthConfig(idle_poll_s=0.05)
        call_count = {"n": 0}
        done = threading.Event()

        def run_check():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("boom")
            return _make_report("ok")

        def on_poll(new, previous):
            done.set()

        poller = HealthIdlePoller(
            config=config,
            run_check=run_check,
            on_poll=on_poll,
        )
        poller.start()
        try:
            assert done.wait(timeout=2.0), "poller died after run_check raised"
        finally:
            poller.stop()
        assert call_count["n"] >= 2

    def test_start_is_idempotent(self) -> None:
        config = HealthConfig(idle_poll_s=0.05)
        poller = HealthIdlePoller(
            config=config,
            run_check=lambda: _make_report("ok"),
            on_poll=lambda new, prev: None,
        )
        poller.start()
        thread1 = poller._thread
        poller.start()
        thread2 = poller._thread
        try:
            assert thread1 is thread2
        finally:
            poller.stop()

    def test_stop_is_safe_if_never_started(self) -> None:
        config = HealthConfig(idle_poll_s=0.05)
        poller = HealthIdlePoller(
            config=config,
            run_check=lambda: _make_report("ok"),
            on_poll=lambda new, prev: None,
        )
        poller.stop()  # must not raise
