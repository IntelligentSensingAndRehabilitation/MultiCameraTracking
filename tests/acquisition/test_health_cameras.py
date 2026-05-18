"""Tests for ``health.check_camera_reachability`` and ``run_health_check``.

Uses fake PySpin handles + injected enumerator/runners — no Spinnaker SDK,
no real cameras, no live DHCP server.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path


from multi_camera.acquisition.health import (
    DetectedCamera,
    HealthConfig,
    _int_to_ipv4,
    _snapshot_from_recorder,
    check_camera_reachability,
    run_health_check,
)


UTC = datetime.timezone.utc


class FakeRecorderCam:
    """Stand-in for a ``simple_pyspin.Camera`` handle.

    Mirrors the read-only attribute access pattern used by ``_snapshot_from_recorder``:
    ``DeviceSerialNumber`` / ``GevCurrentIPAddress`` / ``GevLinkSpeed``.
    """

    def __init__(
        self, serial: str, ip_int: int | None = None, link_speed_bps: int | None = None
    ):
        self.DeviceSerialNumber = serial
        if ip_int is not None:
            self.GevCurrentIPAddress = ip_int
        if link_speed_bps is not None:
            self.GevLinkSpeed = link_speed_bps


class FakeRecorder:
    def __init__(self, cams: list[FakeRecorderCam]):
        self.cams = cams


class TestIpInt:
    def test_converts_ipv4_int(self) -> None:
        # 192.168.1.50 = 0xC0A80132
        assert _int_to_ipv4(0xC0A80132) == "192.168.1.50"

    def test_invalid_returns_none(self) -> None:
        assert _int_to_ipv4(-1) is None


class TestRecorderSnapshot:
    def test_reads_full_snapshot(self) -> None:
        cam = FakeRecorderCam(
            "22315678", ip_int=0xC0A80132, link_speed_bps=1_000_000_000
        )
        snap = _snapshot_from_recorder(cam)
        assert snap is not None
        assert snap.serial == "22315678"
        assert snap.ip == "192.168.1.50"
        assert snap.link_speed_mbps == 1000

    def test_missing_serial_returns_none(self) -> None:
        cam = FakeRecorderCam("")
        cam.DeviceSerialNumber = None
        assert _snapshot_from_recorder(cam) is None

    def test_partial_snapshot_when_attrs_missing(self) -> None:
        cam = FakeRecorderCam("22315678")
        snap = _snapshot_from_recorder(cam)
        assert snap is not None
        assert snap.ip is None
        assert snap.link_speed_mbps is None


class TestEnumeratorPath:
    def test_all_expected_detected(self) -> None:
        def fake_enum() -> list[DetectedCamera]:
            return [
                DetectedCamera(serial="111", ip="192.168.1.51", link_speed_mbps=1000),
                DetectedCamera(serial="222", ip="192.168.1.52", link_speed_mbps=1000),
            ]

        report = check_camera_reachability(
            expected_serials=["111", "222"],
            recorder=None,
            enumerator=fake_enum,
        )
        assert report.missing == []
        assert report.extra == []
        assert report.detected == ["111", "222"]
        assert report.enumerated is True
        assert [f.code for f in report.findings] == ["cameras_ok"]
        assert report.severity == "ok"

    def test_one_missing(self) -> None:
        def fake_enum() -> list[DetectedCamera]:
            return [DetectedCamera(serial="111")]

        report = check_camera_reachability(
            expected_serials=["111", "222"],
            recorder=None,
            enumerator=fake_enum,
        )
        assert report.missing == ["222"]
        assert any(f.code == "camera_unreachable" for f in report.findings)
        # Missing cameras warn but don't escalate to error — the recorder
        # gracefully proceeds with whatever cameras are reachable.
        assert report.severity == "warn"
        missing_cam = next(c for c in report.cameras if c.serial == "222")
        assert missing_cam.detected is False
        assert missing_cam.expected is True

    def test_one_extra_silent_by_default(self) -> None:
        def fake_enum() -> list[DetectedCamera]:
            return [
                DetectedCamera(serial="111"),
                DetectedCamera(serial="222"),
                DetectedCamera(serial="999"),
            ]

        report = check_camera_reachability(
            expected_serials=["111", "222"],
            recorder=None,
            enumerator=fake_enum,
        )
        assert report.extra == ["999"]
        assert not any(f.code == "unexpected_camera" for f in report.findings)
        assert report.severity == "ok"

    def test_one_extra_visible_with_include_unexpected(self) -> None:
        def fake_enum() -> list[DetectedCamera]:
            return [
                DetectedCamera(serial="111"),
                DetectedCamera(serial="222"),
                DetectedCamera(serial="999"),
            ]

        report = check_camera_reachability(
            expected_serials=["111", "222"],
            recorder=None,
            include_unexpected=True,
            enumerator=fake_enum,
        )
        unexpected = [f for f in report.findings if f.code == "unexpected_camera"]
        assert len(unexpected) == 1
        # Demoted to ok-level — informational, not actionable.
        assert unexpected[0].level == "ok"

    def test_low_link_speed_flagged(self) -> None:
        def fake_enum() -> list[DetectedCamera]:
            return [DetectedCamera(serial="111", link_speed_mbps=100)]

        report = check_camera_reachability(
            expected_serials=["111"],
            recorder=None,
            enumerator=fake_enum,
            minimum_link_speed_mbps=1000,
        )
        assert any(f.code == "camera_link_speed_low" for f in report.findings)
        assert report.severity == "warn"

    def test_enumeration_failure_is_error(self) -> None:
        def raising_enum() -> list[DetectedCamera]:
            raise RuntimeError("PySpin bus failure")

        report = check_camera_reachability(
            expected_serials=["111"],
            recorder=None,
            enumerator=raising_enum,
        )
        assert report.missing == ["111"]
        assert any(f.code == "camera_enumeration_failed" for f in report.findings)
        assert report.severity == "error"


class TestThroughputOutlier:
    def test_one_camera_throttled_fires_outlier_finding(self) -> None:
        """Catches the 100 Mbps stuck-link case where GevDeviceLinkSpeed may
        misreport but DeviceLinkThroughputLimit is accurate."""

        def fake_enum() -> list[DetectedCamera]:
            return [
                DetectedCamera(
                    serial="A",
                    link_speed_mbps=1000,
                    link_throughput_bytes_per_sec=92_000_000,
                ),
                DetectedCamera(
                    serial="B",
                    link_speed_mbps=1000,
                    link_throughput_bytes_per_sec=92_000_000,
                ),
                DetectedCamera(
                    serial="C",
                    link_speed_mbps=1000,
                    link_throughput_bytes_per_sec=92_000_000,
                ),
                DetectedCamera(
                    serial="D",
                    link_speed_mbps=1000,
                    link_throughput_bytes_per_sec=12_000_000,
                ),
            ]

        report = check_camera_reachability(
            expected_serials=["A", "B", "C", "D"],
            recorder=None,
            enumerator=fake_enum,
        )
        outliers = [f for f in report.findings if f.code == "camera_throughput_outlier"]
        assert len(outliers) == 1
        assert "D" in outliers[0].message
        assert outliers[0].details["serial"] == "D"

    def test_uniform_throughput_no_outlier(self) -> None:
        def fake_enum() -> list[DetectedCamera]:
            return [
                DetectedCamera(
                    serial=str(i),
                    link_speed_mbps=1000,
                    link_throughput_bytes_per_sec=92_000_000,
                )
                for i in range(4)
            ]

        report = check_camera_reachability(
            expected_serials=[str(i) for i in range(4)],
            recorder=None,
            enumerator=fake_enum,
        )
        assert not any(f.code == "camera_throughput_outlier" for f in report.findings)

    def test_below_three_cameras_skips_detector(self) -> None:
        """Median is meaningless with only 1-2 samples — don't fire."""

        def fake_enum() -> list[DetectedCamera]:
            return [
                DetectedCamera(serial="A", link_throughput_bytes_per_sec=92_000_000),
                DetectedCamera(serial="B", link_throughput_bytes_per_sec=12_000_000),
            ]

        report = check_camera_reachability(
            expected_serials=["A", "B"],
            recorder=None,
            enumerator=fake_enum,
        )
        assert not any(f.code == "camera_throughput_outlier" for f in report.findings)


class TestShortCircuit:
    def test_no_recorder_no_expected_short_circuits(self) -> None:
        """When nothing is configured, skip PySpin enumeration entirely."""

        def boom() -> list[DetectedCamera]:  # pragma: no cover - should not be called
            raise AssertionError("enumerator should not run when nothing to check")

        report = check_camera_reachability(
            expected_serials=[],
            recorder=None,
            enumerator=boom,
        )
        assert report.enumerated is False
        assert report.missing == []
        assert report.detected == []
        assert [f.code for f in report.findings] == ["cameras_not_configured"]
        assert report.severity == "ok"

    def test_empty_recorder_no_expected_also_short_circuits(self) -> None:
        class _EmptyRecorder:
            cams: list = []

        def boom() -> list[DetectedCamera]:  # pragma: no cover
            raise AssertionError("enumerator should not run")

        report = check_camera_reachability(
            expected_serials=[],
            recorder=_EmptyRecorder(),
            enumerator=boom,
        )
        assert [f.code for f in report.findings] == ["cameras_not_configured"]


class TestRecorderHandlePath:
    def test_reads_from_held_handles_no_enumeration(self) -> None:
        recorder = FakeRecorder(
            cams=[
                FakeRecorderCam("111", ip_int=0xC0A80133, link_speed_bps=1_000_000_000),
                FakeRecorderCam("222", ip_int=0xC0A80134, link_speed_bps=1_000_000_000),
            ]
        )

        def boom() -> list[DetectedCamera]:  # pragma: no cover - should not be called
            raise AssertionError("enumerator should not run when recorder has cams")

        report = check_camera_reachability(
            expected_serials=["111", "222"],
            recorder=recorder,
            enumerator=boom,
        )
        assert report.enumerated is False
        assert report.missing == []
        cam_111 = next(c for c in report.cameras if c.serial == "111")
        assert cam_111.ip == "192.168.1.51"
        assert cam_111.link_speed_mbps == 1000

    def test_empty_recorder_falls_through_to_enumerator(self) -> None:
        recorder = FakeRecorder(cams=[])

        def fake_enum() -> list[DetectedCamera]:
            return [DetectedCamera(serial="111")]

        report = check_camera_reachability(
            expected_serials=["111"],
            recorder=recorder,
            enumerator=fake_enum,
        )
        assert report.enumerated is True
        assert report.missing == []


class TestRunHealthCheck:
    def test_end_to_end_all_green(self, tmp_path: Path) -> None:
        lease_file = tmp_path / "dhcpd.leases"
        lease_file.write_text(
            "lease 192.168.1.50 {\n  ends 1 2099/01/01 12:00:00;\n}\n"
        )
        config = HealthConfig(
            deployment_mode="laptop",
            network_interface="enp5s0",
            dhcp_lease_file=lease_file,
            dhcp_expected_interface_ip="192.168.1.1",
        )

        def fake_enum() -> list[DetectedCamera]:
            return [
                DetectedCamera(serial="111", ip="192.168.1.50", link_speed_mbps=1000)
            ]

        report = run_health_check(
            config=config,
            expected_serials=["111"],
            recorder=None,
            recording_state="Idle",
            ip_addr_runner=lambda iface, timeout_s=1.0: "    inet 192.168.1.1/24",
            camera_enumerator=fake_enum,
            now=datetime.datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC),
        )
        # host_network runs against the real host here; in the dev container
        # MTU/rmem aren't configured for jumbo frames, so just verify the dhcp
        # and camera arms are green and that host_network ran (any severity).
        assert report.cameras.severity == "ok"
        assert report.dhcp.severity == "ok"
        assert report.host_network is not None
        assert report.recording_state == "Idle"

    def test_overall_is_worst_severity(self, tmp_path: Path) -> None:
        lease_file = tmp_path / "dhcpd.leases"
        lease_file.write_text(
            "lease 192.168.1.50 {\n  ends 1 2099/01/01 12:00:00;\n}\n"
        )
        config = HealthConfig(
            deployment_mode="laptop",
            network_interface="enp5s0",
            dhcp_lease_file=lease_file,
        )

        def fake_enum() -> list[DetectedCamera]:
            return []  # missing camera -> error

        report = run_health_check(
            config=config,
            expected_serials=["111"],
            recorder=None,
            ip_addr_runner=lambda iface, timeout_s=1.0: "    inet 192.168.1.1/24",
            camera_enumerator=fake_enum,
            now=datetime.datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC),
        )
        assert report.overall == "error"
        assert report.findings[0].level == "error"  # prioritized

    def test_serializes_to_json(self, tmp_path: Path) -> None:
        config = HealthConfig(
            deployment_mode="network",
            dhcp_lease_file=tmp_path / "never-read",
        )
        report = run_health_check(
            config=config,
            expected_serials=[],
            camera_enumerator=lambda: [],
            now=datetime.datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC),
        )
        dumped = report.model_dump(mode="json")
        assert set(dumped.keys()) >= {
            "generated_at",
            "deployment_mode",
            "overall",
            "dhcp",
            "cameras",
            "recording_state",
            "findings",
        }

    def test_completes_quickly_with_recorder(self, tmp_path: Path) -> None:
        lease_file = tmp_path / "dhcpd.leases"
        lease_file.write_text("")
        config = HealthConfig(
            deployment_mode="laptop",
            dhcp_lease_file=lease_file,
        )
        recorder = FakeRecorder(
            cams=[
                FakeRecorderCam("111", ip_int=0xC0A80132, link_speed_bps=1_000_000_000)
            ]
        )
        import time

        start = time.monotonic()
        report = run_health_check(
            config=config,
            expected_serials=["111"],
            recorder=recorder,
            ip_addr_runner=lambda iface, timeout_s=1.0: "    inet 192.168.1.1/24",
        )
        elapsed = time.monotonic() - start
        assert elapsed < 0.2, f"run_health_check took {elapsed:.3f}s, expected <0.2s"
        assert report.cameras.enumerated is False

    def test_skip_camera_enumeration_does_not_call_enumerator(
        self, tmp_path: Path
    ) -> None:
        """During recording, run_health_check must skip the PySpin enum
        (recorder owns the handles) but still run DHCP / host-network checks.
        """
        # Empty lease file with a stale mtime simulates a downed dhcpd.
        lease_file = tmp_path / "dhcpd.leases"
        lease_file.write_text("")
        now = datetime.datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
        stale = (now - datetime.timedelta(hours=1)).timestamp()
        os.utime(lease_file, (stale, stale))
        config = HealthConfig(
            deployment_mode="laptop",
            dhcp_lease_file=lease_file,
        )

        enum_calls = {"n": 0}

        def fake_enum() -> list[DetectedCamera]:
            enum_calls["n"] += 1
            return []

        report = run_health_check(
            config=config,
            expected_serials=["111", "222"],
            recorder=None,
            recording_state="Recording",
            skip_camera_enumeration=True,
            ip_addr_runner=lambda iface, timeout_s=1.0: "    inet 192.168.1.1/24",
            camera_enumerator=fake_enum,
            now=now,
        )

        assert enum_calls["n"] == 0
        assert report.cameras.enumerated is False
        assert report.cameras.findings == []
        # DHCP arm still ran and surfaced the downed service from lease evidence.
        assert any(f.code == "dhcp_service_down" for f in report.dhcp.findings)
        assert report.dhcp.severity == "error"


class TestHealthConfigFromEnv:
    def test_reads_common_env_vars(self) -> None:
        cfg = HealthConfig.from_env(
            {
                "DEPLOYMENT_MODE": "network",
                "NETWORK_INTERFACE": "eth0",
                "HEALTH_IDLE_POLL_S": "15.0",
            }
        )
        assert cfg.deployment_mode == "network"
        assert cfg.network_interface == "eth0"
        assert cfg.idle_poll_s == 15.0

    def test_bad_interval_falls_back_to_default(self) -> None:
        cfg = HealthConfig.from_env({"HEALTH_IDLE_POLL_S": "not a number"})
        assert cfg.idle_poll_s == 30.0

    def test_empty_env_uses_defaults(self) -> None:
        cfg = HealthConfig.from_env({})
        assert cfg.deployment_mode == "laptop"
        assert cfg.network_interface == "enp5s0"


class TestCameraOffSubnet:
    """GigE Vision discovery is link-local broadcast and crosses subnets, so a
    camera with a stale persistent IP or a 169.254.x.x link-local fallback
    still enumerates. ``check_camera_reachability`` must flag these explicitly
    — pre-fix, ``make health`` reported "all reachable" while ``Init()`` failed
    with ``Spinnaker: Camera is on a wrong subnet. [-1015]``.
    """

    def test_off_subnet_camera_produces_error_finding(self) -> None:
        import ipaddress

        def fake_enum() -> list[DetectedCamera]:
            return [
                DetectedCamera(serial="111", ip="192.168.1.51"),
                DetectedCamera(serial="222", ip="169.254.172.46"),
            ]

        report = check_camera_reachability(
            expected_serials=["111", "222"],
            recorder=None,
            enumerator=fake_enum,
            host_interface_subnet=ipaddress.IPv4Network("192.168.1.0/24"),
        )
        off_subnet = [f for f in report.findings if f.code == "camera_off_subnet"]
        assert len(off_subnet) == 1
        assert off_subnet[0].level == "error"
        assert "222" in off_subnet[0].message
        assert "169.254.172.46" in off_subnet[0].message
        assert off_subnet[0].details["serial"] == "222"
        assert off_subnet[0].details["host_subnet"] == "192.168.1.0/24"

    def test_all_on_subnet_emits_no_off_subnet_finding(self) -> None:
        import ipaddress

        def fake_enum() -> list[DetectedCamera]:
            return [
                DetectedCamera(serial="111", ip="192.168.1.51"),
                DetectedCamera(serial="222", ip="192.168.1.52"),
            ]

        report = check_camera_reachability(
            expected_serials=["111", "222"],
            recorder=None,
            enumerator=fake_enum,
            host_interface_subnet=ipaddress.IPv4Network("192.168.1.0/24"),
        )
        assert not any(f.code == "camera_off_subnet" for f in report.findings)

    def test_unknown_subnet_skips_check(self) -> None:
        def fake_enum() -> list[DetectedCamera]:
            return [DetectedCamera(serial="111", ip="169.254.1.1")]

        report = check_camera_reachability(
            expected_serials=["111"],
            recorder=None,
            enumerator=fake_enum,
            host_interface_subnet=None,
        )
        assert not any(f.code == "camera_off_subnet" for f in report.findings)

    def test_camera_without_ip_is_skipped(self) -> None:
        import ipaddress

        def fake_enum() -> list[DetectedCamera]:
            return [DetectedCamera(serial="111", ip=None)]

        report = check_camera_reachability(
            expected_serials=["111"],
            recorder=None,
            enumerator=fake_enum,
            host_interface_subnet=ipaddress.IPv4Network("192.168.1.0/24"),
        )
        assert not any(f.code == "camera_off_subnet" for f in report.findings)

    def test_multiple_off_subnet_each_get_their_own_finding(self) -> None:
        import ipaddress

        def fake_enum() -> list[DetectedCamera]:
            return [
                DetectedCamera(serial="aaa", ip="169.254.1.1"),
                DetectedCamera(serial="bbb", ip="192.168.1.51"),
                DetectedCamera(serial="ccc", ip="10.0.0.5"),
            ]

        report = check_camera_reachability(
            expected_serials=["aaa", "bbb", "ccc"],
            recorder=None,
            enumerator=fake_enum,
            host_interface_subnet=ipaddress.IPv4Network("192.168.1.0/24"),
        )
        off = sorted(
            (f.details["serial"], f.details["camera_ip"])
            for f in report.findings
            if f.code == "camera_off_subnet"
        )
        assert off == [("aaa", "169.254.1.1"), ("ccc", "10.0.0.5")]

    def test_severity_becomes_error_when_off_subnet_present(self) -> None:
        import ipaddress

        def fake_enum() -> list[DetectedCamera]:
            return [DetectedCamera(serial="111", ip="169.254.1.1")]

        report = check_camera_reachability(
            expected_serials=["111"],
            recorder=None,
            enumerator=fake_enum,
            host_interface_subnet=ipaddress.IPv4Network("192.168.1.0/24"),
        )
        assert report.severity == "error"
