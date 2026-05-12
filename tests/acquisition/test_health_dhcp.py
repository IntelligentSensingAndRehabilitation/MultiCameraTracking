"""Tests for ``health.check_dhcp_server`` and related parsing helpers.

All tests use temporary lease files and injected subprocess runners — no
actual DHCP server or systemctl call required.
"""

from __future__ import annotations

import datetime
from pathlib import Path


from multi_camera.acquisition.health import (
    DhcpServerStatus,
    _parse_interface_ipv4,
    _parse_lease_file,
    check_dhcp_server,
)


UTC = datetime.timezone.utc


def _make_runner(stdout: str, returncode: int = 0):
    def runner(service: str, timeout_s: float = 1.0) -> tuple[str, int]:
        return stdout, returncode

    return runner


def _make_ip_runner(stdout: str):
    def runner(interface: str, timeout_s: float = 1.0) -> str:
        return stdout

    return runner


class TestLeaseFileParsing:
    def test_counts_active_leases(self) -> None:
        text = """
        lease 192.168.1.50 {
          starts 1 2026/04/20 12:00:00;
          ends 1 2027/04/20 12:00:00;
          hardware ethernet 00:11:22:33:44:55;
        }
        lease 192.168.1.51 {
          starts 1 2026/04/21 12:00:00;
          ends 1 2027/04/21 12:00:00;
        }
        """
        now = datetime.datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
        assert _parse_lease_file(text, now) == 2

    def test_skips_expired_leases(self) -> None:
        text = """
        lease 192.168.1.50 {
          ends 1 2025/01/01 12:00:00;
        }
        lease 192.168.1.51 {
          ends 1 2099/01/01 12:00:00;
        }
        """
        now = datetime.datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
        assert _parse_lease_file(text, now) == 1

    def test_skips_abandoned_leases(self) -> None:
        text = """
        lease 192.168.1.50 {
          ends 1 2099/01/01 12:00:00;
          abandoned;
        }
        """
        now = datetime.datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
        assert _parse_lease_file(text, now) == 0

    def test_empty_file_returns_zero(self) -> None:
        assert _parse_lease_file("", datetime.datetime.now(UTC)) == 0

    def test_handles_ends_never(self) -> None:
        text = """
        lease 192.168.1.50 {
          ends never;
        }
        """
        assert _parse_lease_file(text, datetime.datetime.now(UTC)) == 1

    def test_handles_malformed_gracefully(self) -> None:
        text = "garbage { not a real lease }"
        assert _parse_lease_file(text, datetime.datetime.now(UTC)) == 0


class TestInterfaceIpParsing:
    def test_extracts_first_ipv4(self) -> None:
        text = "3: enp5s0: <...> mtu 9000 ...\n    inet 192.168.1.1/24 brd 192.168.1.255 scope global enp5s0\n"
        assert _parse_interface_ipv4(text) == "192.168.1.1"

    def test_returns_none_if_no_ipv4(self) -> None:
        text = "3: enp5s0: <...> mtu 9000 ...\n"
        assert _parse_interface_ipv4(text) is None


class TestNetworkMode:
    def test_short_circuits_in_network_mode(self) -> None:
        # Even if runners would fail, network mode should not call them.
        def boom(*args, **kwargs):  # noqa: ARG001
            raise AssertionError("should not be called in network mode")

        status = check_dhcp_server(
            mode="network",
            lease_file=Path("/nonexistent"),
            interface="fake",
            systemctl_runner=boom,
            ip_addr_runner=boom,
        )
        assert status.applicable is False
        assert any(f.code == "dhcp_not_applicable" for f in status.findings)
        assert status.severity == "ok"


class TestLaptopModeHappyPath:
    def test_all_green(self, tmp_path: Path) -> None:
        lease_file = tmp_path / "dhcpd.leases"
        lease_file.write_text(
            "lease 192.168.1.50 {\n  ends 1 2099/01/01 12:00:00;\n}\n"
        )
        status = check_dhcp_server(
            mode="laptop",
            lease_file=lease_file,
            interface="enp5s0",
            expected_interface_ip="192.168.1.1",
            systemctl_runner=_make_runner("active", 0),
            ip_addr_runner=_make_ip_runner("    inet 192.168.1.1/24 brd 192.168.1.255"),
            now=datetime.datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC),
        )
        assert status.applicable is True
        assert status.service_active is True
        assert status.lease_count == 1
        assert status.interface_ip == "192.168.1.1"
        assert status.severity == "ok"
        assert [f.code for f in status.findings] == ["dhcp_ok"]


class TestLaptopModeFailures:
    def test_service_down(self, tmp_path: Path) -> None:
        lease_file = tmp_path / "dhcpd.leases"
        lease_file.write_text("")
        status = check_dhcp_server(
            mode="laptop",
            lease_file=lease_file,
            interface="enp5s0",
            systemctl_runner=_make_runner("inactive", 3),
            ip_addr_runner=_make_ip_runner("    inet 192.168.1.1/24"),
        )
        assert status.service_active is False
        assert status.severity == "error"
        assert any(f.code == "dhcp_service_down" for f in status.findings)

    def test_interface_has_wrong_ip(self, tmp_path: Path) -> None:
        lease_file = tmp_path / "dhcpd.leases"
        lease_file.write_text(
            "lease 192.168.1.50 {\n  ends 1 2099/01/01 12:00:00;\n}\n"
        )
        status = check_dhcp_server(
            mode="laptop",
            lease_file=lease_file,
            interface="enp5s0",
            expected_interface_ip="192.168.1.1",
            systemctl_runner=_make_runner("active", 0),
            ip_addr_runner=_make_ip_runner("    inet 10.0.0.5/24"),
        )
        assert status.interface_ip == "10.0.0.5"
        assert any(f.code == "dhcp_interface_ip_unexpected" for f in status.findings)
        assert status.severity == "error"

    def test_interface_has_no_ip(self, tmp_path: Path) -> None:
        lease_file = tmp_path / "dhcpd.leases"
        lease_file.write_text("")
        status = check_dhcp_server(
            mode="laptop",
            lease_file=lease_file,
            interface="enp5s0",
            systemctl_runner=_make_runner("active", 0),
            ip_addr_runner=_make_ip_runner(""),
        )
        assert status.interface_ip is None
        assert any(f.code == "dhcp_interface_no_ip" for f in status.findings)
        assert status.severity == "error"

    def test_lease_file_missing(self, tmp_path: Path) -> None:
        status = check_dhcp_server(
            mode="laptop",
            lease_file=tmp_path / "does_not_exist.leases",
            interface=None,
            systemctl_runner=_make_runner("active", 0),
            ip_addr_runner=_make_ip_runner(""),
        )
        assert any(f.code == "dhcp_lease_file_missing" for f in status.findings)

    def test_active_but_no_leases(self, tmp_path: Path) -> None:
        lease_file = tmp_path / "dhcpd.leases"
        lease_file.write_text("")
        status = check_dhcp_server(
            mode="laptop",
            lease_file=lease_file,
            interface=None,
            systemctl_runner=_make_runner("active", 0),
            ip_addr_runner=_make_ip_runner(""),
        )
        codes = [f.code for f in status.findings]
        assert "dhcp_no_leases" in codes
        assert status.severity == "warn"

    def test_systemctl_subprocess_error_is_warn_not_error(self, tmp_path: Path) -> None:
        import subprocess

        def raising_runner(service: str, timeout_s: float = 1.0):
            raise subprocess.TimeoutExpired(["systemctl"], timeout_s)

        lease_file = tmp_path / "dhcpd.leases"
        lease_file.write_text("")
        status = check_dhcp_server(
            mode="laptop",
            lease_file=lease_file,
            systemctl_runner=raising_runner,
            ip_addr_runner=_make_ip_runner(""),
        )
        codes = [f.code for f in status.findings]
        assert "dhcp_service_check_failed" in codes
        assert status.severity == "warn"


class TestDhcpServerStatusShape:
    def test_serializes_to_json(self, tmp_path: Path) -> None:
        lease_file = tmp_path / "dhcpd.leases"
        lease_file.write_text("")
        status = check_dhcp_server(
            mode="network",
            lease_file=lease_file,
            interface=None,
            systemctl_runner=_make_runner("active", 0),
            ip_addr_runner=_make_ip_runner(""),
        )
        dumped = status.model_dump(mode="json")
        assert "findings" in dumped
        assert dumped["applicable"] is False

    def test_severity_property(self) -> None:
        status = DhcpServerStatus(applicable=True, findings=[])
        assert status.severity == "ok"
