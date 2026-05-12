"""Live health checks for the acquisition system.

Quick, one-shot checks that can run between recordings or be polled by an
operator-facing client (phone app, React GUI). Narrow scope: DHCP server
status and camera reachability. Per-trial / per-session sync analysis lives in
``diagnostics/json_parser.py``; continuous NIC/CPU/disk sampling during
recording lives in ``diagnostics/system_monitor.py``.

All I/O-doing helpers accept injectable runners/paths so the module
unit-tests without hardware or a live DHCP server.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import ipaddress
import logging
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["ok", "warn", "error", "unknown"]

_SEVERITY_ORDER: dict[Severity, int] = {"unknown": 0, "ok": 1, "warn": 2, "error": 3}


def max_severity(levels: list[Severity]) -> Severity:
    """Return the worst severity in the list. Empty list returns ``"ok"``."""
    if not levels:
        return "ok"
    return max(levels, key=lambda s: _SEVERITY_ORDER[s])


class Finding(BaseModel):
    """A single plain-English health observation for display to the operator.

    ``message`` is the short banner-friendly summary (kept under ~80 chars where
    practical). ``remediation`` is an optional ordered list of recovery steps,
    rendered as a numbered list inside the Diagnostics tab. Long step-by-step
    text belongs in ``remediation``, not ``message``.
    """

    level: Severity
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    remediation: list[str] | None = None


class DhcpServerStatus(BaseModel):
    applicable: bool
    service_active: bool | None = None
    lease_file_path: str | None = None
    lease_count: int = 0
    interface_ip: str | None = None
    findings: list[Finding] = Field(default_factory=list)

    @property
    def severity(self) -> Severity:
        return max_severity([f.level for f in self.findings])


class CameraReachability(BaseModel):
    serial: str
    detected: bool
    expected: bool
    excluded: bool = False
    ip: str | None = None
    link_speed_mbps: int | None = None
    link_throughput_bytes_per_sec: int | None = None
    locked_by_other_process: bool = False
    last_error: str | None = None


class CameraReachabilityReport(BaseModel):
    expected: list[str]
    detected: list[str]
    missing: list[str]
    extra: list[str]
    cameras: list[CameraReachability]
    enumerated: bool
    findings: list[Finding] = Field(default_factory=list)

    @property
    def severity(self) -> Severity:
        return max_severity([f.level for f in self.findings])


class HostNetworkStatus(BaseModel):
    """Host-side network configuration the cameras depend on."""

    interface: str
    interface_present: bool
    carrier_up: bool | None = None
    mtu: int | None = None
    expected_mtu: int = 9000
    rmem_max: int | None = None
    expected_rmem_max: int = 10_000_000
    findings: list[Finding] = Field(default_factory=list)

    @property
    def severity(self) -> Severity:
        return max_severity([f.level for f in self.findings])


class HealthCheckReport(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    generated_at: datetime.datetime
    deployment_mode: Literal["laptop", "network"]
    overall: Severity
    dhcp: DhcpServerStatus
    cameras: CameraReachabilityReport
    host_network: HostNetworkStatus
    recording_state: str
    findings: list[Finding]


class HealthConfig(BaseModel):
    """Tunable thresholds and paths. Load with :func:`HealthConfig.from_env`."""

    deployment_mode: Literal["laptop", "network"] = "laptop"
    network_interface: str = "enp5s0"
    dhcp_lease_file: Path = Path("/var/lib/dhcp/dhcpd.leases")
    dhcp_expected_interface_ip: str = "192.168.1.1"
    cache_ttl_s: float = 2.0
    dhcp_timeout_s: float = 1.0
    # 8s absorbs PySpin GigE-broadcast enumeration on a populated network-mode
    # subnet; the fast path (recorder already holds handles) returns in <1s.
    camera_timeout_s: float = 8.0
    host_network_timeout_s: float = 1.0
    total_timeout_s: float = 10.0
    idle_poll_s: float = 30.0
    inter_trial_health_check: bool = True
    preflight_camera_check: bool = True
    minimum_link_speed_mbps: int = 1000

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "HealthConfig":
        """Populate from environment variables. Unknown/missing vars fall back to defaults.

        Recognized env vars: DEPLOYMENT_MODE, NETWORK_INTERFACE, DHCP_LEASE_FILE,
        DHCP_EXPECTED_INTERFACE_IP, HEALTH_IDLE_POLL_S, HEALTH_CACHE_TTL_S.
        """
        env = env if env is not None else dict(os.environ)
        kwargs: dict[str, Any] = {}

        mode = env.get("DEPLOYMENT_MODE", "").strip().lower()
        if mode in ("laptop", "network"):
            kwargs["deployment_mode"] = mode

        iface = env.get("NETWORK_INTERFACE", "").strip()
        if iface:
            kwargs["network_interface"] = iface

        lease_file = env.get("DHCP_LEASE_FILE", "").strip()
        if lease_file:
            kwargs["dhcp_lease_file"] = Path(lease_file)

        expected_ip = env.get("DHCP_EXPECTED_INTERFACE_IP", "").strip()
        if expected_ip:
            kwargs["dhcp_expected_interface_ip"] = expected_ip

        for key, env_name, caster in [
            ("idle_poll_s", "HEALTH_IDLE_POLL_S", float),
            ("cache_ttl_s", "HEALTH_CACHE_TTL_S", float),
        ]:
            raw = env.get(env_name, "").strip()
            if raw:
                try:
                    kwargs[key] = caster(raw)
                except ValueError:
                    pass

        return cls(**kwargs)


# ---------------------------------------------------------------------------
# DHCP server check
# ---------------------------------------------------------------------------

SystemctlRunner = Callable[[str], tuple[str, int]]
IpAddrRunner = Callable[[str], str]


def _default_systemctl_runner(service: str, timeout_s: float = 1.0) -> tuple[str, int]:
    result = subprocess.run(
        ["systemctl", "is-active", service],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return result.stdout.strip(), result.returncode


def _default_ip_addr_runner(interface: str, timeout_s: float = 1.0) -> str:
    result = subprocess.run(
        ["ip", "-4", "addr", "show", interface],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return result.stdout


def _parse_interface_ipv4(ip_addr_output: str) -> str | None:
    """Extract the first IPv4 address from `ip -4 addr show` output."""
    match = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)", ip_addr_output)
    return match.group(1) if match else None


def _parse_lease_file(lease_file_text: str, now: datetime.datetime) -> int:
    """Count non-expired, non-abandoned DHCP leases in an isc-dhcp-server lease file.

    The format (one block per lease)::

        lease 192.168.1.50 {
          starts N 2026/04/22 12:34:56;
          ends N 2026/04/22 14:34:56;
          ...
          abandoned;    # optional
        }
    """
    count = 0
    for block in re.finditer(
        r"lease\s+\d+\.\d+\.\d+\.\d+\s*\{(.*?)\}", lease_file_text, re.DOTALL
    ):
        body = block.group(1)
        if "abandoned" in body:
            continue
        ends_match = re.search(
            r"ends\s+\d+\s+(\d{4})/(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{1,2}):(\d{1,2})",
            body,
        )
        if ends_match is None:
            # Malformed or lease with "never" expiration: count it.
            if re.search(r"ends\s+never", body):
                count += 1
            continue
        year, month, day, hour, minute, second = (int(g) for g in ends_match.groups())
        try:
            ends = datetime.datetime(
                year, month, day, hour, minute, second, tzinfo=datetime.timezone.utc
            )
        except ValueError:
            continue
        if ends > now:
            count += 1
    return count


def check_dhcp_server(
    mode: Literal["laptop", "network"],
    lease_file: Path = Path("/var/lib/dhcp/dhcpd.leases"),
    interface: str | None = None,
    expected_interface_ip: str = "192.168.1.1",
    systemctl_runner: SystemctlRunner = _default_systemctl_runner,
    ip_addr_runner: IpAddrRunner = _default_ip_addr_runner,
    now: datetime.datetime | None = None,
) -> DhcpServerStatus:
    """Check the DHCP server and interface state the cameras depend on.

    In ``network`` mode the upstream lab infrastructure manages DHCP so this
    check short-circuits and returns ``applicable=False``.
    """
    findings: list[Finding] = []

    if mode == "network":
        findings.append(
            Finding(
                level="ok",
                code="dhcp_not_applicable",
                message="Running in network mode — DHCP is managed upstream.",
            )
        )
        return DhcpServerStatus(
            applicable=False,
            lease_file_path=str(lease_file),
            findings=findings,
        )

    service_active: bool | None = None
    try:
        stdout, returncode = systemctl_runner("isc-dhcp-server")
        service_active = stdout == "active" and returncode == 0
        if not service_active:
            findings.append(
                Finding(
                    level="error",
                    code="dhcp_service_down",
                    message=(
                        "The camera IP server is not running. "
                        "Run: sudo systemctl start isc-dhcp-server"
                    ),
                    details={"systemctl_output": stdout, "returncode": returncode},
                )
            )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        findings.append(
            Finding(
                level="warn",
                code="dhcp_service_check_failed",
                message="Could not check DHCP server status.",
                details={"error": str(e)},
            )
        )

    interface_ip: str | None = None
    if interface:
        try:
            output = ip_addr_runner(interface)
            interface_ip = _parse_interface_ipv4(output)
            if interface_ip is None:
                findings.append(
                    Finding(
                        level="error",
                        code="dhcp_interface_no_ip",
                        message=f"Network interface {interface} has no IPv4 address.",
                    )
                )
            elif interface_ip != expected_interface_ip:
                findings.append(
                    Finding(
                        level="error",
                        code="dhcp_interface_ip_unexpected",
                        message=(
                            f"Network interface {interface} has IP {interface_ip} "
                            f"(expected {expected_interface_ip}). The DHCP server "
                            "won't bind to the camera subnet at this address — "
                            "cameras will not receive leases. Activate the "
                            "DHCP-Server nmcli profile, or check that the cable is "
                            "in the configured network port."
                        ),
                        details={
                            "actual": interface_ip,
                            "expected": expected_interface_ip,
                        },
                    )
                )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            findings.append(
                Finding(
                    level="warn",
                    code="dhcp_interface_check_failed",
                    message=f"Could not read interface {interface}.",
                    details={"error": str(e)},
                )
            )

    lease_count = 0
    if lease_file.exists():
        try:
            text = lease_file.read_text()
            lease_count = _parse_lease_file(
                text, now or datetime.datetime.now(datetime.timezone.utc)
            )
            if service_active and lease_count == 0:
                findings.append(
                    Finding(
                        level="warn",
                        code="dhcp_no_leases",
                        message=(
                            "DHCP server is running but no cameras have leases. "
                            "Cameras may be unplugged or still booting."
                        ),
                    )
                )
        except OSError as e:
            findings.append(
                Finding(
                    level="warn",
                    code="dhcp_lease_file_unreadable",
                    message=f"Could not read DHCP lease file at {lease_file}.",
                    details={"error": str(e)},
                )
            )
    else:
        # Missing lease file is not necessarily an error: server may not have issued leases yet.
        findings.append(
            Finding(
                level="warn",
                code="dhcp_lease_file_missing",
                message=f"DHCP lease file not found at {lease_file}.",
            )
        )

    if not findings:
        findings.append(
            Finding(level="ok", code="dhcp_ok", message="DHCP server is healthy.")
        )

    return DhcpServerStatus(
        applicable=True,
        service_active=service_active,
        lease_file_path=str(lease_file),
        lease_count=lease_count,
        interface_ip=interface_ip,
        findings=findings,
    )


# ---------------------------------------------------------------------------
# Camera reachability check
# ---------------------------------------------------------------------------


class DetectedCamera(BaseModel):
    """Minimal snapshot of a camera seen by PySpin or a held recorder handle."""

    serial: str
    ip: str | None = None
    link_speed_mbps: int | None = None
    # Camera's negotiated outbound throughput cap. Auto-set from link speed at
    # Init time, so a value much lower than other cameras' is the strongest
    # available signal that the camera fell back to a slower link (e.g. 100 Mbps
    # on a degraded cable). Only readable from an Init'd device, so the bare
    # PySpin enumerator path leaves this None.
    link_throughput_bytes_per_sec: int | None = None


class RecorderLike(Protocol):
    cams: list[Any]


def _read_camera_attribute(cam: Any, attr: str) -> Any:
    """Safely read a PySpin attribute from a ``simple_pyspin.Camera`` handle."""
    try:
        value = getattr(cam, attr)
        if callable(value):
            return value()
        return value
    except Exception:  # noqa: BLE001 — PySpin raises many custom exceptions
        return None


def _snapshot_from_recorder(cam: Any) -> DetectedCamera | None:
    """Build a DetectedCamera from a ``simple_pyspin.Camera`` handle already held open
    by a running FlirRecorder. Read-only attribute access; safe while recording."""
    serial = _read_camera_attribute(cam, "DeviceSerialNumber")
    if not serial:
        return None
    ip_int = _read_camera_attribute(cam, "GevCurrentIPAddress")
    ip = _int_to_ipv4(ip_int) if isinstance(ip_int, int) else None
    link_speed = _read_camera_attribute(cam, "GevLinkSpeed")
    link_mbps = (
        int(link_speed) // 1_000_000
        if isinstance(link_speed, int) and link_speed > 0
        else None
    )
    throughput = _read_camera_attribute(cam, "DeviceLinkThroughputLimit")
    throughput_int = int(throughput) if isinstance(throughput, int) else None
    return DetectedCamera(
        serial=str(serial),
        ip=ip,
        link_speed_mbps=link_mbps,
        link_throughput_bytes_per_sec=throughput_int,
    )


def _int_to_ipv4(value: int) -> str | None:
    try:
        return str(ipaddress.IPv4Address(value))
    except (ValueError, ipaddress.AddressValueError):
        return None


CameraEnumerator = Callable[[], list[DetectedCamera]]


def _default_camera_enumerator() -> list[DetectedCamera]:
    """Enumerate cameras via PySpin without initializing them.

    Read-only: opens the transport-layer node map for each GigE device, reads
    ``DeviceSerialNumber`` / ``GevDeviceIPAddress`` / ``GevDeviceLinkSpeed``,
    and returns a list. Never calls ``Camera.init()`` or any mutating operation.

    Importing PySpin requires the Spinnaker SDK; this function is only called
    from code paths that also depend on PySpin, so lazy import is fine.
    """
    try:
        import PySpin  # type: ignore[import-untyped]
    except ImportError:
        return []

    detected: list[DetectedCamera] = []
    system = PySpin.System.GetInstance()
    try:
        cam_list = system.GetCameras()
        try:
            for i in range(cam_list.GetSize()):
                cam = cam_list.GetByIndex(i)
                try:
                    tl_node_map = cam.GetTLDeviceNodeMap()
                    serial = _read_tl_string(tl_node_map, "DeviceSerialNumber")
                    if not serial:
                        continue
                    ip_int = _read_tl_int(tl_node_map, "GevDeviceIPAddress")
                    link_speed = _read_tl_int(tl_node_map, "GevDeviceLinkSpeed")
                    detected.append(
                        DetectedCamera(
                            serial=serial,
                            ip=_int_to_ipv4(ip_int) if ip_int is not None else None,
                            link_speed_mbps=(
                                link_speed // 1_000_000
                                if link_speed is not None and link_speed > 0
                                else None
                            ),
                        )
                    )
                finally:
                    del cam
        finally:
            cam_list.Clear()
    finally:
        # Don't release the system — simple_pyspin caches a singleton instance.
        pass

    return detected


def _read_tl_string(node_map: Any, name: str) -> str | None:
    try:
        import PySpin  # type: ignore[import-untyped]

        node = PySpin.CStringPtr(node_map.GetNode(name))
        if PySpin.IsAvailable(node) and PySpin.IsReadable(node):
            return node.GetValue()
    except Exception:  # noqa: BLE001
        return None
    return None


def _read_tl_int(node_map: Any, name: str) -> int | None:
    try:
        import PySpin  # type: ignore[import-untyped]

        node = PySpin.CIntegerPtr(node_map.GetNode(name))
        if PySpin.IsAvailable(node) and PySpin.IsReadable(node):
            return int(node.GetValue())
    except Exception:  # noqa: BLE001
        return None
    return None


def check_camera_reachability(
    expected_serials: list[str],
    recorder: RecorderLike | None = None,
    minimum_link_speed_mbps: int = 1000,
    include_unexpected: bool = False,
    enumerator: CameraEnumerator = _default_camera_enumerator,
    excluded_serials: set[str] | None = None,
) -> CameraReachabilityReport:
    """Report which expected cameras are reachable right now.

    - If ``recorder`` has open camera handles, read serials / IP / link speed
      directly off those handles — no new enumeration, no lock contention,
      safe during recording.
    - Otherwise, enumerate via PySpin transport-layer nodes (read-only; never
      calls ``Camera.init()``).
    - If neither the recorder has cams nor any expected serials are known,
      short-circuit: PySpin discovery in that state produces no useful signal
      and can be slow (GigE broadcast in network mode).
    """
    expected = [str(s) for s in expected_serials]
    recorder_has_cams = recorder is not None and bool(getattr(recorder, "cams", None))

    if not recorder_has_cams and not expected:
        return CameraReachabilityReport(
            expected=[],
            detected=[],
            missing=[],
            extra=[],
            cameras=[],
            enumerated=False,
            findings=[
                Finding(
                    level="ok",
                    code="cameras_not_configured",
                    message=(
                        "No camera config loaded — select a config to enable "
                        "camera reachability checks."
                    ),
                )
            ],
        )

    detected_cams: list[DetectedCamera] = []
    enumerated_fresh = True
    if recorder_has_cams:
        enumerated_fresh = False
        for cam in recorder.cams:
            snapshot = _snapshot_from_recorder(cam)
            if snapshot is not None:
                detected_cams.append(snapshot)
    else:
        try:
            detected_cams = enumerator()
        except Exception as e:  # noqa: BLE001 — PySpin raises custom exceptions
            return CameraReachabilityReport(
                expected=expected,
                detected=[],
                missing=expected,
                extra=[],
                cameras=[
                    CameraReachability(
                        serial=s, detected=False, expected=True, last_error=str(e)
                    )
                    for s in expected
                ],
                enumerated=True,
                findings=[
                    Finding(
                        level="error",
                        code="camera_enumeration_failed",
                        message="Could not enumerate cameras via PySpin.",
                        details={"error": str(e)},
                    )
                ],
            )

    detected_by_serial = {c.serial: c for c in detected_cams}
    detected_serials = sorted(detected_by_serial.keys())
    expected_set = set(expected)
    detected_set = set(detected_serials)
    missing = sorted(expected_set - detected_set)
    extra = sorted(detected_set - expected_set)

    excluded_set = set(excluded_serials or set())
    cameras: list[CameraReachability] = []
    for serial in sorted(expected_set | detected_set | excluded_set):
        cam = detected_by_serial.get(serial)
        cameras.append(
            CameraReachability(
                serial=serial,
                detected=cam is not None,
                expected=serial in expected_set,
                excluded=serial in excluded_set,
                ip=cam.ip if cam else None,
                link_speed_mbps=cam.link_speed_mbps if cam else None,
                link_throughput_bytes_per_sec=(
                    cam.link_throughput_bytes_per_sec if cam else None
                ),
            )
        )

    findings: list[Finding] = []
    for serial in missing:
        findings.append(
            Finding(
                level="warn",
                code="camera_unreachable",
                message=f"Camera {serial} is not reachable. Check its ethernet cable and power.",
                details={"serial": serial},
            )
        )

    # When some expected cameras are missing, confirm which ones ARE working
    # so the operator can distinguish per-camera issues from a system-wide one.
    # When all expected are reachable, fall through to the summary at the end.
    expected_detected = [c for c in cameras if c.detected and c.expected]
    if missing and expected_detected:
        for cam_info in expected_detected:
            speed_str = (
                f" at {cam_info.link_speed_mbps} Mbps"
                if cam_info.link_speed_mbps is not None
                else ""
            )
            findings.append(
                Finding(
                    level="ok",
                    code="camera_reachable",
                    message=f"Camera {cam_info.serial} is reachable{speed_str}.",
                    details={
                        "serial": cam_info.serial,
                        "ip": cam_info.ip,
                        "link_speed_mbps": cam_info.link_speed_mbps,
                    },
                )
            )

    # Hidden by default: in network mode the subnet is shared with other lab
    # setups, so detected-but-unexpected cameras are routine, not actionable.
    if include_unexpected:
        for serial in extra:
            findings.append(
                Finding(
                    level="ok",
                    code="unexpected_camera",
                    message=f"Camera {serial} is on the network but not in the current config.",
                    details={"serial": serial},
                )
            )

    for cam_info in cameras:
        if cam_info.detected and cam_info.link_speed_mbps is not None:
            if cam_info.link_speed_mbps < minimum_link_speed_mbps:
                findings.append(
                    Finding(
                        level="warn",
                        code="camera_link_speed_low",
                        message=(
                            f"Camera {cam_info.serial} is negotiated at "
                            f"{cam_info.link_speed_mbps} Mbps "
                            f"(expected ≥{minimum_link_speed_mbps}). Check cable/switch."
                        ),
                        details={
                            "serial": cam_info.serial,
                            "link_speed_mbps": cam_info.link_speed_mbps,
                        },
                    )
                )

    # Throughput-outlier detector. DeviceLinkThroughputLimit is the camera's
    # configured outbound cap, derived by Spinnaker from the link speed
    # negotiated at Init. A value far below other cameras is strong (but
    # indirect) evidence the link came up at a lower rate (e.g. 100 Mbps
    # fallback) — useful when GevDeviceLinkSpeed reports 1000 but the
    # negotiated PHY rate is actually lower. Fires when one camera's
    # throughput is < 50% of the median across detected cameras.
    throughputs = [
        c.link_throughput_bytes_per_sec
        for c in cameras
        if c.detected and c.link_throughput_bytes_per_sec is not None
    ]
    if len(throughputs) >= 3:
        sorted_t = sorted(throughputs)
        median_t = sorted_t[len(sorted_t) // 2]
        outlier_threshold = median_t // 2
        for cam_info in cameras:
            if (
                cam_info.detected
                and cam_info.link_throughput_bytes_per_sec is not None
                and cam_info.link_throughput_bytes_per_sec < outlier_threshold
            ):
                cam_mbps = cam_info.link_throughput_bytes_per_sec * 8 // 1_000_000
                median_mbps = median_t * 8 // 1_000_000
                findings.append(
                    Finding(
                        level="error",
                        code="camera_throughput_outlier",
                        message=(
                            f"Camera {cam_info.serial} link is at ~{cam_mbps} Mbps "
                            f"(others at ~{median_mbps} Mbps) — recordings unusable."
                        ),
                        remediation=[
                            "Unplug and re-plug the camera's ethernet cable at both ends to force PHY renegotiation.",
                            "If the issue returns within a session, swap the cable with a known-good one to rule out cable damage.",
                            "If a fresh cable doesn't help, move the camera to a different switch port.",
                        ],
                        details={
                            "serial": cam_info.serial,
                            "link_throughput_bytes_per_sec": cam_info.link_throughput_bytes_per_sec,
                            "median_throughput_bytes_per_sec": median_t,
                        },
                    )
                )

    if not missing and expected:
        findings.append(
            Finding(
                level="ok",
                code="cameras_ok",
                message=f"All {len(expected)} expected cameras are reachable.",
            )
        )

    if not findings:
        findings.append(
            Finding(
                level="ok",
                code="cameras_ok",
                message="Camera reachability check complete.",
            )
        )

    return CameraReachabilityReport(
        expected=expected,
        detected=detected_serials,
        missing=missing,
        extra=extra,
        cameras=cameras,
        enumerated=enumerated_fresh,
        findings=findings,
    )


# ---------------------------------------------------------------------------
# Host-network check
# ---------------------------------------------------------------------------


def _read_link_state(interface: str, timeout_s: float = 1.0) -> dict[str, Any]:
    """Read interface presence, carrier state, and MTU from /sys/class/net.

    Sysfs reads avoid the iproute2 dependency. With ``network_mode: host`` the
    container's /sys/class/net is the host's. ``timeout_s`` is unused but kept
    so integration tests can monkeypatch with the same signature.
    """
    base = Path("/sys/class/net") / interface
    if not base.exists():
        return {"present": False, "carrier": None, "mtu": None}

    mtu: int | None = None
    try:
        mtu = int((base / "mtu").read_text().strip())
    except (OSError, ValueError):
        pass

    carrier: bool | None = None
    try:
        carrier = (base / "carrier").read_text().strip() == "1"
    except OSError:
        # /sys/class/net/<iface>/carrier is unreadable when the interface is
        # administratively DOWN — for our purposes that's "no carrier."
        carrier = False

    return {"present": True, "carrier": carrier, "mtu": mtu}


def _read_sysctl_int(key: str, timeout_s: float = 1.0) -> int | None:
    """Read a sysctl integer from /proc/sys directly.

    Translates dot-separated keys (e.g. ``net.core.rmem_max``) to procfs paths
    (``/proc/sys/net/core/rmem_max``). Avoids the ``sysctl`` binary dependency.
    """
    path = Path("/proc/sys") / key.replace(".", "/")
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def check_host_network(
    interface: str,
    expected_mtu: int = 9000,
    expected_rmem_max: int = 10_000_000,
) -> HostNetworkStatus:
    """Verify the interface exists, has carrier, jumbo MTU, and adequate rmem_max."""
    findings: list[Finding] = []

    link = _read_link_state(interface)
    interface_present = bool(link.get("present"))
    carrier = link.get("carrier")
    mtu = link.get("mtu")

    if not interface_present:
        findings.append(
            Finding(
                level="error",
                code="host_iface_missing",
                message=(
                    f"Network interface {interface} not found. Plug the camera-network "
                    "ethernet adapter into the laptop, or check NETWORK_INTERFACE in .env."
                ),
                details={"interface": interface},
            )
        )
    else:
        if carrier is False:
            findings.append(
                Finding(
                    level="error",
                    code="host_iface_no_carrier",
                    message=(
                        f"Network interface {interface} has no link. Check the ethernet "
                        "cable from the laptop to the network switch, and verify the switch "
                        "is powered on."
                    ),
                    details={"interface": interface},
                )
            )
        if mtu is not None and mtu != expected_mtu:
            findings.append(
                Finding(
                    level="error",
                    code="host_iface_mtu_mismatch",
                    message=(
                        f"Network interface {interface} MTU is {mtu}, expected "
                        f"{expected_mtu}. Run: sudo ip link set {interface} mtu "
                        f"{expected_mtu}"
                    ),
                    details={
                        "interface": interface,
                        "actual": mtu,
                        "expected": expected_mtu,
                    },
                )
            )

    rmem_max = _read_sysctl_int("net.core.rmem_max")
    if rmem_max is None:
        findings.append(
            Finding(
                level="warn",
                code="host_rmem_unreadable",
                message="Could not read net.core.rmem_max — skipping buffer-size check.",
            )
        )
    elif rmem_max < expected_rmem_max:
        findings.append(
            Finding(
                level="error",
                code="host_rmem_too_low",
                message=(
                    f"Network receive buffer (net.core.rmem_max) is {rmem_max:,} bytes, "
                    f"expected at least {expected_rmem_max:,}. Run: "
                    f"sudo sysctl -w net.core.rmem_max={expected_rmem_max}"
                ),
                details={"actual": rmem_max, "expected": expected_rmem_max},
            )
        )

    if not findings:
        findings.append(
            Finding(
                level="ok",
                code="host_network_ok",
                message=(
                    f"Host network is configured correctly "
                    f"(MTU {mtu}, rmem_max {rmem_max:,})."
                ),
            )
        )

    return HostNetworkStatus(
        interface=interface,
        interface_present=interface_present,
        carrier_up=carrier,
        mtu=mtu,
        expected_mtu=expected_mtu,
        rmem_max=rmem_max,
        expected_rmem_max=expected_rmem_max,
        findings=findings,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_health_check(
    config: HealthConfig,
    expected_serials: list[str],
    recorder: RecorderLike | None = None,
    recording_state: str = "Idle",
    include_unexpected_cameras: bool = False,
    skip_camera_enumeration: bool = False,
    dhcp_runner: SystemctlRunner | None = None,
    ip_addr_runner: IpAddrRunner | None = None,
    camera_enumerator: CameraEnumerator | None = None,
    now: datetime.datetime | None = None,
) -> HealthCheckReport:
    """Run DHCP + camera-reachability + host-network checks concurrently.

    ``include_unexpected_cameras`` controls whether cameras detected on the
    network but not in the current config emit findings. Default False — in
    network-mode deployments other lab setups share the subnet and produce
    routine "extras" that aren't actionable. Operators can opt in via the
    CLI's ``--show-all-cameras`` flag.

    ``skip_camera_enumeration`` short-circuits the PySpin GigE-broadcast
    camera reachability check. Used by the idle poller during an active
    recording — the recorder owns the camera handles, GigE enumeration is
    both slow and thread-unsafe in that state, and per-camera issues are
    surfaced via the recorder's diagnostics_callback instead. DHCP and
    host-network checks still run.
    """
    now = now or datetime.datetime.now(datetime.timezone.utc)

    # Do NOT use `with ThreadPoolExecutor(...)`: the context manager waits for
    # all submitted tasks to finish before exiting, so a hung PySpin call
    # would block the HTTP response even after we've already given up on the
    # future via future.result(timeout=...). Use an explicit non-waiting
    # shutdown so we return as soon as the per-check timeouts fire.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    try:
        dhcp_future = executor.submit(
            check_dhcp_server,
            mode=config.deployment_mode,
            lease_file=config.dhcp_lease_file,
            interface=config.network_interface,
            expected_interface_ip=config.dhcp_expected_interface_ip,
            systemctl_runner=dhcp_runner or _default_systemctl_runner,
            ip_addr_runner=ip_addr_runner or _default_ip_addr_runner,
            now=now,
        )
        camera_future = None
        if not skip_camera_enumeration:
            excluded_serials = (
                getattr(recorder, "excluded_serials", None)
                if recorder is not None
                else None
            )
            camera_future = executor.submit(
                check_camera_reachability,
                expected_serials=expected_serials,
                recorder=recorder,
                minimum_link_speed_mbps=config.minimum_link_speed_mbps,
                include_unexpected=include_unexpected_cameras,
                enumerator=camera_enumerator or _default_camera_enumerator,
                excluded_serials=excluded_serials,
            )
        host_future = executor.submit(
            check_host_network,
            interface=config.network_interface,
        )

        try:
            dhcp = dhcp_future.result(timeout=config.dhcp_timeout_s)
        except concurrent.futures.TimeoutError:
            dhcp = DhcpServerStatus(
                applicable=config.deployment_mode == "laptop",
                lease_file_path=str(config.dhcp_lease_file),
                findings=[
                    Finding(
                        level="warn",
                        code="dhcp_check_timeout",
                        message=f"DHCP check timed out after {config.dhcp_timeout_s}s.",
                    )
                ],
            )

        if camera_future is None:
            cameras = CameraReachabilityReport(
                expected=list(expected_serials),
                detected=list(expected_serials),
                missing=[],
                extra=[],
                cameras=[],
                enumerated=False,
                findings=[],
            )
        else:
            try:
                cameras = camera_future.result(timeout=config.camera_timeout_s)
            except concurrent.futures.TimeoutError:
                cameras = CameraReachabilityReport(
                    expected=list(expected_serials),
                    detected=[],
                    missing=list(expected_serials),
                    extra=[],
                    cameras=[],
                    enumerated=recorder is None,
                    findings=[
                        Finding(
                            level="warn",
                            code="camera_check_timeout",
                            message=(
                                f"Camera reachability check timed out after "
                                f"{config.camera_timeout_s}s."
                            ),
                        )
                    ],
                )

        try:
            host_network = host_future.result(timeout=config.host_network_timeout_s)
        except concurrent.futures.TimeoutError:
            host_network = HostNetworkStatus(
                interface=config.network_interface,
                interface_present=False,
                findings=[
                    Finding(
                        level="warn",
                        code="host_network_check_timeout",
                        message=(
                            f"Host-network check timed out after "
                            f"{config.host_network_timeout_s}s."
                        ),
                    )
                ],
            )
    finally:
        executor.shutdown(wait=False)

    all_findings = (
        list(host_network.findings) + list(dhcp.findings) + list(cameras.findings)
    )
    severities = [host_network.severity, dhcp.severity, cameras.severity]
    overall = max_severity(severities)

    return HealthCheckReport(
        generated_at=now,
        deployment_mode=config.deployment_mode,
        overall=overall,
        dhcp=dhcp,
        cameras=cameras,
        host_network=host_network,
        recording_state=recording_state,
        findings=_prioritize_findings(all_findings),
    )


def _prioritize_findings(findings: list[Finding]) -> list[Finding]:
    """Sort findings by severity (worst first), preserving order within the same level."""
    return sorted(findings, key=lambda f: -_SEVERITY_ORDER[f.level])


# ---------------------------------------------------------------------------
# Background idle poller
# ---------------------------------------------------------------------------


class HealthIdlePoller:
    """Background daemon thread that periodically runs ``run_health_check``.

    Calls ``on_poll(new_report, previous_report)`` after every completed
    check — consumers decide whether to broadcast based on severity changes,
    new findings, etc. The ``run_check`` callable is responsible for
    adapting to recording state (e.g. by passing
    ``skip_camera_enumeration=True`` to ``run_health_check`` while a
    recording is active).
    """

    def __init__(
        self,
        config: HealthConfig,
        run_check: Callable[[], HealthCheckReport],
        on_poll: Callable[[HealthCheckReport, HealthCheckReport | None], None],
        logger: logging.Logger | None = None,
    ):
        self._config = config
        self._run_check = run_check
        self._on_poll = on_poll
        self._logger = logger or logging.getLogger(__name__)

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_report: HealthCheckReport | None = None

    @property
    def last_report(self) -> HealthCheckReport | None:
        return self._last_report

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="health_idle_poller", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self._config.idle_poll_s + 2)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop_event.wait(self._config.idle_poll_s):
            try:
                new_report = self._run_check()
            except Exception as e:  # noqa: BLE001 — log and keep the loop alive
                self._logger.error(f"Idle health check failed: {e}", exc_info=True)
                continue
            previous = self._last_report
            self._last_report = new_report
            try:
                self._on_poll(new_report, previous)
            except Exception as e:  # noqa: BLE001
                self._logger.error(f"on_poll callback raised: {e}", exc_info=True)


def severity_changed(
    new_report: HealthCheckReport, previous: HealthCheckReport | None
) -> bool:
    """Return True iff overall severity differs between new and previous report."""
    if previous is None:
        return new_report.overall != "ok"
    return new_report.overall != previous.overall


# ---------------------------------------------------------------------------
# Auto-remediation
# ---------------------------------------------------------------------------


class RemediationAttempt(BaseModel):
    """One attempted fix for a known-remediable finding."""

    code: str
    success: bool
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class RemediationReport(BaseModel):
    attempts: list[RemediationAttempt] = Field(default_factory=list)

    @property
    def any_attempted(self) -> bool:
        return bool(self.attempts)

    @property
    def all_succeeded(self) -> bool:
        return all(a.success for a in self.attempts)


SudoRunner = Callable[[list[str]], tuple[str, int]]


def _default_sudo_runner(args: list[str], timeout_s: float = 5.0) -> tuple[str, int]:
    """Run a sudo command non-interactively.

    Returns (combined_output, returncode). ``-n`` makes sudo fail rather than
    prompt for a password — the caller should already have passwordless sudo
    set up for the specific commands we issue here.
    """
    cmd = ["sudo", "-n", *args]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s
        )
        return (result.stdout + result.stderr).strip(), result.returncode
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return f"command failed: {e}", -1


def run_host_remediation(
    report: HealthCheckReport,
    config: HealthConfig,
    sudo_runner: SudoRunner = _default_sudo_runner,
) -> RemediationReport:
    """Try to fix known drift before failing.

    For each remediable finding code in ``report``, dispatches to a handler
    that issues a single non-interactive sudo command. Handlers do not
    re-check — the caller should re-run :func:`run_health_check` after this
    to confirm the drift is resolved.

    Currently remediates:
      - ``host_iface_mtu_mismatch`` → ``ip link set <iface> mtu <expected>``
      - ``host_rmem_too_low``       → ``sysctl -w net.core.rmem_max=<expected>``
      - ``dhcp_service_down``       → ``systemctl start isc-dhcp-server``
    """
    attempts: list[RemediationAttempt] = []
    finding_codes = {f.code for f in report.findings}
    host = report.host_network

    if "host_iface_mtu_mismatch" in finding_codes:
        out, rc = sudo_runner(
            [
                "ip",
                "link",
                "set",
                config.network_interface,
                "mtu",
                str(host.expected_mtu),
            ]
        )
        attempts.append(
            RemediationAttempt(
                code="fix_host_iface_mtu",
                success=(rc == 0),
                message=(
                    f"Applied MTU {host.expected_mtu} to {config.network_interface}."
                    if rc == 0
                    else f"Failed to apply MTU: {out.splitlines()[0] if out else 'no output'}"
                ),
                details={
                    "interface": config.network_interface,
                    "output": out,
                    "returncode": rc,
                },
            )
        )

    if "host_rmem_too_low" in finding_codes:
        out, rc = sudo_runner(
            ["sysctl", "-w", f"net.core.rmem_max={host.expected_rmem_max}"]
        )
        attempts.append(
            RemediationAttempt(
                code="fix_host_rmem_max",
                success=(rc == 0),
                message=(
                    f"Applied net.core.rmem_max={host.expected_rmem_max}."
                    if rc == 0
                    else f"Failed to apply sysctl: {out}"
                ),
                details={"output": out, "returncode": rc},
            )
        )

    if "dhcp_service_down" in finding_codes:
        out, rc = sudo_runner(["systemctl", "start", "isc-dhcp-server"])
        attempts.append(
            RemediationAttempt(
                code="start_dhcp_service",
                success=(rc == 0),
                message=(
                    "Started isc-dhcp-server."
                    if rc == 0
                    else f"Failed to start isc-dhcp-server: {out}"
                ),
                details={"output": out, "returncode": rc},
            )
        )

    return RemediationReport(attempts=attempts)


__all__ = [
    "CameraEnumerator",
    "CameraReachability",
    "CameraReachabilityReport",
    "DetectedCamera",
    "DhcpServerStatus",
    "Finding",
    "HealthCheckReport",
    "HealthConfig",
    "HealthIdlePoller",
    "HostNetworkStatus",
    "IpAddrRunner",
    "RecorderLike",
    "RemediationAttempt",
    "RemediationReport",
    "Severity",
    "SudoRunner",
    "SystemctlRunner",
    "check_camera_reachability",
    "check_dhcp_server",
    "check_host_network",
    "max_severity",
    "run_health_check",
    "run_host_remediation",
    "severity_changed",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_finding(f: Finding) -> str:
    sigil = {"ok": "✓", "warn": "!", "error": "✗", "unknown": "?"}[f.level]
    return f"  [{sigil}] {f.message}"


def _list_camera_configs(configs_dir: Path) -> list[Path]:
    """Return YAML/YML files under the camera-configs directory, sorted by name."""
    if not configs_dir.is_dir():
        return []
    return sorted(
        p
        for p in configs_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".yaml", ".yml")
    )


def _serials_from_config(config_path: Path) -> list[str]:
    """Extract camera serials from a camera-config YAML.

    The expected schema (mirrored elsewhere in the acquisition system)::

        camera-info:
          "23015083": { lens_info: ... }
          "23106526": { lens_info: ... }
    """
    import yaml

    data = yaml.safe_load(config_path.read_text()) or {}
    camera_info = data.get("camera-info") or {}
    return [str(serial) for serial in camera_info.keys()]


def _prompt_camera_config(configs_dir: Path) -> list[str]:
    """Interactive picker. Returns selected serials or [] on opt-out / non-TTY."""
    if not sys.stdin.isatty():
        return []

    configs = _list_camera_configs(configs_dir)
    if not configs:
        print(f"No camera configs found in {configs_dir} — skipping camera check.")
        print()
        return []

    print(f"Camera configs in {configs_dir}:")
    print("  [0] Skip — don't run camera reachability check  (default)")
    print("  " + "─" * 58)
    for i, path in enumerate(configs, start=1):
        print(f"   {i:>2}. {path.name}")
    print()

    while True:
        try:
            raw = input("Select a config (number, default 0): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return []
        if raw == "" or raw == "0":
            return []
        try:
            idx = int(raw)
        except ValueError:
            print(f"Not a number: {raw!r}. Try again.")
            continue
        if not (1 <= idx <= len(configs)):
            print(f"Out of range. Pick 0–{len(configs)}.")
            continue
        chosen = configs[idx - 1]
        try:
            serials = _serials_from_config(chosen)
        except Exception as e:  # noqa: BLE001 — yaml errors, missing key, etc.
            print(f"Could not read {chosen.name}: {e}")
            return []
        if not serials:
            print(f"{chosen.name} has no camera-info entries.")
            return []
        print(
            f"Using {chosen.name} ({len(serials)} cameras: "
            f"{', '.join(serials[:3])}{'...' if len(serials) > 3 else ''})"
        )
        print()
        return serials


def _cli_main(argv: list[str] | None = None) -> int:
    """Operator-facing CLI: ``python -m multi_camera.acquisition.health``.

    Runs a single health check against the host's current state, prints a
    prioritized findings list, and exits with a code indicating severity:

      0 — OK
      1 — warnings (transient or non-blocking issues)
      2 — errors (something the operator should act on)
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m multi_camera.acquisition.health",
        description="Quick health check for the camera acquisition host.",
    )
    parser.add_argument(
        "--remediate",
        action="store_true",
        help="Attempt safe auto-remediation (MTU, rmem_max, isc-dhcp-server) "
        "for any remediable findings, then re-check and report.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of human-readable text "
        "(also disables the interactive config picker).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a camera-config YAML to use for the camera-reachability "
        "check. If omitted, the CLI prompts interactively (see --no-config).",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Skip the camera-config picker and run the host/DHCP checks only.",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("/configs"),
        help="Directory holding camera-config YAMLs (default: /configs, "
        "matching the docker-compose volume mount).",
    )
    parser.add_argument(
        "--show-all-cameras",
        action="store_true",
        help="Also list cameras detected on the network that aren't in the "
        "selected config. Off by default — in network-mode deployments other "
        "lab setups produce routine extras that aren't actionable.",
    )
    args = parser.parse_args(argv)

    config = HealthConfig.from_env()

    expected_serials: list[str] = []
    if args.config is not None:
        try:
            expected_serials = _serials_from_config(args.config)
        except Exception as e:  # noqa: BLE001
            print(f"Could not read {args.config}: {e}", file=sys.stderr)
            return 2
    elif not args.no_config and not args.json:
        expected_serials = _prompt_camera_config(args.configs_dir)

    report = run_health_check(
        config=config,
        expected_serials=expected_serials,
        include_unexpected_cameras=args.show_all_cameras,
    )

    remediation: RemediationReport | None = None
    if args.remediate:
        remediation = run_host_remediation(report=report, config=config)
        report = run_health_check(
            config=config,
            expected_serials=expected_serials,
            include_unexpected_cameras=args.show_all_cameras,
        )

    if args.json:
        import json

        payload: dict[str, Any] = {
            "report": report.model_dump(mode="json"),
        }
        if remediation is not None:
            payload["remediation"] = remediation.model_dump(mode="json")
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"Acquisition host health — overall: {report.overall.upper()}")
        print(f"Deployment mode: {config.deployment_mode}")
        print(f"Network interface: {config.network_interface}")
        if remediation is not None and remediation.any_attempted:
            print()
            print("Remediation:")
            for attempt in remediation.attempts:
                sigil = "✓" if attempt.success else "✗"
                print(f"  [{sigil}] {attempt.message}")
        print()
        print("Findings:")
        for finding in report.findings:
            print(_format_finding(finding))

    severity_to_exit = {"ok": 0, "unknown": 0, "warn": 1, "error": 2}
    return severity_to_exit[report.overall]


if __name__ == "__main__":  # pragma: no cover — exercised via subprocess
    sys.exit(_cli_main(sys.argv[1:]))
