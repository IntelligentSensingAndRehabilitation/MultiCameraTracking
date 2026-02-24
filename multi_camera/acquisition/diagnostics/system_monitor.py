"""System-level monitoring for acquisition diagnostics.

Samples NIC statistics, CPU utilization, and disk I/O from Linux procfs.
All sampling functions are pure (path-parameterized) for testability.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path


def read_nic_stats(
    interface: str, sysfs_root: Path = Path("/sys/class/net")
) -> dict[str, int]:
    """Read NIC statistics from /sys/class/net/<interface>/statistics/."""
    stats_dir = sysfs_root / interface / "statistics"
    fields = [
        "rx_dropped",
        "rx_errors",
        "rx_crc_errors",
        "rx_missed_errors",
        "rx_packets",
        "tx_packets",
    ]
    result: dict[str, int] = {}
    for field in fields:
        path = stats_dir / field
        try:
            result[field] = int(path.read_text().strip())
        except (FileNotFoundError, ValueError):
            pass
    return result


def read_cpu_ticks(proc_stat: Path = Path("/proc/stat")) -> dict[str, int]:
    """Parse first line of /proc/stat for CPU tick counters.

    Returns dict with keys: user, nice, system, idle, iowait.
    """
    line = proc_stat.read_text().split("\n")[0]
    parts = line.split()
    keys = ["user", "nice", "system", "idle", "iowait"]
    return {k: int(parts[i + 1]) for i, k in enumerate(keys) if i + 1 < len(parts)}


def compute_cpu_utilization(
    prev: dict[str, int], curr: dict[str, int]
) -> dict[str, float]:
    """Compute CPU utilization percentages between two tick snapshots.

    Returns dict with cpu_percent, iowait_percent.
    """
    prev_total = sum(prev.values())
    curr_total = sum(curr.values())
    delta_total = curr_total - prev_total
    if delta_total == 0:
        return {"cpu_percent": 0.0, "iowait_percent": 0.0}

    delta_idle = curr.get("idle", 0) - prev.get("idle", 0)
    delta_iowait = curr.get("iowait", 0) - prev.get("iowait", 0)

    cpu_percent = 100.0 * (1.0 - delta_idle / delta_total)
    iowait_percent = 100.0 * (delta_iowait / delta_total)
    return {
        "cpu_percent": round(cpu_percent, 1),
        "iowait_percent": round(iowait_percent, 1),
    }


def read_diskstats(
    device: str, proc_diskstats: Path = Path("/proc/diskstats")
) -> dict[str, int]:
    """Read sectors read/written for a specific block device from /proc/diskstats.

    Returns dict with read_sectors, write_sectors.
    """
    try:
        text = proc_diskstats.read_text()
    except FileNotFoundError:
        return {}

    for line in text.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 10 and parts[2] == device:
            return {"read_sectors": int(parts[5]), "write_sectors": int(parts[9])}
    return {}


def compute_disk_delta(prev: dict[str, int], curr: dict[str, int]) -> dict[str, int]:
    """Compute sector deltas between two diskstats snapshots."""
    result: dict[str, int] = {}
    for key in ["read_sectors", "write_sectors"]:
        if key in curr and key in prev:
            result[key] = curr[key] - prev[key]
    return result


def detect_network_interface(camera_ip: str | None = None) -> str | None:
    """Detect the network interface cameras are connected to.

    If camera_ip is provided, tries to match it against interface subnets.
    Otherwise returns the first non-loopback interface found in /sys/class/net/.
    """
    sysfs = Path("/sys/class/net")
    if not sysfs.exists():
        return None

    candidates = [d.name for d in sysfs.iterdir() if d.is_dir() and d.name != "lo"]
    if not candidates:
        return None

    if camera_ip is not None:
        import ipaddress

        target = ipaddress.ip_address(camera_ip)
        for iface in candidates:
            try:
                operstate = (sysfs / iface / "operstate").read_text().strip()
                if operstate != "up":
                    continue
            except FileNotFoundError:
                continue
            # Check if the interface has an IP in the same /24 subnet
            try:
                import subprocess

                result = subprocess.run(
                    ["ip", "-4", "addr", "show", iface],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                for line in result.stdout.split("\n"):
                    line = line.strip()
                    if line.startswith("inet "):
                        net = ipaddress.ip_network(line.split()[1], strict=False)
                        if target in net:
                            return iface
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

    # Fallback: return first non-loopback interface that is up
    for iface in candidates:
        try:
            operstate = (sysfs / iface / "operstate").read_text().strip()
            if operstate == "up":
                return iface
        except FileNotFoundError:
            continue

    return candidates[0] if candidates else None


class SystemMonitor:
    """Background thread that samples system metrics at a fixed interval.

    Start/stop around a recording. Snapshots are accumulated in-memory
    and retrieved via get_snapshots() after stopping.
    """

    def __init__(
        self,
        interface: str,
        disk_device: str | None = None,
        interval_s: float = 10.0,
        sysfs_root: Path = Path("/sys/class/net"),
        proc_stat: Path = Path("/proc/stat"),
        proc_diskstats: Path = Path("/proc/diskstats"),
    ):
        self.interface = interface
        self.disk_device = disk_device
        self.interval_s = interval_s
        self.sysfs_root = sysfs_root
        self.proc_stat = proc_stat
        self.proc_diskstats = proc_diskstats

        self._snapshots: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._prev_cpu: dict[str, int] | None = None
        self._prev_disk: dict[str, int] | None = None

    def _take_snapshot(self) -> dict:
        snapshot: dict = {"wall_clock": time.monotonic()}

        nic = read_nic_stats(self.interface, self.sysfs_root)
        if nic:
            snapshot["nic"] = nic

        curr_cpu = read_cpu_ticks(self.proc_stat)
        if curr_cpu and self._prev_cpu is not None:
            snapshot["cpu"] = compute_cpu_utilization(self._prev_cpu, curr_cpu)
        self._prev_cpu = curr_cpu

        if self.disk_device:
            curr_disk = read_diskstats(self.disk_device, self.proc_diskstats)
            if curr_disk and self._prev_disk is not None:
                snapshot["disk"] = compute_disk_delta(self._prev_disk, curr_disk)
            self._prev_disk = curr_disk

        return snapshot

    def _loop(self) -> None:
        # Take an initial reading for delta baselines
        self._prev_cpu = read_cpu_ticks(self.proc_stat)
        if self.disk_device:
            self._prev_disk = read_diskstats(self.disk_device, self.proc_diskstats)

        while not self._stop_event.wait(self.interval_s):
            self._snapshots.append(self._take_snapshot())

    def start(self) -> None:
        if self._thread is not None:
            return
        self._snapshots = []
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="system_monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self.interval_s + 2)
        self._thread = None

    def get_snapshots(self) -> list[dict]:
        return list(self._snapshots)

    def get_latest_snapshot(self) -> dict | None:
        """Return the most recent snapshot, or None if no snapshots yet."""
        return self._snapshots[-1] if self._snapshots else None
