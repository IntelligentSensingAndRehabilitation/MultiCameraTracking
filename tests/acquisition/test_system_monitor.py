"""Tests for system_monitor.py — procfs sampling and SystemMonitor thread.

All tests use synthetic procfs data via tmp_path — no real hardware needed.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from multi_camera.acquisition.diagnostics.system_monitor import (
    SystemMonitor,
    compute_cpu_utilization,
    compute_disk_delta,
    read_cpu_ticks,
    read_diskstats,
    read_nic_stats,
)


class TestReadNicStats:
    def test_reads_all_fields(self, tmp_path: Path) -> None:
        stats_dir = tmp_path / "eth0" / "statistics"
        stats_dir.mkdir(parents=True)
        (stats_dir / "rx_dropped").write_text("42\n")
        (stats_dir / "rx_errors").write_text("7\n")
        (stats_dir / "rx_crc_errors").write_text("0\n")
        (stats_dir / "rx_missed_errors").write_text("3\n")
        (stats_dir / "rx_packets").write_text("100000\n")
        (stats_dir / "tx_packets").write_text("50000\n")

        result = read_nic_stats("eth0", sysfs_root=tmp_path)

        assert result == {
            "rx_dropped": 42,
            "rx_errors": 7,
            "rx_crc_errors": 0,
            "rx_missed_errors": 3,
            "rx_packets": 100000,
            "tx_packets": 50000,
        }

    def test_missing_interface_returns_empty(self, tmp_path: Path) -> None:
        result = read_nic_stats("nonexistent", sysfs_root=tmp_path)
        assert result == {}

    def test_partial_fields(self, tmp_path: Path) -> None:
        stats_dir = tmp_path / "eth0" / "statistics"
        stats_dir.mkdir(parents=True)
        (stats_dir / "rx_dropped").write_text("10\n")

        result = read_nic_stats("eth0", sysfs_root=tmp_path)
        assert result == {"rx_dropped": 10}


class TestReadCpuTicks:
    def test_parses_proc_stat(self, tmp_path: Path) -> None:
        proc_stat = tmp_path / "stat"
        proc_stat.write_text(
            "cpu  10000 500 3000 80000 1500 0 200 0 0 0\ncpu0  5000 250\n"
        )

        result = read_cpu_ticks(proc_stat=proc_stat)

        assert result == {
            "user": 10000,
            "nice": 500,
            "system": 3000,
            "idle": 80000,
            "iowait": 1500,
        }


class TestComputeCpuUtilization:
    def test_basic_utilization(self) -> None:
        prev = {"user": 100, "nice": 0, "system": 50, "idle": 800, "iowait": 50}
        curr = {"user": 200, "nice": 0, "system": 100, "idle": 1600, "iowait": 100}

        result = compute_cpu_utilization(prev, curr)

        expected_cpu = 100.0 * (1.0 - 800 / 1000)
        assert result["cpu_percent"] == pytest.approx(expected_cpu, abs=0.1)
        assert result["iowait_percent"] == pytest.approx(5.0, abs=0.1)

    def test_zero_delta(self) -> None:
        ticks = {"user": 100, "nice": 0, "system": 50, "idle": 800, "iowait": 50}
        result = compute_cpu_utilization(ticks, ticks)
        assert result["cpu_percent"] == 0.0
        assert result["iowait_percent"] == 0.0


class TestReadDiskstats:
    def test_parses_matching_device(self, tmp_path: Path) -> None:
        diskstats = tmp_path / "diskstats"
        diskstats.write_text(
            "   8       0 sda 1000 0 20000 500 2000 0 40000 1000 0 800 1500 0 0 0 0 0 0\n"
            "   8       1 sda1 900 0 18000 400 1800 0 36000 900 0 700 1300 0 0 0 0 0 0\n"
        )

        result = read_diskstats("sda", proc_diskstats=diskstats)
        assert result == {"read_sectors": 20000, "write_sectors": 40000}

    def test_nonexistent_device(self, tmp_path: Path) -> None:
        diskstats = tmp_path / "diskstats"
        diskstats.write_text(
            "   8       0 sda 1000 0 20000 500 2000 0 40000 1000 0 800 1500\n"
        )

        result = read_diskstats("nvme0n1", proc_diskstats=diskstats)
        assert result == {}

    def test_missing_file(self, tmp_path: Path) -> None:
        result = read_diskstats("sda", proc_diskstats=tmp_path / "nonexistent")
        assert result == {}


class TestComputeDiskDelta:
    def test_basic_delta(self) -> None:
        prev = {"read_sectors": 1000, "write_sectors": 2000}
        curr = {"read_sectors": 1500, "write_sectors": 3000}
        result = compute_disk_delta(prev, curr)
        assert result == {"read_sectors": 500, "write_sectors": 1000}

    def test_empty_inputs(self) -> None:
        assert compute_disk_delta({}, {}) == {}


class TestSystemMonitor:
    def _setup_fake_procfs(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        sysfs = tmp_path / "sysfs"
        stats_dir = sysfs / "eth0" / "statistics"
        stats_dir.mkdir(parents=True)
        for field in [
            "rx_dropped",
            "rx_errors",
            "rx_crc_errors",
            "rx_missed_errors",
            "rx_packets",
            "tx_packets",
        ]:
            (stats_dir / field).write_text("0\n")

        proc_stat = tmp_path / "proc_stat"
        proc_stat.write_text("cpu  10000 500 3000 80000 1500 0 200 0 0 0\n")

        proc_diskstats = tmp_path / "proc_diskstats"
        proc_diskstats.write_text(
            "   8       0 sda 1000 0 20000 500 2000 0 40000 1000 0 800 1500\n"
        )

        return sysfs, proc_stat, proc_diskstats

    def test_start_stop_collects_snapshots(self, tmp_path: Path) -> None:
        sysfs, proc_stat, proc_diskstats = self._setup_fake_procfs(tmp_path)

        monitor = SystemMonitor(
            interface="eth0",
            disk_device="sda",
            interval_s=0.1,
            sysfs_root=sysfs,
            proc_stat=proc_stat,
            proc_diskstats=proc_diskstats,
        )

        monitor.start()
        time.sleep(0.5)
        monitor.stop()

        snapshots = monitor.get_snapshots()
        assert len(snapshots) >= 2
        assert "wall_clock" in snapshots[0]
        assert "nic" in snapshots[0]

    def test_cpu_utilization_computed_after_second_sample(self, tmp_path: Path) -> None:
        sysfs, proc_stat, proc_diskstats = self._setup_fake_procfs(tmp_path)

        monitor = SystemMonitor(
            interface="eth0",
            interval_s=0.1,
            sysfs_root=sysfs,
            proc_stat=proc_stat,
            proc_diskstats=proc_diskstats,
        )

        monitor.start()
        time.sleep(0.4)
        monitor.stop()

        snapshots = monitor.get_snapshots()
        assert len(snapshots) >= 1
        # First snapshot after baseline should have cpu data (since ticks didn't change, it'll be 0%)
        assert "cpu" in snapshots[0]
        assert "cpu_percent" in snapshots[0]["cpu"]

    def test_get_latest_snapshot(self, tmp_path: Path) -> None:
        sysfs, proc_stat, proc_diskstats = self._setup_fake_procfs(tmp_path)

        monitor = SystemMonitor(
            interface="eth0",
            interval_s=0.1,
            sysfs_root=sysfs,
            proc_stat=proc_stat,
            proc_diskstats=proc_diskstats,
        )

        assert monitor.get_latest_snapshot() is None

        monitor.start()
        time.sleep(0.3)
        monitor.stop()

        latest = monitor.get_latest_snapshot()
        assert latest is not None
        assert "wall_clock" in latest

    def test_idempotent_start_stop(self, tmp_path: Path) -> None:
        sysfs, proc_stat, proc_diskstats = self._setup_fake_procfs(tmp_path)

        monitor = SystemMonitor(
            interface="eth0",
            interval_s=0.1,
            sysfs_root=sysfs,
            proc_stat=proc_stat,
            proc_diskstats=proc_diskstats,
        )

        monitor.start()
        monitor.start()
        time.sleep(0.2)
        monitor.stop()
        monitor.stop()
        assert len(monitor.get_snapshots()) >= 1
