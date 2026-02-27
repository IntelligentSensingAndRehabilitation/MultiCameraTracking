# Acquisition Diagnostics

This page covers two diagnostic systems:

1. **System diagnostics** (`acquisition_diagnostics.sh`) — pre-recording system validation (network, DHCP, cameras, hardware)
2. **Sync diagnostics** (`json_parser.py` + `flir_recording_api.py` instrumentation) — per-frame sync quality analysis during and after recording

## Sync Diagnostics

Sync diagnostics analyze timestamp synchronization, frame ID alignment, and camera health across multi-camera recordings.

### Pre-Recording Validation

Run a short acquisition burst to check sync quality before a real recording:

```bash
make validate-sync CONFIG=/configs/your_config.yaml
```

This acquires 100 frames (no video/JSON written) and reports whether any frames exceeded the timespread threshold.

### Diagnostic Recording

Record with full instrumentation enabled:

```bash
make diag-recording CONFIG=/configs/your_config.yaml
make diag-recording CONFIG=/configs/your_config.yaml FRAMES=300
```

Output goes to `/data/diagnostics/YYYYMMDD/`. JSON metadata includes per-frame sync wait cycles, queue depths, frame ID cross-camera deltas, bottleneck cameras, PTP offsets, camera temperatures, and system monitor snapshots.

### Post-Hoc Analysis

Analyze JSON metadata from any recording (diagnostic or normal with `diagnostics_level >= 1`):

```bash
make diag-analyze DATA=/data/diagnostics/20260226
# Or directly:
python -m multi_camera.acquisition.diagnostics.json_parser /path/to/json/dir
```

This produces:
- **Terminal report**: per-trial summary, aggregate statistics, acquisition diagnostics, and diagnostic insights
- **Interactive HTML plots** (unless `--no-plots`): timestamp deltas, frame ID deltas, sync wait cycles, queue depths, camera errors

### What It Detects

The `diagnose_sync_issues()` function runs 17 detectors:

| Detector | What it catches |
|----------|----------------|
| Frame-period spikes | Max dT ≈ 1 frame period → alignment error, not clock drift |
| Repeat offender cameras | Same camera drifting across multiple trials |
| Burst patterns | 3+ consecutive trials with frame ID drift |
| Reference camera jumps | All delta cameras shift simultaneously → reference camera skipped |
| Sync bottleneck | One camera disproportionately slow at sync barrier |
| Queue depth spikes | Metadata thread falling behind acquisition |
| Frame ID misalignment | Persistent cross-camera frame ID disagreement |
| Per-camera error rates | Incomplete frames, exceptions, frame ID gaps |
| Stream stats | Dropped frames and queue overflows at device/network level |
| PTP offset drift | Significant PTP offset change during recording |
| Temperature rise | Camera temperature increase > 5°C during recording |
| Sync timeouts | Frames where sync barrier waited > threshold |
| Timespread alerts | Frames exceeding timespread threshold |
| NIC rx_dropped | Network interface packet drops during recording |
| Frame skip events | Camera frame ID gaps with placeholder recovery status |
| PTP timestamp jumps | Per-camera clock discontinuities with pair detection and usability assessment |

### Frame Skip Recovery

When a camera drops frames (frame ID gap), the acquisition system inserts black placeholder frames into both the video and metadata queues to maintain sync alignment. This is controlled by `frame_skip_recovery` (default: on). Skip events are logged in JSON metadata and surfaced in the post-hoc report.

### Diagnostics Level

The `diagnostics_level` parameter (default 1) controls instrumentation depth:

- **Level 0**: Per-frame sync wait cycles, queue depths, frame ID cross-deltas, bottleneck cameras, camera error counters, stream stats
- **Level 1**: All of level 0, plus PTP offset sampling, camera temperature monitoring, system monitor (NIC/CPU/disk), timespread alerts, sync timeout detection

Set via the FastAPI `/new_trial` endpoint or `FlirRecorder.start_acquisition(diagnostics_level=...)`.

---

## System Diagnostics

The system diagnostics script is a comprehensive troubleshooting tool that collects system information, validates configuration, and identifies issues with the acquisition system.

## Usage

From the root of the MultiCameraTracking repository:

```bash
sudo ./scripts/acquisition/acquisition_diagnostics.sh
```

### Command Line Options

**Verbose mode** - Show additional detailed output:

```bash
sudo ./scripts/acquisition/acquisition_diagnostics.sh --verbose
```

**Quick mode** - Run only critical checks (network, MTU, DHCP):

```bash
sudo ./scripts/acquisition/acquisition_diagnostics.sh --quick
```

Quick mode is useful for fast pre-recording validation and skips:
- Camera detection (section 5)
- System logs collection (section 6)
- Recent recordings check (section 7)
- Hardware information (section 8)
- DataJoint connectivity test (section 9)
- Tarball generation

Running with sudo is recommended to access all system information.

### Deployment Mode Awareness

The diagnostics script automatically detects the `DEPLOYMENT_MODE` from your `.env` file:

- **Laptop mode**: Runs all DHCP server checks (section 4)
- **Network mode**: Skips DHCP server checks since they're not needed when using building network infrastructure

This ensures you only see relevant diagnostics for your deployment configuration.

## What It Checks

The diagnostics script performs comprehensive system validation across 9 sections:

### 1. System Information

- Operating system version (checks for Ubuntu 22.04)
- Kernel version
- Current timestamp

### 2. Environment Configuration

- Verifies `.env` file exists and is properly configured
- Checks all required environment variables are set
- Validates disk space on DATA_VOLUME
- Confirms NETWORK_INTERFACE is defined

### 3. Network Configuration

- Verifies network interface exists and is UP
- Checks MTU is set to 9000 (jumbo frames)
- Validates IP address (expects 192.168.1.1 for laptop mode)
- Confirms NetworkManager DHCP-Server profile is active
- Checks network buffer settings (rmem_max, rmem_default)

### 4. DHCP Server Status

- Checks deployment mode from `.env` file
- In laptop mode:
  - Verifies isc-dhcp-server is running
  - Shows recent DHCP logs
  - Lists active DHCP leases
  - Displays connected camera IP addresses
- In network mode:
  - Skips all DHCP checks (not required when using building network infrastructure)

### 5. Camera Detection

- Uses Docker to detect cameras on the network
- Shows which cameras are available vs. in-use
- Validates camera configuration files exist
- Checks config files match detected cameras

### 6. System Logs

- Extracts relevant system logs (dmesg, journalctl)
- Filters for network, USB, camera, and error messages
- Saves full logs to tarball for detailed analysis

### 7. Recent Recordings

- Finds the most recent recording directory
- Validates video files were created
- Checks file sizes (detects suspiciously small files)
- Analyzes JSON metadata for timestamp synchronization quality

### 8. Hardware Information

- Lists PCI Ethernet and Thunderbolt adapters
- Shows detailed adapter information (ethtool)
- Identifies 10GbE network cards

### 9. DataJoint Database Connectivity

- Tests connection to DataJoint database
- Measures connection latency
- Warns if connection is slow (> 2 seconds)

## Output

The script generates two files in `./diagnostics_output/`:

1. **Log file**: `acquisition_diagnostics_YYYYMMDD_HHMMSS.log`
   - Human-readable diagnostic report
   - Color-coded output (errors, warnings, success)
   - Summary of issues found

2. **Tarball**: `acquisition_diagnostics_YYYYMMDD_HHMMSS.tar.gz`
   - Contains the log file
   - Full system logs (dmesg, journalctl)
   - Useful for sharing with support

### Common Issues and Fixes

For detailed solutions to all issues, see the [Troubleshooting Guide](troubleshooting.md):

- [Environment Configuration Issues](troubleshooting.md#environment-configuration-issues)
- [Network Interface Issues](troubleshooting.md#network-interface-issues)
- [MTU Configuration Issues](troubleshooting.md#mtu-configuration-issues)
- [Network Buffer Issues](troubleshooting.md#network-buffer-issues)
- [DHCP Server Issues](troubleshooting.md#dhcp-server-issues)
- [Camera Detection Issues](troubleshooting.md#camera-detection-issues)
- [DataJoint Connectivity Issues](troubleshooting.md#datajoint-connectivity-issues)