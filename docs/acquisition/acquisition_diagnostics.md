# Acquisition Diagnostics

The acquisition diagnostics script is a comprehensive troubleshooting tool that collects system information, validates configuration, and identifies issues with the acquisition system.

## When to Use

Use the diagnostics script when:
- Setting up the acquisition system for the first time
- Troubleshooting issues with camera detection or recording
- Performance problems (dropped frames, slow recording)
- Network connectivity issues
- Before reporting bugs or requesting support

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

**Combined options**:

```bash
sudo ./scripts/acquisition/acquisition_diagnostics.sh --quick --verbose
```

Running with sudo is recommended to access all system information, though the script will run without it (with limited diagnostics).

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

## Interpreting Results

### Color Coding

- **Green [OK]**: Check passed, no issues
- **Yellow [WARNING]**: Non-critical issue, review recommended
- **Red [ERROR]**: Critical issue that must be fixed

### Issue Summary

At the end, the script displays:
- Total number of errors found
- Total number of warnings found
- Recommendations for fixes

### Common Issues and Fixes

For detailed solutions to all issues, see the [Troubleshooting Guide](troubleshooting.md):

- [Environment Configuration Issues](troubleshooting.md#environment-configuration-issues)
- [Network Interface Issues](troubleshooting.md#network-interface-issues)
- [MTU Configuration Issues](troubleshooting.md#mtu-configuration-issues)
- [Network Buffer Issues](troubleshooting.md#network-buffer-issues)
- [DHCP Server Issues](troubleshooting.md#dhcp-server-issues)
- [Camera Detection Issues](troubleshooting.md#camera-detection-issues)
- [DataJoint Connectivity Issues](troubleshooting.md#datajoint-connectivity-issues)

## Command Line Options

### Verbose Mode

```bash
sudo ./scripts/acquisition/acquisition_diagnostics.sh --verbose
# or short form:
sudo ./scripts/acquisition/acquisition_diagnostics.sh -v
```

Shows additional information:
- Full DHCP lease file contents
- Complete camera configuration files
- Detailed network logs

### Quick Mode

```bash
sudo ./scripts/acquisition/acquisition_diagnostics.sh --quick
# or short form:
sudo ./scripts/acquisition/acquisition_diagnostics.sh -q
```

Runs only critical checks for fast validation:
- System information (section 1)
- Environment configuration (section 2)
- Network configuration (section 3)
- DHCP server status (section 4, if laptop mode)

Skips camera detection, log collection, recordings check, hardware info, and DataJoint connectivity.

### Help

```bash
./scripts/acquisition/acquisition_diagnostics.sh --help
```

Shows usage information (invalid arguments will also display usage).

## Use Cases

### System Checks Before Recording

For a fast validation before recording, use quick mode:

```bash
sudo ./scripts/acquisition/acquisition_diagnostics.sh --quick
```

This runs in seconds and validates:
- 0 errors found
- Network is configured (MTU 9000, IP 192.168.1.1 in laptop mode)
- DHCP server running (laptop mode only)
- Sufficient disk space

For comprehensive pre-recording checks including camera detection:

```bash
sudo ./scripts/acquisition/acquisition_diagnostics.sh
```

The system is ready when:
- 0 errors found
- Network is configured (MTU 9000, IP 192.168.1.1 in laptop mode)
- DHCP server running (laptop mode only)
- Cameras detected
- Sufficient disk space

### Troubleshooting Camera Issues

If cameras aren't being detected:

1. Run diagnostics in verbose mode:
   ```bash
   sudo ./scripts/acquisition/acquisition_diagnostics.sh --verbose
   ```

2. Check section 4 (DHCP Server Status):
   - Are DHCP leases being issued?
   - Do IP addresses appear in expected range (192.168.1.10-100)?

3. Check section 5 (Camera Detection):
   - Does Docker camera detection script run successfully?
   - Are camera serial numbers listed?

4. Check section 6 (System Logs):
   - Look for USB errors or disconnection messages
   - Check for network interface errors

### Performance Issues

If experiencing dropped frames or slow recording:

1. Run diagnostics and review:
   - Section 3: Is MTU set to 9000?
   - Section 3: Are network buffers configured properly?
   - Section 8: Is the correct network adapter being used (10GbE)?

2. Check recent recordings (Section 7):
   - Look at timestamp synchronization quality
   - Check if video file sizes are consistent

3. Review system logs (Section 6):
   - Look for buffer overflow warnings
   - Check for network packet drop messages

### Preparing Bug Reports

When reporting issues to support:

1. Run diagnostics:
   ```bash
   sudo ./scripts/acquisition/acquisition_diagnostics.sh --verbose
   ```

2. Share the generated tarball:
   ```
   ./diagnostics_output/acquisition_diagnostics_YYYYMMDD_HHMMSS.tar.gz
   ```

The tarball contains all necessary information for debugging without exposing sensitive data from `.env`.

## Integration with Startup Script

The startup script performs similar checks but is optimized for quick validation. The diagnostics script is more comprehensive and designed for troubleshooting.

**When to use startup script:**
- Regular daily operation
- Quick system validation before starting
- You just want to start acquisition

**When to use diagnostics script:**
- Detailed troubleshooting needed
- First-time setup validation
- Preparing for support
- Investigating performance issues

## Automation

You can automate diagnostics collection on a schedule:

```bash
# Add to crontab (run daily at 2 AM)
0 2 * * * cd /path/to/MultiCameraTracking && sudo ./scripts/acquisition/acquisition_diagnostics.sh
```

This creates daily diagnostic snapshots for historical troubleshooting.

## Technical Details

### Requirements

- Must be run from MultiCameraTracking repository root
- Requires sudo for complete diagnostics
- Checks for `docker-compose.yml` to verify correct directory

### Dependencies

- bash
- Docker (for camera detection)
- Standard Linux utilities (ip, sysctl, ethtool, etc.)

### Exit Codes

- `0`: Diagnostics completed successfully (may still contain errors/warnings)
- `1`: Failed to run (wrong directory, missing dependencies)

The script does not fail based on diagnostic results - it always completes and reports findings.
