# Unified Startup Script

The unified startup script simplifies the acquisition system startup process by automating system checks, network activation, and system validation.

## Usage

From the root of the MultiCameraTracking repository:

```bash
make run
```

This command runs the startup script with full system validation (recommended).

## What the Script Does

### 1. System Checks

The script validates your configuration before starting:

- **Environment file**: Checks that `.env` exists and contains required variables
- **Network interface**: Verifies the configured interface exists
- **MTU setting**: Confirms MTU is set to 9000 for optimal performance
- **Network buffers**: Checks that receive buffers are properly configured
- **Disk space**: Ensures sufficient free space for recordings
- **Camera configs**: Verifies camera configuration directory exists

If any check fails, the script will display an error message and stop.

### 2. Network Activation (Laptop Mode Only)

In laptop mode, the script will:
- Activate the DHCP-Server NetworkManager profile
- Verify the network interface receives the correct IP (192.168.1.1)
- Check that the DHCP server is running

In network mode, this step is skipped.

### 3. Camera Detection

The script waits up to 30 seconds for cameras to connect and show up on the network. This is optional and non-blocking.

### 4. Start Acquisition Software

If all checks pass, the script launches the acquisition Docker container.

## Command Line Options

### Skip System Checks (Quick Start)

```bash
make run-no-checks
```

Or directly:
```bash
./scripts/acquisition/start_acquisition.sh --skip-checks
```

Bypasses all validation checks and starts immediately. Useful for quick restarts when you know your system is already configured correctly.

### Help

```bash
./scripts/acquisition/start_acquisition.sh --help
```

Shows usage information and available options.

## Troubleshooting

For detailed solutions to common issues, see the [Troubleshooting Guide](troubleshooting.md):

- [Environment Configuration Issues](troubleshooting.md#environment-configuration-issues)
- [Network Interface Issues](troubleshooting.md#network-interface-issues)
- [MTU Configuration Issues](troubleshooting.md#mtu-configuration-issues)
- [Network Buffer Issues](troubleshooting.md#network-buffer-issues)
- [DHCP Server Issues](troubleshooting.md#dhcp-server-issues)
- [Camera Configuration Issues](troubleshooting.md#camera-configuration-issues)
- [Disk Space Issues](troubleshooting.md#disk-space-issues)

## Deployment Modes

The script adapts its behavior based on the `DEPLOYMENT_MODE` variable in `.env`:

**laptop**:
- Activates DHCP-Server network profile
- Checks DHCP server status
- Waits for cameras to get DHCP leases

**network**:
- Skips DHCP-related checks
- Assumes network infrastructure handles DHCP
- Only validates MTU and buffer settings

## Exit Codes

- `0`: Success - acquisition system started
- `1`: System checks failed or network activation failed

## Available Make Commands

The Makefile provides convenient shortcuts:

- **`make run`**: Start with full system validation (recommended)
- **`make run-no-checks`**: Quick start without validation checks
- **`make build-mocap`**: Build the acquisition Docker image

## Logs

The startup script outputs to stdout/stderr. For more detailed diagnostics, use the diagnostics script:

```bash
./scripts/acquisition/acquisition_diagnostics.sh
```
