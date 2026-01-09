# Unified Startup Script

The unified startup script simplifies the acquisition system startup process by automating system checks, network activation, and system validation.

## Usage

From the root of the MultiCameraTracking repository:

```bash
./scripts/acquisition/start_acquisition.sh
```

Or using the Makefile:

```bash
make start-acquisition
```

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

If all checks pass, the script launches the acquisition system using `make run`.

## Command Line Options

### Skip System Checks

```bash
./scripts/acquisition/start_acquisition.sh --skip-checks
```

Bypasses all validation checks and starts immediately. Not recommended unless you're debugging the checks themselves.

### Help

```bash
./scripts/acquisition/start_acquisition.sh --help
```

Shows usage information and available options.

## Troubleshooting

### Error: .env file not found

Create `.env` from `.env.template` and fill in the required values. See [Acquisition Software Setup](acquisition_software_setup.md).

### Error: Network interface not found

The `NETWORK_INTERFACE` in your `.env` file doesn't match any physical interface.

Check available interfaces:
```bash
ip link show
```

Update `NETWORK_INTERFACE` in `.env` to match your actual interface name.

### Warning: MTU is not 9000

Run the persistence script to make MTU settings permanent:
```bash
./scripts/acquisition/make_settings_persistent.sh
```

Or set it manually for this session:
```bash
sudo sh scripts/acquisition/set_mtu.sh
```

### Warning: Network buffers not configured

Run the persistence script to configure buffers permanently:
```bash
./scripts/acquisition/make_settings_persistent.sh
```

### Error: Low disk space

Free up space in your `DATA_VOLUME` directory. The threshold is set by `DISK_SPACE_WARNING_THRESHOLD_GB` in your `.env` file (default: 50GB).

Check disk usage:
```bash
df -h /data  # or your DATA_VOLUME path
```

### Error: DHCP-Server profile not found

Complete the DHCP setup first. See [DHCP Server Setup](dhcp_setup.md).

### Warning: DHCP server not running

If you've run the persistence script, the DHCP server should start automatically on boot.

Start it manually:
```bash
sudo systemctl start isc-dhcp-server
```

Check status:
```bash
sudo systemctl status isc-dhcp-server
```

### Warning: No camera config files found

Create a camera configuration YAML file in your `CAMERA_CONFIGS` directory. See [Example Config](example_config.md).

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

## Integration with Make

A Makefile target is available for convenience:

```bash
make start-acquisition
```

This is equivalent to running the startup script directly.

## Logs

The startup script outputs to stdout/stderr. For more detailed diagnostics, use the diagnostics script:

```bash
./scripts/acquisition/acquisition_diagnostics.sh
```
