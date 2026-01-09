# Acquisition System Troubleshooting Guide

This guide provides solutions to common issues with the acquisition system. Each section can be referenced directly by other documentation.

## Table of Contents

- [Environment Configuration Issues](#environment-configuration-issues)
- [Network Interface Issues](#network-interface-issues)
- [MTU Configuration Issues](#mtu-configuration-issues)
- [Network Buffer Issues](#network-buffer-issues)
- [DHCP Server Issues](#dhcp-server-issues)
- [Docker Issues](#docker-issues)
- [Camera Detection Issues](#camera-detection-issues)
- [Disk Space Issues](#disk-space-issues)
- [DataJoint Connectivity Issues](#datajoint-connectivity-issues)
- [Camera Configuration Issues](#camera-configuration-issues)
- [Performance Issues](#performance-issues)

---

## Environment Configuration Issues

### Error: .env file not found

**Cause**: The `.env` file hasn't been created yet.

**Solution**:
1. Copy the template file:
   ```bash
   cp .env.template .env
   ```
2. Edit `.env` and fill in the required values
3. See [Acquisition Software Setup](acquisition_software_setup.md) for details

### Error: Missing required environment variables

**Cause**: Your `.env` file is incomplete.

**Solution**:
1. Compare your `.env` file with `.env.template`
2. Ensure all required variables are set:
   - `DJ_USER`, `DJ_PASS`, `DJ_HOST`, `DJ_PORT`
   - `NETWORK_INTERFACE`
   - `DEPLOYMENT_MODE`
   - `DATA_VOLUME`
   - `CAMERA_CONFIGS`

---

## Network Interface Issues

### Error: Network interface not found

**Cause**: The `NETWORK_INTERFACE` in your `.env` file doesn't match any physical interface.

**Solution**:
1. Check available interfaces:
   ```bash
   ip link show
   ```
2. Look for interface names like `enp5s0`, `enp37s0`, etc.
3. Update `NETWORK_INTERFACE` in `.env` to match your actual interface name

### How to identify the correct network interface

If you have multiple ethernet interfaces:

1. Check the MAC address if you know it:
   ```bash
   ip link show
   ```
   Match the MAC address to the interface name

2. Physical test method:
   ```bash
   watch -n 1 ip link show
   ```
   Unplug and replug the cable - watch which interface appears/disappears

---

## MTU Configuration Issues

### Error/Warning: MTU is not 9000

**Cause**: MTU (Maximum Transmission Unit) isn't set to 9000, which is required for jumbo frames and optimal camera streaming performance.

**Solution Option 1 - Make it persistent** (recommended):
```bash
./scripts/acquisition/make_settings_persistent.sh
```

**Solution Option 2 - Set manually for current session**:
```bash
sudo sh scripts/acquisition/set_mtu.sh
```

### Verify MTU setting

**For laptop mode**:
```bash
nmcli con show DHCP-Server | grep mtu
```

**For network mode**:
```bash
ip link show <interface-name>
```

Look for `mtu 9000` in the output.

### MTU not persisting across reboots

**Cause**: NetworkManager settings haven't been configured.

**Solution**: Run the persistence script:
```bash
./scripts/acquisition/make_settings_persistent.sh
```

---

## Network Buffer Issues

### Warning: Network buffers not configured

**Cause**: Kernel receive buffer settings aren't optimized for high-bandwidth video streams.

**Solution**: Run the persistence script:
```bash
./scripts/acquisition/make_settings_persistent.sh
```

### Verify buffer settings

```bash
sysctl net.core.rmem_max net.core.rmem_default
```

Expected values:
- `net.core.rmem_max = 10000000`
- `net.core.rmem_default = 10000000`

### Buffer settings not persisting across reboots

**Cause**: Settings haven't been added to `/etc/sysctl.conf`.

**Solution**: Run the persistence script:
```bash
./scripts/acquisition/make_settings_persistent.sh
```

---

## DHCP Server Issues

These issues only apply to **laptop mode**. In **network mode**, DHCP is provided by your building's network infrastructure.

### Error: DHCP-Server profile not found

**Cause**: The NetworkManager DHCP-Server connection profile hasn't been created.

**Solution**: Complete the DHCP setup first:
- See [DHCP Server Setup](dhcp_setup.md) for manual setup
- Or run the [Automated Setup Wizard](automated_setup.md)

### Error: isc-dhcp-server is NOT running

**Cause**: The DHCP server service isn't running.

**Solution**:

1. Start the service:
   ```bash
   sudo systemctl start isc-dhcp-server
   ```

2. Check status:
   ```bash
   sudo systemctl status isc-dhcp-server
   ```

3. If there are errors, check the logs:
   ```bash
   sudo journalctl -u isc-dhcp-server
   ```

4. Verify configuration file:
   ```bash
   sudo nano /etc/dhcp/dhcpd.conf
   ```

### DHCP server not starting on boot

**Cause**: Service isn't enabled for auto-start.

**Solution**:

1. Enable auto-start:
   ```bash
   sudo systemctl enable isc-dhcp-server
   ```

2. Or run the persistence script:
   ```bash
   ./scripts/acquisition/make_settings_persistent.sh
   ```

### Warning: IP address is not 192.168.1.1

**Cause**: The DHCP-Server NetworkManager profile isn't active.

**Solution**:

1. Activate the profile:
   ```bash
   nmcli con up DHCP-Server
   ```

2. Verify IP address:
   ```bash
   ip addr show <interface-name>
   ```

### DHCP server not issuing leases

**Cause**: Configuration error or network connectivity issue.

**Solution**:

1. Check DHCP leases file:
   ```bash
   cat /var/lib/dhcp/dhcpd.leases
   ```

2. Verify MAC address in `/etc/dhcp/dhcpd.conf` matches your network switch

3. Restart the DHCP server:
   ```bash
   sudo systemctl restart isc-dhcp-server
   ```

4. Check that cameras are plugged in AFTER DHCP server is running

---

## Docker Issues

### Docker commands require sudo

**Cause**: Current user isn't in the docker group.

**Solution**:

1. Add user to docker group:
   ```bash
   sudo usermod -aG docker $USER
   ```

2. Apply changes (choose one):
   - Option A: Run `newgrp docker` in current terminal
   - Option B: Log out and log back in

3. Verify:
   ```bash
   groups
   ```
   Should show "docker" in the list

### Docker daemon not running

**Cause**: Docker service isn't started.

**Solution**:
```bash
sudo systemctl start docker
sudo systemctl enable docker
```

### Docker group doesn't exist

**Solution**:
```bash
sudo groupadd docker
sudo usermod -aG docker $USER
```

---

## Camera Detection Issues

### Error: No cameras detected

**Possible causes and solutions**:

1. **Cameras not powered**
   - Check that cameras are connected to PoE+ switch
   - Look for green LED on camera backs (should double-blink when connected)

2. **DHCP server not running** (laptop mode only)
   - See [DHCP Server Issues](#dhcp-server-issues)
   - Start DHCP server before plugging in cameras

3. **Network connectivity**
   - Verify switch is powered
   - Check ethernet cable connections
   - Ensure MTU is set to 9000

4. **Wrong subnet**
   - Cameras expect 192.168.1.0/24 network
   - Verify DHCP configuration in `/etc/dhcp/dhcpd.conf`

5. **Switch MAC address mismatch**
   - Check MAC address in `/etc/dhcp/dhcpd.conf` under `host switch` section
   - Update to match your actual switch MAC address

### Cameras detected but not all of them

**Solution**:

1. Check DHCP leases:
   ```bash
   cat /var/lib/dhcp/dhcpd.leases
   ```

2. Verify IP range in `/etc/dhcp/dhcpd.conf` allows enough addresses:
   ```
   range 192.168.1.10 192.168.1.100;
   ```

3. Check camera power - some may not be getting PoE

4. Verify each camera's LED is double-blinking

### Camera configuration files don't match detected cameras

**Cause**: Config files reference camera serial numbers that don't match your actual cameras.

**Solution**:

1. Detect your cameras to find serial numbers:
   ```bash
   make run
   # Check web UI at http://localhost:3000
   ```

2. Create/update YAML config files with correct serial numbers
   - See [Example Config](example_config.md)

---

## Disk Space Issues

### Error: Low disk space

**Cause**: Insufficient free space in `DATA_VOLUME` directory for recordings.

**Solution**:

1. Check disk usage:
   ```bash
   df -h /data  # or your DATA_VOLUME path
   ```

2. Free up space by:
   - Deleting old recordings
   - Moving recordings to external storage
   - Increasing disk size

3. The threshold is set by `DISK_SPACE_WARNING_THRESHOLD_GB` in `.env` (default: 50GB)

---

## DataJoint Connectivity Issues

### Error: Cannot connect to DataJoint database

**Cause**: Database server isn't reachable or credentials are wrong.

**Solution**:

1. Verify database settings in `.env`:
   - `DJ_HOST`: Should be reachable from your computer
   - `DJ_PORT`: Usually 3306 for MySQL
   - `DJ_USER` and `DJ_PASS`: Correct credentials

2. Test network connectivity:
   ```bash
   ping $DJ_HOST
   ```

3. Test database port:
   ```bash
   nc -zv $DJ_HOST $DJ_PORT
   ```

4. Check `datajoint_config.json` configuration

### Warning: DataJoint connection slow (> 2 seconds)

**Cause**: Network latency to database server.

**Solution**:
- Consider using a local database for better performance
- Check network connectivity to database host
- Verify database server isn't overloaded

---

## Camera Configuration Issues

### Warning: No camera config files found

**Cause**: No YAML configuration files exist in `CAMERA_CONFIGS` directory.

**Solution**:

1. Create a camera configuration file in your `CAMERA_CONFIGS` directory
2. Use the format from [Example Config](example_config.md)
3. Include camera serial numbers and acquisition settings

### Error: Invalid camera configuration

**Cause**: YAML file has syntax errors or missing required fields.

**Solution**:

1. Validate YAML syntax using an online validator
2. Ensure required fields are present:
   - Camera serial numbers
   - Exposure settings
   - Frame rate
   - Video segment length

3. See [Example Config](example_config.md) for reference

---

## Performance Issues

### Dropped frames during recording

**Possible causes and solutions**:

1. **MTU not set to 9000**
   - See [MTU Configuration Issues](#mtu-configuration-issues)

2. **Network buffers not configured**
   - See [Network Buffer Issues](#network-buffer-issues)

3. **Wrong network adapter**
   - Verify you're using 10GbE adapter, not 1GbE
   - Check with:
     ```bash
     ethtool <interface-name> | grep Speed
     ```
   - Should show `Speed: 10000Mb/s`

4. **Insufficient CPU/RAM**
   - Check system resources during recording:
     ```bash
     htop
     ```
   - See [General System Setup](general_system_setup.md) for minimum requirements

5. **Disk write speed too slow**
   - Test disk speed:
     ```bash
     dd if=/dev/zero of=/data/test.img bs=1G count=1 oflag=dsync
     ```
   - SSD recommended for multi-camera recording

### Slow recording startup

**Cause**: Cameras taking time to initialize.

**Solution**: This is normal - wait for all cameras to show double-blink LED before starting recording.

### Video files are suspiciously small

**Cause**: Recording failed or stopped prematurely.

**Solution**:

1. Check system logs:
   ```bash
   sudo journalctl -xe
   ```

2. Verify disk space wasn't exhausted during recording

3. Check for network errors during recording

4. Run diagnostics:
   ```bash
   sudo ./scripts/acquisition/acquisition_diagnostics.sh
   ```

---

## Getting Additional Help

If you've tried the solutions above and still have issues:

1. **Run the diagnostics script**:
   ```bash
   sudo ./scripts/acquisition/acquisition_diagnostics.sh --verbose
   ```

2. **Review the diagnostic output** for specific errors

3. **Report issues** with the diagnostic tarball at:
   https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/issues
