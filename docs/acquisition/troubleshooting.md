# Acquisition System Troubleshooting Guide

This guide provides solutions to common issues with the acquisition system. Each section can be referenced directly by other documentation.

## Table of Contents

- [Network Interface Issues](#network-interface-issues)
- [MTU Configuration Issues](#mtu-configuration-issues)
- [Network Buffer Issues](#network-buffer-issues)
- [DHCP Server Issues](#dhcp-server-issues)
- [Camera Detection Issues](#camera-detection-issues)
- [DataJoint Connectivity Issues](#datajoint-connectivity-issues)
- [Performance Issues](#performance-issues)

---

## Network Interface Issues

### Error: Network interface not found

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

**Solution Option 1 - Make it persistent** (recommended):
```bash
./scripts/acquisition/make_settings_persistent.sh
```

**Solution Option 2 - Set manually for current session**:
```bash
sudo sh scripts/acquisition/set_mtu.sh
```

### Verify MTU setting

```bash
ip link show <interface-name>
```

Look for `mtu 9000` in the output.

### MTU not persisting across reboots

**Solution**: Run the persistence script:
```bash
./scripts/acquisition/make_settings_persistent.sh
```

---

## Network Buffer Issues

### Warning: Network buffers not configured

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

**Solution**: Run the persistence script:
```bash
./scripts/acquisition/make_settings_persistent.sh
```

---

## DHCP Server Issues

These issues only apply to **laptop mode**.

### Error: isc-dhcp-server is NOT running

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

**Solution**:

1. Enable auto-start:
   ```bash
   sudo systemctl enable isc-dhcp-server
   ```

2. Or run the persistence script:
   ```bash
   ./scripts/acquisition/make_settings_persistent.sh
   ```

### DHCP server not issuing leases

**Solution**:

1. Verify MAC address in `/etc/dhcp/dhcpd.conf` matches your network switch

2. Restart the DHCP server:
   ```bash
   sudo systemctl restart isc-dhcp-server
   ```

3. Check that cameras are plugged in AFTER DHCP server is running

---

## Camera Detection Issues

### Error: No cameras detected

**Possible causes and solutions**:

1. **Cameras not powered**
   - Check that cameras are connected to PoE+ switch
   - Look for green LED on camera backs (should double-blink when they get their IP address)

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

---
## Misc troubleshooting steps:

JC-MOBILE02: always do a laptop reboot before starting setup.

Troubleshooting order of operations (especially if there are "Spinnaker" or Spinview issues):
   1. In the GUI, in the config dropdown menu, select the <blank> option. Wait for GUI to say "Idle" or "Sync". Then in the config dropdown menu, select the <<maclab>> config and wait for GUI to say "Idle" or "Sync", and do a preview/test recording
   2. Close GUI. In Terminal, type "CTRL+C", then "make run". Open GUI and do a preview/test recording.
   3. In the GUI, click the [Reset Cameras] button, and try to do a preview/test recording.
   4.  In the Linux "Start" menu (top left corner of JC-MOBILE02 [--] icon), search for "Spinview" and open it. On all lines that have a red dot/exclamation point, right click and select <Force IP>; otherwise doubleclick on each line to see if you can open it. In GUI, do a preview/test recording.
   5. Close GUI. In Terminal, type "CTRL+C", then "make reset". Wait for Terminal to stop printing, then type "make run". Open GUI and do a preview/test recording. (This will reset ALL cameras in ALL configs, so if someone else is simultaneously recording, it'll reset their cameras.)
   6. Ethernet adapter should have all green lights. If there's any orange lights that means there's not enough bandwidth, no point in waiting to see if it resolves. Unplug and replug the ethernet adapater. Run diagnostics, if no errors, then "make run".
   7. In Terminal, CTRL+C. Unplug and replug the 12 ethernet cords in the camera/switch box. Then "make run".
   8. Reboot entire laptop and start over from scratch. (FYI rebooting JC-COMPUTE02 will unmount and only James is able to re-mount.)
