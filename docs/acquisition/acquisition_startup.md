# Acquisition System Startup

## Complete all [Acquisition Setup Guides](../README.md)
## Create a [config.yaml](example_config.md) file in your configs/ directory

## Quick Start (Recommended)

If you have completed the [Persistent Settings](persistent_settings.md) setup, use the unified startup script:

```bash
./scripts/acquisition/start_acquisition.sh
```

The script validates your system and starts acquisition automatically. For details, see [Startup Script](startup_script.md).

For manual startup or if you haven't run the persistence script, see the sections below.

## Manual Startup

### 1. Hardware Setup

- Confirm the network switch(es) are powered.
- Connect the computer to the network switch.

### 2. Activate the DHCP-Server network profile (Laptop mode only)

```bash
nmcli con up id "DHCP-Server"
```

Or select the DHCP-Server profile from the Network settings GUI.

For troubleshooting, see [DHCP Server Issues](troubleshooting.md#dhcp-server-issues).

### 3. Set MTU

```bash
sudo sh scripts/acquisition/set_mtu.sh
```

For troubleshooting, see [MTU Configuration Issues](troubleshooting.md#mtu-configuration-issues).

### 4. Starting the DHCP Server (Laptop mode only)

```bash
sudo service isc-dhcp-server start
```

Verify it's running:
```bash
sudo service isc-dhcp-server status
```

For troubleshooting, see [DHCP Server Issues](troubleshooting.md#dhcp-server-issues).

### 5. Plug cameras into network switch

This step should be done **after** the DHCP server is started (laptop mode).

### 6. Confirm there is a double blink on the green LED on the back of the cameras

### 7. Start the acquisition software

```bash
make run
```

### 8. Open the web interface

Navigate to http://localhost:3000/ in a browser:

1. Select a config file from the dropdown
2. Set a Participant ID and click **New Session**
3. Update the **Recording Base Filename** if desired
4. Start recording:
   - **Preview**: View live video without recording
   - **New Trial**: Record video from all cameras

For troubleshooting, see the [Troubleshooting Guide](troubleshooting.md).