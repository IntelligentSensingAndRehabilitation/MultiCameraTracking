# Automated Setup Wizard

The setup wizard automates the complete acquisition system setup process from a fresh Ubuntu installation to a fully configured system ready for use.

## Overview

The wizard:
- Validates system prerequisites
- Installs Docker and dependencies
- Configures DHCP server (laptop mode)
- Installs switch discovery and PoE control tools (laptop mode)
- Sets the power profile to performance and disables automatic battery saver
- Sets up network interfaces
- Creates required directories
- Generates environment configuration (including SNMP community string)
- Creates the DataJoint config file
- Applies persistent settings and passwordless-sudo rules
- Builds Docker images

**Time required:** 15-30 minutes (mostly Docker image building)

## Prerequisites

Before running the setup wizard:

1. **Fresh Ubuntu 22.04 or 24.04 LTS installation**
2. **Repository cloned:**
   ```bash
   git clone https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking
   cd MultiCameraTracking
   ```
3. **Network adapter available** (the 10 GbE interface must be present — a PCIe NIC in a tower or a Thunderbolt adapter on a laptop — so the wizard can detect the interface and its MAC address; the switches and cameras can be connected later)

## Running the Setup Wizard

From the MultiCameraTracking repository root:

```bash
sudo ./scripts/acquisition/setup_acquisition_system.sh
```

The wizard must run with sudo to install packages and configure system settings.

## Detailed Steps

### Step 1: System Prerequisites Check

Validates your system meets requirements (Ubuntu version, RAM, CPU cores, repository structure).

### Step 2: Deployment Mode Selection

Choose how the system will be deployed:

**Option 1: Laptop mode (portable system with DHCP server)**
- Laptop acts as DHCP server for cameras
- Cameras connect via network switch to laptop
- Most common setup
- Requires DHCP server configuration

**Option 2: Network mode (building network infrastructure)**
- Computer and cameras on existing network
- Network already provides DHCP/routing
- Less common
- Skips DHCP configuration steps

### Step 3: Docker Installation

Checks if Docker is installed. If not, the wizard will install it automatically.

**Note:** You'll need to activate the docker group after setup (see [After Setup](#after-setup)).

For manual installation details, see [Docker Setup](docker_setup.md).

### Step 4: Network Interface Detection

Auto-detects available ethernet interfaces and prompts you to select which one connects to your cameras.

**Selection:**
- If only one interface found: Auto-selected
- If multiple interfaces: Choose from numbered list

For help identifying the correct interface, see [Network Interface Issues](troubleshooting.md#network-interface-issues).

### Step 5: DHCP Server Setup (Laptop Mode Only)

Skipped in network mode.

In laptop mode, the wizard automatically configures a DHCP server so your laptop can assign IP addresses to the cameras.

For configuration details and manual setup, see [DHCP Server Setup](dhcp_setup.md).

### Step 5b: Switch Management Tools (Laptop Mode Only)

Skipped in network mode.

Installs `lldpd` (for automatic switch discovery via LLDP) and `snmp`
(net-snmp tools for PoE port control), then enables `lldpd` on boot.  Each
switch still needs a one-time SNMP community setup via its web interface —
see [Switch Setup](switch_setup.md).

### Step 5c: Power Profile (Laptop Mode)

Sets `powerprofilesctl` to `performance` and disables GNOME's automatic
battery saver (which switches to `power-saver` on low battery, throttling
the system mid-recording).  Skipped gracefully if `powerprofilesctl` or
`gsettings` are not available.

### Step 6: Directory Creation

Prompts for directory locations with sensible defaults:

**Data storage directory** (default: /data)
- Where recordings will be saved
- Requires sufficient free space

**Camera configs directory** (default: /camera_configs)
- Where camera YAML configuration files are stored

**DataJoint external storage** (default: /mnt/datajoint_external)
- Optional directory for DataJoint database external storage
- Can skip if not using DataJoint

The wizard creates directories if they don't exist and sets proper ownership.

### Step 7: Environment Configuration

Creates `.env` file with your configuration, including:
- DataJoint database credentials
- Deployment mode and network interface
- Directory paths
- System thresholds
- SNMP community string (laptop mode — used by `make run-fresh` for PoE cycling)

If `.env` already exists, prompts before overwriting.

For details on all environment variables, see [Acquisition Software Setup](acquisition_software_setup.md).

### Step 7b: DataJoint Config File

Creates `datajoint_config.json` from `template.datajoint_config.json` with
the database host and port from `.env`.  Credentials stay in `.env` and are
forwarded to the container at runtime.  Skipped if the file already exists.

### Step 8: Persistent Network Settings

Automatically runs the persistence script to make network settings survive reboots.

For details, see [Persistent Settings](persistent_settings.md).

### Step 8b: Passwordless-sudo Rules

Installs `/etc/sudoers.d/mocap-acquisition` so the startup script can
auto-remediate common network problems (MTU, receive buffers, DHCP server,
lldpd) without prompting for a password.  This is optional — without it the
system still detects problems but prints manual instructions instead of
fixing them automatically.

### Step 9: Download FLIR SDK

Downloads the FLIR Spinnaker SDK required for camera support.

If the SDK is already downloaded, this step is automatically skipped.

### Step 10: Docker Image Build

Builds the mocap Docker image required for acquisition.

## After Setup

1. **Activate docker group** (choose one):
   - **Option A (immediate):** Run `newgrp docker` to activate in current shell
   - **Option B (permanent):** Log out and log back in
   - Verify with: `groups` (should show "docker")

2. **Create camera configuration files**
   - Location: The camera configs directory you specified (e.g., /camera_configs)
   - Format: YAML files with camera serial numbers and settings
   - See: [Example Config](example_config.md)

3. **Configure each switch's SNMP community** (one-time, per switch)
   - The wizard installs the tools but cannot configure the switches themselves
   - Log into each switch's web interface and set the SNMP community string
   - See: [Switch Setup](switch_setup.md)

## Manual Setup Alternative

If you prefer manual setup or the wizard doesn't work for your environment, follow the individual setup guides:

1. [General System Setup](general_system_setup.md)
2. [Docker Setup](docker_setup.md)
3. [DHCP Server Setup](dhcp_setup.md) (laptop mode only)
4. [Switch Setup](switch_setup.md) (laptop mode only)
5. [Acquisition Software Setup](acquisition_software_setup.md)
6. [Persistent Settings](persistent_settings.md)

## What the Wizard Doesn't Do

The wizard automates system setup but **does not:**

1. **Create camera configuration files**
   - You must create these manually for your specific cameras
   - See [Example Config](example_config.md)

2. **Configure switch SNMP communities**
   - Each switch needs a one-time setup via its web interface
   - See [Switch Setup](switch_setup.md)

3. **Install Spinnaker GUI application**
   - Only needed if you want to use Spinnaker's native GUI
   - See [Spinnaker App Setup](spinnaker_app_setup.md)

4. **Configure kernel pinning**
   - Recommended to prevent auto-updates breaking camera drivers
   - See [General System Setup](general_system_setup.md)

## Getting Help

If you encounter issues during setup:

1. **Review error messages** - the wizard provides specific guidance
2. **Check the [Troubleshooting Guide](troubleshooting.md)** for solutions to common issues
3. **Check individual setup guides** for detailed manual steps
4. **Report issues** with diagnostic output at:
   https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/issues
