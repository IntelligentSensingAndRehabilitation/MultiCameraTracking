# Automated Setup Wizard

The setup wizard automates the complete acquisition system setup process from a fresh Ubuntu installation to a fully configured system ready for use.

## Overview

The wizard:
- Validates system prerequisites
- Installs Docker and dependencies
- Configures DHCP server (laptop mode)
- Sets up network interfaces
- Creates required directories
- Generates environment configuration
- Applies persistent settings
- Builds Docker images

**Time required:** 15-30 minutes (mostly Docker image building)

## Prerequisites

Before running the setup wizard:

1. **Fresh Ubuntu 22.04 LTS installation**
2. **Repository cloned:**
   ```bash
   git clone https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking
   cd MultiCameraTracking
   ```
3. **Network adapter connected** (for laptop mode)

## Running the Setup Wizard

From the MultiCameraTracking repository root:

```bash
sudo ./scripts/acquisition/setup_acquisition_system.sh
```

The wizard must run with sudo to install packages and configure system settings.

## Detailed Steps

The wizard guides you through 9 steps:

### Step 1: System Prerequisites Check

Validates your system meets requirements

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

If `.env` already exists, prompts before overwriting.

For details on all environment variables, see [Acquisition Software Setup](acquisition_software_setup.md).

### Step 8: Persistent Network Settings

Automatically runs the persistence script to make network settings survive reboots.

For details, see [Persistent Settings](persistent_settings.md).

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

## Manual Setup Alternative

If you prefer manual setup or the wizard doesn't work for your environment, follow the individual setup guides:

1. [General System Setup](general_system_setup.md)
2. [Docker Setup](docker_setup.md)
3. [DHCP Server Setup](dhcp_setup.md) (laptop mode only)
4. [Acquisition Software Setup](acquisition_software_setup.md)
5. [Persistent Settings](persistent_settings.md)

## What the Wizard Doesn't Do

The wizard automates system setup but **does not:**

1. **Create camera configuration files**
   - You must create these manually for your specific cameras
   - See [Example Config](example_config.md)

2. **Install Spinnaker GUI application**
   - Only needed if you want to use Spinnaker's native GUI
   - See [Spinnaker App Setup](spinnaker_app_setup.md)

3. **Configure kernel pinning**
   - Recommended to prevent auto-updates breaking camera drivers
   - See [General System Setup](general_system_setup.md)

## Getting Help

If you encounter issues during setup:

1. **Review error messages** - the wizard provides specific guidance
2. **Check the [Troubleshooting Guide](troubleshooting.md)** for solutions to common issues
3. **Check individual setup guides** for detailed manual steps
4. **Report issues** with diagnostic output at:
   https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/issues
