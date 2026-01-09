# Acquisition System Startup

## Complete all [Acquisition Setup Guides](../README.md)
## Create a [config.yaml](example_config.md) file in your configs/ directory

## Quick Start (Recommended)

If you have completed the [Persistent Settings](persistent_settings.md) setup, use the unified startup script:

```bash
./scripts/acquisition/start_acquisition.sh
```

The script will:
1. Run system checks to validate your configuration
2. Activate the DHCP-Server network profile (laptop mode only)
3. Verify network settings (MTU, IP, DHCP server)
4. Check disk space and camera configs
5. Wait for cameras to connect
6. Start the acquisition software

The browser will automatically open at http://localhost:3000

For manual startup or if you haven't run the persistence script, see the sections below.

## Step-by-Step Startup (After Persistence Setup)

### 1. Hardware Setup

- Confirm the network switch(es) are powered.
- Connect the computer to the network switch.

### 2. Activate the DHCP-Server network profile (Laptop mode only)

Select the **DHCP-Server profile** from the **Network settings in the top right corner** of the screen **or run the following command**:

```
nmcli con up id "DHCP-Server"
```

From the prerequisites, the MTU will be set automatically and the DHCP server will start automatically.

Skip this step if using network mode (building network).

### 3. Plug cameras into network switch

This step should be done **after** the DHCP server is started (laptop mode) or after connecting to the network (network mode).

### 4. Confirm there is a double blink on the green LED on the back of the cameras

### 5. Start the acquisition software

In the root of the `MultiCameraTracking`:

```
make run
```

### 6. Open a browser http://localhost:3000/
- Select a config file from the dropdown
- Set a Participant ID and click **New Session**
- Update the **Recording Base Filename** if desired
- Start recording from cameras by:
    1. **Preview**: Will show video being collected from cameras but will not save any videos
    2. **New Trial**: Will record video from each camera and will save the videos in the **Recording Dir** in the format of `RecordingBaseFilename_YYYYMMDD_HHMMSS.camera_id.mp4`

## Manual Startup

If you have not run the persistence script, follow these steps:

### 1. Hardware Setup

- Confirm the network switch(es) are powered.
- Connect the computer to the network switch.

### 2. Activate the DHCP-Server network profile (Laptop mode only)

Select the **DHCP-Server profile** from the **Network settings in the top right corner** of the screen **or run the following command**:

```
nmcli con up id "DHCP-Server"
```

### 3. Set MTU

Go to the root of the `MultiCameraTracking` repository and run:

```
sudo sh scripts/acquisition/set_mtu.sh
```

Ensure the script runs without any errors. The most likely error would be something like 'Cannot find device enp37s0'. If this error shows up, make sure the set_mtu.sh script has the correct adapter name set and ensure that adapter is connected to the computer.

### 4. Starting the DHCP Server (Laptop mode only)

To start the DHCP server, use:

```
sudo service isc-dhcp-server start
```

You can confirm if the server is running correctly with:

```
sudo service isc-dhcp-server status
```

Make sure there are no errors in the log.

If there are errors, ensure everything is connected correctly and run:

```
sudo service isc-dhcp-server restart
```

### 5. Plug cameras into network switch

This step should be done **after** the DHCP server is started (laptop mode).

### 6. Confirm there is a double blink on the green LED on the back of the cameras

### 7. Start the acquisition software

In the root of the `MultiCameraTracking`:

```
make run
```

### 8. Open a browser http://localhost:3000/
- Select a config file from the dropdown
- Set a Participant ID and click **New Session**
- Update the **Recording Base Filename** if desired
- Start recording from cameras by:
    1. **Preview**: Will show video being collected from cameras but will not save any videos
    2. **New Trial**: Will record video from each camera and will save the videos in the **Recording Dir** in the format of `RecordingBaseFilename_YYYYMMDD_HHMMSS.camera_id.mp4`