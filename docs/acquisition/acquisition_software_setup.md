# Acquisition Software Installation Guide

## MultiCameraTracking Setup

1. Clone **[MultiCameraTracking](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking)**

2. Run `download_flir.sh` from the `MultiCameraTracking/docker/` directory
3. Create a **data** directory locally (this is where the videos and JSONs will be written)
4. Create a **configs** directory locally (this is where the camera config files will be stored)
    - An example config.yaml file can be found [here](example_config.md)
5. Rename `.env.template` to `.env` in the root of the **MultiCameraTracking** repo
    - Fill in `.env` file with the following info, corresponding to your setup

        ```python
        DJ_USER=root # DataJoint username
        DJ_PASS=pose # DataJoint password
        DJ_HOST=127.0.0.1 # DataJoint DB Hostname
        DJ_PORT=3306 # DataJoint DB port number
        NETWORK_INTERFACE=enp5s0 # You get this from the DHCP Setup step
        DEPLOYMENT_MODE=laptop # "laptop" (with DHCP server) or "network" (building network)
        REACT_APP_BASE_URL=localhost # Can set this to FQDN of the machine running the software
        DATA_VOLUME=/data # This is the path from Step 3
        CAMERA_CONFIGS=/camera_configs # This is the path from Step 4
        DATAJOINT_EXTERNAL=/mnt/datajoint_external # This is your datajoint external localattach
        DISK_SPACE_WARNING_THRESHOLD_GB=50 # Minimum free disk space in GB
        ```
6. Rename `template.datajoint_config.json` to `datajoint_config.json`
7. Run `make build-mocap` from the root of **MultiCameraTracking** to build the Docker container
8. Run the persistence script to make network settings survive reboots (see [Persistent Settings](persistent_settings.md))
    ```bash
    ./scripts/acquisition/make_settings_persistent.sh
    ```