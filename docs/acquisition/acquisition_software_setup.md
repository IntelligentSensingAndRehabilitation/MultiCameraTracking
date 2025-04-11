# Acquisition Software Installation Guide

## MultiCameraTracking Setup

1. Clone **[MultiCameraTracking](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking)**

2. Run `download_flir.sh` from the `MultiCameraTracking/docker/` directory
    
3. Create a **data** directory locally (this is where the videos and JSONs will be written)
4. Create a **configs** directory locally (this is where the camera config files will be stored)
    - An example config.yaml file can be found [here](example_config.md)
5. Copy `datajoint_config.json` from **PosePipeline** into the root of **MultiCameraTracking**
    - In **PosePipeline**, this file is called `Example_dj_local_conf.json` so rename it to `datajoint_config.json`
    - In `datajoint_config.json` update the `"stores" > "localattach" > "location"`" to point to `"/datajoint_external"`
6. Create a `.env` file in the root of the **MultiCameraTracking** repo
    - Fill in the `.env` file with the following info, corresponding to your setup
        
        ```python
        NETWORK_INTERFACE=enp5s0 # You get this from the DHCP Setup step
        REACT_APP_BASE_URL=localhost # Dont need to change this
        DATA_VOLUME=/data # This is the path from Step 3 
        CAMERA_CONFIGS=/camera_configs # This is the path from Step 4
        DATAJOINT_EXTERNAL=/mnt/datajoint_external
        ```
        
7. Run `make build-mocap` from the root of **MultiCameraTracking** to build the Docker container