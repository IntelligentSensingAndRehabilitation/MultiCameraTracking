# Annotation Software Installation Guide

## MultiCameraTracking Setup

1. Clone **[MultiCameraTracking](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking)**

2. Copy the SMPL files into the `model_data/` directory
    
3. Copy `datajoint_config.json` from **PosePipeline** into the root of **MultiCameraTracking**
    - In **PosePipeline**, this file is called `Example_dj_local_conf.json` so rename it to `datajoint_config.json`
    - In `datajoint_config.json` update the `"stores" > "localattach" > "location"`" to point to `"/datajoint_external"`

4. Run `make build-annotate` from the root of **MultiCameraTracking** to build the Docker container