# SMPL Model Setup

The pipeline requires SMPL body model files for fitting 3D skeletal data to a parametric body model. This guide explains what files are needed and how to set them up.

## Minimal Files Required

For basic SMPL fitting, you need only three files:

```
model_data/
├── smplx/
│   ├── J_regressor_body25.npy              (OpenPose joint mapping)
│   └── smpl/
│       └── SMPL_NEUTRAL.pkl                (Body model for fitting)
└── smpl_clean/
    └── SMPL_NEUTRAL.pkl                    (Body model for annotation)
```

Total size: ~220 MB

## Obtaining SMPL Models

### 1. Download SMPL Model Files

Visit the [SMPL website](https://smpl.is.tue.mpg.de/) and:
- Create a free account
- Download the "SMPL for Python v1.1.0" package
- Extract the files

### 2. Get the Joint Regressor

The `J_regressor_body25.npy` file is available in the [EasyMocap repository](https://github.com/zju3dv/EasyMocap):
- This file maps SMPL vertices to 25 OpenPose joints
- It is required for pose estimation compatibility

## Directory Structure

Place the files in the `model_data/` directory at the root of the MultiCameraTracking repository:

```
MultiCameraTracking/
├── model_data/
│   ├── smplx/
│   │   ├── J_regressor_body25.npy
│   │   └── smpl/
│   │       └── SMPL_NEUTRAL.pkl
│   └── smpl_clean/
│       └── SMPL_NEUTRAL.pkl
```

The code looks for SMPL files in `model_data/` relative to the repository root.

## File Details

### SMPL_NEUTRAL.pkl

This file contains the parametric body model:
- 6890 vertices representing the body mesh
- 2667 triangular faces
- Blending shapes for pose and body shape parameters
- Used for all SMPL fitting (gender='neutral' is hardcoded)

This file is needed in two locations:
1. `model_data/smplx/smpl/` - Used for 3D pose fitting
2. `model_data/smpl_clean/` - Used for mesh extraction during annotation

### J_regressor_body25.npy

Maps SMPL's 6890 vertices to 25 OpenPose joints:
- Required for converting SMPL vertex predictions to joint coordinates
- Must be placed in `model_data/smplx/`
- Automatically loaded during fitting

## Custom Paths

If you have SMPL files in a different location, you can override the default paths with environment variables:

```bash
# Override SMPL location for fitting
export SMPLX_PATH="/path/to/model_data/smplx"

# Override SMPL location for annotation
export SMPL_CLEAN_PATH="/path/to/model_data/smpl_clean"
```

## Extended Models (Optional)

The pipeline supports SMPLx (with hand and face parameters) and SMPLh (with hand parameters), though these are not commonly used in the standard workflow. If you want to enable these models in the future, add:

```
model_data/smplx/
├── smplx/
│   └── SMPLX_NEUTRAL.pkl        (SMPL + hands + face)
└── J_regressor_body25_smplx.txt (Joint regressor for SMPLx)
```

Currently, MultiCameraTracking uses SMPL by default. Gender-specific models (MALE, FEMALE) are available in the SMPL download but are not used by the pipeline, which is hardcoded to use the neutral model.
