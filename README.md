# MultiCameraTracking

A comprehensive system for multi-camera video acquisition, pose estimation, 3D reconstruction, and SMPL body model fitting. Designed for clinical movement analysis and biomechanics research using multi-view synchronous recording.

## Overview

This library provides an end-to-end processing pipeline from multi-camera video acquisition through to 3D skeletal reconstruction and body model fitting:

- **Acquisition**: Multi-camera video recording with hardware synchronization (FLIR cameras with IEEE1588 support)
- **2D Pose Estimation**: Per-view pose detection using PosePipeline (state-of-the-art models)
- **3D Reconstruction**: Multi-view triangulation and temporal tracking of skeletal keypoints
- **Data Management**: Structured database pipeline via DataJoint for organizing recordings, calibrations, and results

### Core Modules

- `multi_camera.acquisition` - Recording software
- `multi_camera.acquisition.diagnostics` - Debugging tools for the recording system
- `multi_camera.analysis` - Core 3D reconstruction and analysis algorithms
- `multi_camera.datajoint` - Database schema and computational pipelines
- `multi_camera.utils` - Miscellaneous utilities

## Project Status and Development Note

This tool is under active development and used by multiple research labs. The infrastructure is technical to set up and primarily supports internal use and collaborators. We have limited bandwidth to support special or custom use cases beyond the current core functionality.

Breaking changes may be introduced during development. Issues and pull requests are welcome ([CONTRIBUTING](CONTRIBUTING)).

## Platform Compatibility

This software is designed and tested on Linux. While it has been used on Windows previously, we only guarantee functionality on Linux, and we will not provide support for cross-platform issues.

## Acquisition System Setup

To set up the acquisition system, follow the comprehensive setup **[docs](docs/README.md)**:

1. **[Hardware Setup](docs/acquisition/acquisition_hardware.md)** - Required camera equipment
2. **[System Setup](docs/acquisition/general_system_setup.md)** - Network and OS configuration
3. **[Docker Setup](docs/acquisition/docker_setup.md)** - Build the acquisition system
4. **[Acquisition Software Setup](docs/acquisition/acquisition_software_setup.md)** - Configure the recording application
5. **[Calibration Procedure](docs/calibration/calibration_procedure.md)** - Camera calibration
6. **[Annotation Setup](docs/annotation/annotation_software_setup.md)** - Annotation system configuration

## Installation (Analysis Pipeline Only)

To install the multi_camera package:

```bash
pip install -e .
```

NOTE: This pip install does not install the acquisition system, just the code required for the analysis pipeline

## Data Processing Pipeline

See the [Processing Pipeline documentation](docs/analysis/processing_pipeline.md) for detailed instructions on the complete workflow from acquisition through final 3D reconstruction, including:

- Automated pipeline script: `session_pipeline.py`
- Information about each pipeline stage

## SMPL Model Setup

See the [SMPL Model Setup documentation](docs/analysis/smpl_setup.md) for details on:

- Obtaining SMPL model files
- Directory structure and file placement
- Configuration and custom paths
- Optional extended models (SMPLx, SMPLh)

## Key Dependencies

- [PosePipeline](https://github.com/IntelligentSensingAndRehabilitation/PosePipeline) - 2D pose detection
- [EasyMocap](https://github.com/IntelligentSensingAndRehabilitation/EasyMocap/) - SMPL model fitting
- [DataJoint](https://datajoint.io/) - Database management and pipeline orchestration
- [Aniposelib](https://github.com/lambdaloop/aniposelib) - Camera calibration and triangulation
