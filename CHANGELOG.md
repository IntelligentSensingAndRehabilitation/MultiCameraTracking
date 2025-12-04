# Changelog - Multi-Camera Tracking
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Calendar Versioning](https://calver.org/).

## [2025.12.03] - 2025-12-03

### Added
- Ability to annotate FMS activities [[#71]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/71)
- Table to link Sessions and Calibrations allowing us to run reconstructions on Sessions that had multiple calibration hashes [[#74]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/74)
- Ability to handle different pixel formats [[#78]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/78)
- New visualizations [[#79]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/79)
- Add more activities for annotation and session_pipeline fixes [[#84]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/84)
- Added kinematic reconstruction monitoring to the pipeline dashboard [[#89]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/89)
- Added authentication to dashboard and updated queries to get more accurate numbers [[#94]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/94)
- Added hand pose estimation to post annotation [[#95]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/95)
- Added acquisition diagnostics script [[#96]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/96)
- Added tests for data integrity [[#101]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/101)

### Updated
- Updated Acquisition Docker and documentation in acquisition [[#75]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/75)
- Updated session_pipeline and the way we handle SMPL paths [[#77]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/77)
- Docker no longer is built on PosePipeline base image [[#80]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/80)
- Updates to session_pipeline defaults, filtering, and flags [[#83]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/83)
- Full rewrite of calibration and releveling [[#89]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/89)
- Migrated dashboard code to isr-dashboards [[#102]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/102)
- General repo cleanup for public release [[#103]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/103)


### Fixed
- [[#72]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/72)
- [[#76]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/76)
- [[#81]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/81)
- [[#82]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/82)
- [[#85]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/85)
- [[#86]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/86)
- [[#92]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/92)
- [[#93]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/93)
- [[#97]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/97)
- [[#100]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/100)

## [2024.01.14] - 2025-01-14
- Initial Release

### Added
- Config file support to select cameras to record from and expose commonly used settings [[#2]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/2)
- React frontend with FastAPI backend [[#6]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/6)
- Dockerized Acquisition Software [[#7]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/7)
- Annotation tables and Dashboard [[#25]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/25)
- Moving from Checkerboard calibration to ChAruco [[#43]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/43)
- Initial pytests [[#52]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/52)
- Initial quality metrics [[#55]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/55)
- Annotation as a standalone application [[#60]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/60)
- Transition to uv as package manager [[#63]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/63)
- Implementing threads for acquisition from cameras [[#67]](https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/pull/67)
