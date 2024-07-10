import pytest
import asyncio
import tempfile
import os
from multi_camera.acquisition.flir_recording_api import FlirRecorder

@pytest.mark.parametrize('config_file', [os.path.join(os.path.dirname(__file__), 'test_configs/mobile_system_config.yaml')])

def test_flir_recording(config_file):
    acquisition = FlirRecorder()

    asyncio.run(acquisition.configure_cameras(config_file=config_file))
    print("Camera Status:")
    print(asyncio.run(acquisition.get_camera_status()))

    # Create temporary file to use for recording
    # temp_file = tempfile.NamedTemporaryFile(delete=False)

    # acquisition.start_acquisition(recording_path=temp_file.name, max_frames=100)

    # acquisition.close()