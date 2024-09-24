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

    # Use the mounted volume for output
    output_dir = '/Mocap/tests/testdata'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique filename for this test run
    video_filename = "test_recording"
    recording_path = os.path.join(output_dir, video_filename)
    
    print("Recording Path: ", recording_path)

    acquisition.start_acquisition(recording_path=recording_path, max_frames=1000)

    acquisition.close()