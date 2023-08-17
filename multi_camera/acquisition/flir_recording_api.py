import PySpin
import simple_pyspin
from simple_pyspin import Camera
import numpy as np
from tqdm import tqdm
from datetime import datetime
from queue import Queue
from typing import List, Callable, Awaitable
from pydantic import BaseModel
import concurrent.futures
import threading
import asyncio
import json
import time
import cv2
import os
import yaml
import pandas as pd
import hashlib


# Data structures we will expose outside this library
class CameraStatus(BaseModel):
    # This contains the information from init_camera
    SerialNumber: str
    Status: str = "Not Initialized"
    # PixelSize: float = 0.0
    PixelFormat: str = ""
    BinningHorizontal: int = 0
    BinningVertical: int = 0
    Width: int = 0
    Height: int = 0
    SyncOffset: float = 0.0


def select_interface(interface, cameras):
    # This method takes in an interface and list of cameras (if a config
    # file is provided) or number of cameras. It checks if the current
    # interface has cameras and returns a list of valid camera IDs or
    # number of cameras

    print("Update cameras:", interface.UpdateCameras())

    # Check the current interface to see if it has cameras
    interface_cams = interface.GetCameras()
    # Get the number of cameras on the current interface
    num_interface_cams = interface_cams.GetSize()

    retval = None

    if num_interface_cams > 0:
        # If camera list is passed, confirm all SNs are valid
        if isinstance(cameras, list):
            camera_id_list = []

            for c in cameras:
                cam = interface_cams.GetBySerial(str(c))
                if cam.IsValid():
                    camera_id_list.append(str(c))

                del cam  # must release handle

            # if the camera_ID_list does not contain any valid cameras
            # based on the serial numbers present in the config file
            # return None
            if len(camera_id_list) > 0:
                # Find any invalid IDs in the config
                invalid_ids = [c for c in cameras if str(c) not in camera_id_list]

                if invalid_ids:
                    print(f"The following camera ID(s) from are missing: {invalid_ids} but continuing")

                retval = camera_id_list

        # If num_cams is passed, confirm it is less than or equal to
        # the size of interface_cams and return the correct num_cams
        if isinstance(cameras, int):
            # if num_cams is larger than the # cameras on current interface,
            # raise an error
            assert (
                cameras <= num_interface_cams
            ), f"num_cams={cameras} but the current interface only has {num_interface_cams} cameras."

            # Otherwise, set num_cams to the # of available cameras
            num_cams = cameras
            print(f"No config file passed. Selecting the first {num_cams} cameras in the list.")

            retval = num_cams

    # need to make sure we release this handle
    interface_cams.Clear()

    # If there are no cameras on the interface, return None
    return retval


def init_camera(
    c: Camera,
    jumbo_packet: bool = True,
    triggering: bool = True,
    throughput_limit: int = 125000000,
    resend_enable: bool = False,
    binning: int = 1,
    exposure_time: int = 15000,
):
    """
    Initialize camera with settings for recording

        Args:
            c (Camera): Camera object
            jumbo_packet (bool): Enable jumbo packets
            triggering (bool): Enable network triggering for start
            throughput_limit (int): Throughput limit for camera.
            resend_enable (bool): Enable packet resend
            binning (int): Factor by which the image resolution is reduced

        Throughput should be limited for multiple cameras but reduces frame rate. Can use 125000000 for maximum
        frame rate or 85000000 when using more cameras with a 10GigE switch.
    """

    # Initialize each available camera
    c.init()

    # Resetting binning to 1 to allow for maximum frame size
    c.BinningHorizontal = 1
    c.BinningVertical = 1

    # Ensuring height and width are set to maximum
    c.Width = c.WidthMax
    c.Height = c.HeightMax

    c.PixelFormat = "BayerRG8"  # BGR8 Mono8
    
    # Now applying desired binning to maximum frame size
    c.BinningHorizontal = binning
    c.BinningVertical = binning

    # use a fixed exposure time to ensure good synchronization. also want to keep this relatively
    # low to reduce blur while obtaining sufficient light
    c.ExposureAuto = "Off"
    c.ExposureTime = exposure_time

    # let the auto gain match the brightness across images as much as possible
    c.GainAuto = "Continuous"
    # c.Gain = 10

    c.ImageCompressionMode = "Off"  # Lossless might get frame rate up but not working currently
    # c.IspEnable = True  # if trying to adjust the color transformations  this is needed

    if jumbo_packet:
        c.GevSCPSPacketSize = 9000
    else:
        c.GevSCPSPacketSize = 1500

    c.DeviceLinkThroughputLimit = throughput_limit
    c.GevSCPD = 25000

    # c.StreamPacketResendEnable = resend_enable

    if triggering:
        # set up masks for triggering
        c.ActionDeviceKey = 0
        c.ActionGroupKey = 1
        c.ActionGroupMask = 1

        # set up trigger setting
        c.TriggerMode = "Off"
        c.TriggerSelector = "AcquisitionStart"  # Need to select AcquisitionStart for real time clock
        c.TriggerSource = "Action0"
        c.TriggerMode = "On"


def write_queue(
    vid_file: str, image_queue: Queue, json_queue: Queue, serial, pixel_format: str, acquisition_fps: float
):
    """
    Write images from the queue to a video file

    Args:
        vid_file (str): Path to video file
        image_queue (Queue): Queue to read images from
        json_queue (Queue): Queue to write the json information about timestamps to
        serial (str): Camera serial number
        pixel_format (str): Pixel format of the camera
        acquisition_fps (float): Frame rate of camera in Hz

    Filename is determined by the vid_file and time_str. The serial number is appended to the end of the filename.

    This is expected to be called from a standalone thread and will autoamtically terminate when the image_queue is empty.
    """

    vid_file = os.path.splitext(vid_file)[0] + f".{serial}.mp4"

    print(vid_file)

    timestamps = []
    real_times = []
    frame_spreads = []

    out_video = None

    for frame in iter(image_queue.get, None):
        if frame is None:
            break
        timestamps.append(frame["timestamps"])
        real_times.append(frame["real_times"])

        im = frame["im"]

        if pixel_format == "BayerRG8":
            im = cv2.cvtColor(im, cv2.COLOR_BAYER_RG2RGB)

        # need to collect two frames to track the FPS
        if out_video is None and len(real_times) == 1:
            last_im = im

        elif out_video is None and len(real_times) > 1:
            tqdm.write(f"Writing FPS: {acquisition_fps}")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_video = cv2.VideoWriter(vid_file, fourcc, acquisition_fps, (im.shape[1], im.shape[0]))
            out_video.write(last_im)

        else:
            out_video.write(im)

            # Check the timestamp spread between the current frame and previous frame
            # print(f"{serial} Timestamp spread: {timestamps[-1] - timestamps[-2]}")
            if (timestamps[-1] - timestamps[-2]) * 1e-6 > acquisition_fps * 1.2:
                print(f"Warning | {serial} Timestamp spread: {(timestamps[-1] - timestamps[-2]) * 1e-6} {acquisition_fps} {acquisition_fps * 1.2}")
                # frame_spreads.append((timestamps[-1] - timestamps[-2]) * 1e-6)

        image_queue.task_done()

    out_video.release()

    # Adding the json info corresponding to the current camera to its own queue
    json_queue.put({"serial": serial, "timestamps": timestamps, "real_times": real_times})

    # average frame time from ns to s
    ts = np.asarray(timestamps)
    delta = np.mean(np.diff(ts, axis=0)) * 1e-9
    fps = 1.0 / delta

    print(f"Finished writing images. Final fps: {fps}")

    # indicate the last None event is handled
    image_queue.task_done()


class FlirRecorder:
    def __init__(
        self,
        status_callback: Callable[[str], None] = None,
    ):
        self._get_pyspin_system()

        # Set up thread safe semaphore to stop recording from a different thread
        self.stop_recording = threading.Event()

        self.preview_callback = None
        self.cams = []
        self.image_queue_dict = {}
        self.config_file = None
        self.iface = None
        self.status_callback = status_callback
        self.set_status("Uninitialized")

    def get_config_hash(self,yaml_content,hash_len=10):

        # Sorting keys to ensure consistent hashing
        file_str = json.dumps(yaml_content,sort_keys=True)
        encoded_config = file_str.encode('utf-8')

        # Create hash of encoded config file and return
        return hashlib.sha256(encoded_config).hexdigest()[:hash_len]

    def _get_pyspin_system(self):
        # use this to ensure both calls with simple pyspin and locally use the same references
        simple_pyspin.list_cameras()
        self.system = simple_pyspin._SYSTEM  # PySpin.System.GetInstance()

    def get_acquisition_status(self):
        return self.status

    def set_status(self, status):
        print("setting status: ", status)
        self.status = status
        if self.status_callback is not None:
            self.status_callback(status)

    def set_progress(self, progress):
        if self.status_callback is not None:
            self.status_callback(self.status, progress=progress)

    async def synchronize_cameras(self):
        if not all([c.GevIEEE1588 for c in self.cams]):
            self.set_status("Synchronizing")

            print("Cameras not synchronized. Enabling IEEE1588 (takes 10 seconds)")
            for c in self.cams:
                c.GevIEEE1588 = True

            await asyncio.sleep(10)

        self.set_status("Synchronized")

    async def configure_cameras(
        self, config_file: str = None, num_cams: int = None, trigger: bool = True
    ) -> Awaitable[List[CameraStatus]]:
        """
        Configure cameras for recording

        Args:
            config_file (str): Path to config file
            num_cams (int): Number of cameras to configure (if not using config file)
            trigger (bool): Enable network synchronized triggering
        """

        self.config_file = config_file

        iface_list = self.system.GetInterfaces()

        if config_file:
            with open(config_file, "r") as file:
                self.camera_config = yaml.safe_load(file)

            # Updating interface_cameras if a config file is passed
            # with the camera IDs passed
            requested_cameras = list(self.camera_config["camera-info"].keys())
        else:
            assert num_cams is not None, "Must provide number of cameras if no config file is provided"
            requested_cameras = num_cams
            self.camera_config = {}

        print(f"Requested cameras: {requested_cameras}")

        # Identify the interface we are going to send a command for synchronous recording
        iface = None
        for i, current_iface in enumerate(iface_list):
            selected_cams = select_interface(current_iface, requested_cameras)

            # If the value returned from select_interface is not None,
            # select the current interface
            if selected_cams is not None:
                # Break out of the loop after finding the interface and cameras
                break

        print(f"Using interface {i} with {selected_cams} cameras. In use: {current_iface.IsInUse()}")

        iface_list.Clear()

        # Confirm that cameras were found on an interface
        assert current_iface is not None, "Unable to find valid interface."
        self.iface = current_iface
        self.iface_cameras = selected_cams

        self.trigger = trigger

        self.iface.TLInterface.GevActionDeviceKey.SetValue(0)
        self.iface.TLInterface.GevActionGroupKey.SetValue(1)
        self.iface.TLInterface.GevActionGroupMask.SetValue(1)

        if type(self.iface_cameras) is int:
            self.cams = [Camera(i, lock=True) for i in range(self.iface_cameras)]
        else:
            # if config is passed then use the config list
            # of cameras to select
            self.cams = [Camera(i, lock=True) for i in self.iface_cameras]

        if self.camera_config:
            # Parse additional parameters from the config file
            exposure_time = self.camera_config["acquisition-settings"]["exposure_time"]
            frame_rate = self.camera_config["acquisition-settings"]["frame_rate"]
        else:
            # If no config file is passed, use default values
            exposure_time = 15000
            frame_rate = 30

        # Updating the binning needed to run at 60 Hz. 
        # TODO: make this check more robust in the future
        if frame_rate == 60:
            binning = 2
        else:
            binning = 1

        config_params = {
            "jumbo_packet": True,
            "triggering": self.trigger,
            "throughput_limit": 125000000,
            "resend_enable": False,
            "binning": binning,
            "exposure_time": exposure_time,
        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.cams)) as executor:
            list(executor.map(lambda c: init_camera(c, **config_params), self.cams))

        await self.synchronize_cameras()

        self.cams.sort(key=lambda x: x.DeviceSerialNumber)
        self.pixel_format = self.cams[0].PixelFormat

        self.set_status("Idle")

    async def get_camera_status(self) -> List[CameraStatus]:
        status = [
            CameraStatus(
                SerialNumber=c.DeviceSerialNumber,
                Status="Initialized",
                # PixelSize=c.PixelSize,
                PixelFormat=c.PixelFormat,
                BinningHorizontal=c.BinningHorizontal,
                BinningVertical=c.BinningVertical,
                Width=c.Width,
                Height=c.Height,
            )
            for c in self.cams
        ]

        for c in self.cams:
            c.GevIEEE1588DataSetLatch()
            # print(
            #    "Primary" if c.GevIEEE1588StatusLatched == "Master" else "Secondary",
            #    c.GevIEEE1588OffsetFromMasterLatched,
            # )

            # set the corresponding camera status
            for cs in status:
                if cs.SerialNumber == c.DeviceSerialNumber:
                    cs.SyncOffset = c.GevIEEE1588OffsetFromMasterLatched

        status.sort(key=lambda x: x.SerialNumber)

        return status

    def start_acquisition(self, recording_path=None, preview_callback: callable = None, max_frames: int = 1000):
        self.set_status("Recording")

        self.preview_callback = preview_callback
        self.video_base_file = recording_path

        # Initializing an image queue for each camera
        self.image_queue_dict = {c.DeviceSerialNumber: Queue(max_frames) for c in self.cams}

        # set up the threads to write videos to disk, if requested
        if self.video_base_file is not None:
            json_queue = {}

            # Start a writing thread for each camera
            for c in self.cams:
                serial = c.DeviceSerialNumber
                # Initializing queue to store json info for each camera
                json_queue[c.DeviceSerialNumber] = Queue(max_frames)
                threading.Thread(
                    target=write_queue,
                    kwargs={
                        "vid_file": self.video_base_file,
                        "image_queue": self.image_queue_dict[serial],
                        "json_queue": json_queue[c.DeviceSerialNumber],
                        "serial": serial,
                        "pixel_format": self.pixel_format,
                        "acquisition_fps": c.AcquisitionFrameRate,
                    },
                ).start()

        def start_cam(i):
            # this won't truly start them until command is send below
            self.cams[i].start()

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.cams)) as executor:
            l = list(executor.map(start_cam, range(len(self.cams))))

        print("Acquisition, Resulting, Exposure, DeviceLinkThroughputLimit:")
        for c in self.cams:
            print(f"{c.DeviceSerialNumber}: {c.AcquisitionFrameRate}, {c.AcquisitionResultingFrameRate}, {c.ExposureTime}, {c.DeviceLinkThroughputLimit} ")
            print(f"Frame Size: {c.Width} {c.Height}")

        # schedule a command to start in 250 ms in the future
        self.cams[0].TimestampLatch()
        value = self.cams[0].TimestampLatchValue
        latchValue = int(value + 0.250 * 1e9)
        self.iface.TLInterface.GevActionTime.SetValue(latchValue)
        self.iface.TLInterface.GevActionGroupKey.SetValue(1)  # these group/mask/device numbers should match above
        self.iface.TLInterface.GevActionGroupMask.SetValue(1)
        self.iface.TLInterface.GevActionDeviceKey.SetValue(0)
        self.iface.TLInterface.ActionCommand()

        all_timestamps = []

        for frame_idx in tqdm(range(max_frames)):
            # Use thread safe checking of semaphore to determine whether to stop recording
            if self.stop_recording.is_set():
                self.stop_recording.clear()
                print("Stopping recording")
                break

            self.set_progress(frame_idx / max_frames)

            # get the image raw data
            # for each camera, get the current frame and assign it to
            # the corresponding camera
            real_times = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            real_time_images = []

            frame_timestamps = {"real_times": real_times}

            for c in self.cams:
                im_ref = c.get_image()
                timestamps = im_ref.GetTimeStamp()

                # store camera timestamps
                frame_timestamps[c.DeviceSerialNumber] = timestamps

                # get the data array
                # Using try/except to handle frame tearing
                try:
                    im = im_ref.GetNDArray()
                    im_ref.Release()

                    if self.preview_callback is not None:
                        # if preview is enabled, save the size of the first image
                        # and append the image from each camera to a list
                        real_time_images.append(im)

                except Exception as e:
                    tqdm.write("Bad frame")
                    continue

                # Writing the frame information for the current camera to its queue
                self.image_queue_dict[c.DeviceSerialNumber].put(
                    {"im": im, "real_times": real_times, "timestamps": timestamps}
                )

            all_timestamps.append(frame_timestamps)

            if self.preview_callback:
                self.preview_callback(real_time_images)

        if self.preview_callback:
            self.preview_callback(None)

        exposure_times = []
        frame_rates = []
        camera_ids = []
        for c in self.cams:
            # Recording the final exposure times and requested frame rates for each camera
            # Actual frame rate can be calculated from the timestamps in the output json
            exposure_times.append(c.ExposureTime)
            frame_rates.append(c.BinningHorizontal * 30)

            camera_ids.append(c.DeviceSerialNumber)
            # Stopping each camera
            c.stop()

        # Creating a dictionary to hold the contents of each camera's json queue
        output_json = {}

        # convert list of dicts to dict of lists
        all_timestamps = {k: [dic[k] for dic in all_timestamps] for k in all_timestamps[0]}

        output_json["real_times"] = all_timestamps.pop("real_times")
        output_json["serials"] = []  # list(all_timestamps.keys())
        output_json["timestamps"] = []  # all_timestamps
        for k, v in all_timestamps.items():
            output_json["serials"].append(k)
            output_json["timestamps"].append(v)
        output_json["timestamps"] = np.array(output_json["timestamps"]).T.tolist()
        print(np.array(output_json["timestamps"]).shape)

        if self.camera_config:
            output_json["meta_info"] = self.camera_config["meta-info"]
            output_json["camera_info"] = self.camera_config["camera-info"]
            output_json["exposure_times"] = exposure_times
            output_json["frame_rate_requested"] = frame_rates
            camera_config_hash = self.get_config_hash(self.camera_config)
            print("CONFIG HASH",camera_config_hash)
            output_json["camera_config_hash"] = camera_config_hash
        else:
            output_json["meta_info"] = "No Config"
            output_json["camera_info"] = camera_ids
            output_json["exposure_times"] = exposure_times
            output_json["frame_rate_requested"] = frame_rates
            output_json["camera_config_hash"] = None

        if self.video_base_file is not None:
            # stop video writing threads and output json file

            # to allow each queue to be processed before moving on
            for c in self.cams:
                self.image_queue_dict[c.DeviceSerialNumber].put(None)
                self.image_queue_dict[c.DeviceSerialNumber].join()

            # defining the filename for the json file
            json_file = os.path.splitext(self.video_base_file)[0] + ".json"

            # writing the json file for the current recording session
            json.dump(output_json, open(json_file, "w"))

        # Calculating metrics to determine drift
        ts = pd.DataFrame(output_json["timestamps"])

        # interpolating any timestamps that are 0s
        ts.replace(0, np.nan, inplace=True)
        ts.interpolate(method='linear', axis=0, limit=1, limit_direction='both', inplace=True)
        initial_ts = ts.iloc[0,0]
        dt = (ts - initial_ts) / 1e9
        spread = dt.max(axis=1) - dt.min(axis=1)

        ts['std'] = ts.std(axis=1) / 1e6
        if np.all(spread < 1e-6):
            print("Timestamps well aligned and clean")
        else:
            print(f"Timestamps showed a maximum spread of {np.max(spread) * 1000} ms")
            print(f"Timestamp standard deviation {ts['std'].max() -  ts['std'].min()} ms")

        self.set_status("Idle")

        return {"timestamp_spread": np.max(spread) * 1000, "recording_timestamp": output_json["real_times"][0]}

    def stop_acquisition(self):
        self.stop_recording.set()

    async def reset_cameras(self):
        """Reset all the cameras and reopen the system"""

        self.set_status("Resetting")
        await asyncio.sleep(0.1)  # let the web service update with this message

        # store the serial numbers to get and reset
        serials = [c.DeviceSerialNumber for c in self.cams]
        config_file = self.config_file  # grab this before closing as it is cleared

        # this releases all the handles to the pyspin system.
        self.close()

        print("Reopening and resetting")
        ########## working with new, temporary, reference to PySpin system
        # this seems important for reliability

        # find the set of cameras and trigger a reset on them
        system = PySpin.System.GetInstance()
        cams = system.GetCameras()

        def reset_cam(s):
            print("Opening and resetting camera", s)
            c = cams.GetBySerial(s)
            c.Init()
            c.DeviceReset()
            c.DeInit()
            del c  # force release of the handle

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(serials)) as executor:
            executor.map(reset_cam, serials)

        cams.Clear()
        system.ReleaseInstance()

        ########## go back to the original reference to the PySpin system

        self.set_status("Reset complete. Waiting to reconfigure.")
        await asyncio.sleep(15)

        # set up the PySpin system reference again
        self._get_pyspin_system()

        if config_file is not None and config_file != "":
            await self.configure_cameras(config_file)

    def close(self):
        """Close all the cameras and release the system"""

        if len(self.cams) > 0:

            def close_cam(c):
                print("Closing camera", c.DeviceSerialNumber)
                c.cam.DeInit()
                c.close()
                del c

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.cams)) as executor:
                executor.map(close_cam, self.cams)

        self.cams = []

        if self.iface is not None:
            del self.iface
            self.iface = None

        simple_pyspin._SYSTEM.ReleaseInstance()
        del simple_pyspin._SYSTEM
        simple_pyspin._SYSTEM = None

        self.system = None

        self.config_file = None

        print("PySpin system released")

    def reset(self):
        self.close()
        self._get_pyspin_system()


if __name__ == "__main__":
    import argparse
    import signal

    parser = argparse.ArgumentParser(description="Record video from GigE FLIR cameras")
    parser.add_argument("vid_file", help="Video file to write")
    parser.add_argument("-m", "--max_frames", type=int, default=1000, help="Maximum frames to record")
    parser.add_argument("-n", "--num_cams", type=int, default=4, help="Number of input cameras")
    parser.add_argument("-r", "--reset", default=False, action="store_true", help="Reset cameras first")
    parser.add_argument(
        "-p", "--preview", default=False, action="store_true", help="Allow real-time visualization of video"
    )
    parser.add_argument(
        "-s",
        "--scaling",
        type=float,
        default=0.5,
        help="Ratio to use for scaling the real-time visualization output (should be a float between 0 and 1)",
    )
    parser.add_argument("-c", "--config", default="", type=str, help="Path to a config.yaml file")
    args = parser.parse_args()

    print(args.config)
    acquisition = FlirRecorder()
    asyncio.run(acquisition.configure_cameras(config_file=args.config, num_cams=args.num_cams))

    print(asyncio.run(acquisition.get_camera_status()))

    if args.reset:
        print("reset")
        asyncio.run(acquisition.reset_cameras())

    # time.sleep(5)

    # Get the timestamp that should be used for the file names
    now = datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S")
    filename = f"{args.vid_file}_{time_str}.mp4"

    # install a signal handler to call stop acquisition on Ctrl-C
    signal.signal(signal.SIGINT, lambda sig, frame: acquisition.stop_acquisition())

    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(acquisition.start_acquisition(recording_path=filename, max_frames=args.max_frames))
    acquisition.start_acquisition(recording_path=filename, max_frames=args.max_frames)

    acquisition.close()
