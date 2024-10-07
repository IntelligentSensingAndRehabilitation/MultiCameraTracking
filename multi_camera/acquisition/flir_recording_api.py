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
import platform
import psutil


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
    gpio_settings: dict = {},
    chunk_data: list = [],
    camera_info: dict = {},
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
            exposure_time (int): Exposure time in microseconds
            gpio_settings (dict): Dictionary of GPIO settings
            chunk_data (list): List of chunk data to be enabled

        Throughput should be limited for multiple cameras but reduces frame rate. Can use 125000000 for maximum
        frame rate or 85000000 when using more cameras with a 10GigE switch.
    """

    # Initialize each available camera
    c.init()

    print(f"Initializing {c.DeviceSerialNumber}")

    # Resetting binning to 1 to allow for maximum frame size
    c.BinningHorizontal = 1
    c.BinningVertical = 1

    # Ensuring height and width are set to maximum
    c.Width = c.WidthMax
    c.Height = c.HeightMax

    c.ReverseX = False
    c.ReverseY = False

    if camera_info:
        if "flip_image" in camera_info[int(c.DeviceSerialNumber)]:
            flip_image = camera_info[int(c.DeviceSerialNumber)]["flip_image"]

            if flip_image:
                print(f"Flipping image for camera {c.DeviceSerialNumber}")
                c.ReverseX = True
                c.ReverseY = True


    # c.ReverseX = True
    # c.ReverseY = True

    # c.PixelFormat = "BayerBG8"  # BGR8 Mono8

    # c.ReverseX = True
    # c.ReverseY = True
    
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

    line0 = gpio_settings['line0']
    #line1 = gpio_settings['line1'] line1 currently unused
    line2 = gpio_settings['line2']
    line3 = gpio_settings['line3']

    if line2 == '3V3_Enable':
        c.LineSelector = 'Line2'
        c.LineMode = 'Output'
    else:
        if line2 != 'Off':
            print(f"{line2} is not valid for line2. Setting to 'Off'")

    if chunk_data:
        c.ChunkModeActive = True
        for chunk_var in chunk_data:
            c.ChunkSelector = chunk_var
            c.ChunkEnable = True

    if triggering:
        # set up masks for triggering
        c.ActionDeviceKey = 0
        c.ActionGroupKey = 1
        c.ActionGroupMask = 1

        # Check the gpio settings
        if line0 == 'ArduinoTrigger':
            c.TriggerMode = "Off"
            c.TriggerSelector = "FrameStart"
            c.TriggerSource = "Line0"
            c.TriggerActivation = "RisingEdge"
            c.TriggerOverlap = "ReadOut"
            c.TriggerMode = "On"
        else:
            if line0 != 'Off':
                print(f"{line0} is not valid for line0. Setting to 'Off'")
            c.TriggerMode = "Off"
            c.TriggerSelector = "AcquisitionStart"  # Need to select AcquisitionStart for real time clock
            c.TriggerSource = "Action0"
            c.TriggerMode = "On"

        if line3 == 'SerialOn':
            c.SerialPortSelector = "SerialPort0"
            c.SerialPortSource = "Line3"
            c.SerialPortBaudRate = "Baud115200"
            c.SerialPortDataBits = 8
            c.SerialPortStopBits = "Bits1"
            c.SerialPortParity = "None"
        else:
            if line3 != 'Off':
                print(f"{line3} is not valid for line3. Setting to 'Off'")

def write_image_queue(
    vid_file: str, image_queue: Queue, serial, pixel_format: str, acquisition_fps: float, acquisition_type: str, video_segment_len: int
):
    """
    Write images from the queue to a video file

    Args:
        vid_file (str): Path to video file
        image_queue (Queue): Queue to read images from
        serial (str): Camera serial number
        pixel_format (str): Pixel format of the camera
        acquisition_fps (float): Frame rate of camera in Hz
        acquisition_type (str): Type of acquisition (continuous or max_frames)
        video_segment_len (int): Number of frames to write to each video file

    Filename is determined by the vid_file and time_str. The serial number is appended to the end of the filename.

    This is expected to be called from a standalone thread and will automatically terminate when the image_queue is empty.
    """

    timestamps = []
    real_times = []
    frame_spreads = []

    out_video = None

    for frame_num, frame in enumerate(iter(image_queue.get, None)):
        if frame is None:
            break

        timestamps.append(frame["timestamps"])
        real_times.append(frame["real_times"])

        im = frame["im"]

        if pixel_format == "BayerRG8":
            im = cv2.cvtColor(im, cv2.COLOR_BAYER_RG2RGB)
        elif pixel_format == "BayerBG8":
            im = cv2.cvtColor(im, cv2.COLOR_BAYER_BG2RGB)

        # need to collect two frames to track the FPS
        if out_video is None and len(real_times) == 1:
            last_im = im

        elif out_video is None and len(real_times) > 1:
            # Get the video file for the current frame
            vid_file = frame["base_filename"] + f".{serial}.mp4"

            tqdm.write(f"Writing FPS: {acquisition_fps}")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            out_video = cv2.VideoWriter(vid_file, fourcc, acquisition_fps, (im.shape[1], im.shape[0]))
            out_video.write(last_im)

        elif frame_num % video_segment_len == 0 and acquisition_type == "continuous":
            # video_segment_num += 1

            out_video.release()

            # Get the video file for the current frame
            vid_file = frame["base_filename"] + f".{serial}.mp4"

            tqdm.write(f"Writing FPS: {acquisition_fps}")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            print("writing to", vid_file)
            out_video = cv2.VideoWriter(vid_file, fourcc, acquisition_fps, (im.shape[1], im.shape[0]))
            out_video.write(im)

        else:
            out_video.write(im)

            # # Check the timestamp spread between the current frame and previous frame
            # # Skip if either timestamp is 0
            # if timestamps[-1] != 0 and timestamps[-2] != 0:
            #     spread = (timestamps[-1] - timestamps[-2]) * 1e-6
            #     buffer_fps = acquisition_fps * 1.2
            #     if spread > buffer_fps:
            #         print(f"Warning | {serial} Timestamp spread: {spread} {acquisition_fps} {buffer_fps}")
            #         print("Timestamps: ",timestamps[-1], timestamps[-2])
            #         # frame_spreads.append((timestamps[-1] - timestamps[-2]) * 1e-6)

        image_queue.task_done()

    out_video.release()

    # average frame time from ns to s
    ts = np.asarray(timestamps)
    delta = np.mean(np.diff(ts, axis=0)) * 1e-9
    fps = 1.0 / delta

    print(f"{serial}: Finished writing images. Final fps: {fps}")

    # indicate the last None event is handled
    image_queue.task_done()

def calculate_timespread_drift(timestamps):
    # Calculating metrics to determine drift
    ts = pd.DataFrame(timestamps)

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

    return np.max(spread) * 1000

def write_metadata_queue(json_queue: Queue, records_queue: Queue, json_file: str, config_metadata: dict):
    """
    Write metadata from the queue to a json file

    Args:
        json_queue (Queue): Queue to read metadata from
        json_file (str): Path to json file

    This is expected to be called from a standalone thread and will automatically terminate when the json_queue is empty.
    """

    current_filename = json_file

    local_times = []

    json_data = {}
    json_data["real_times"] = []
    json_data["timestamps"] = []
    json_data["frame_id"] = []
    json_data["frame_id_abs"] = []
    json_data["chunk_serial_data"] = []
    json_data["serial_msg"] = []

    for frame_num, frame in enumerate(iter(json_queue.get, None)):
        if frame is None:
            break

        if current_filename != frame["base_filename"]:
            
            # This means a new file should be started
            json_file = current_filename + ".json"

            # Get the camera serial IDs
            json_data["serials"] = frame["camera_serials"]
            json_data["camera_config_hash"] = config_metadata["camera_config_hash"]
            json_data["camera_info"] = config_metadata["camera_info"]
            json_data["meta_info"] = config_metadata["meta_info"]
            json_data["system_info"] = config_metadata["system_info"]
            # Get the current camera settings for each camera before writing
            json_data["exposure_times"] = frame["exposure_times"]
            json_data["frame_rates_requested"] = frame["frame_rates_requested"]
            json_data["frame_rates_binning"] = frame["frame_rates_binning"]


            with open(json_file, "w") as f:
                json.dump(json_data, f)
                f.write("\n")

            max_timespread = calculate_timespread_drift(json_data["timestamps"])

            # add the current filename, max timespread, first of the local_times to the records queue
            records_queue.put({"filename": current_filename, "timestamp_spread": max_timespread, "recording_timestamp": local_times[0]})

            current_filename = frame["base_filename"]

            # reset the json_lists
            json_data = {}
            json_data["real_times"] = [frame["real_times"]]
            local_times = [frame["local_times"]]
            json_data["timestamps"] = [frame["timestamps"]]
            json_data["frame_id"] = [frame["frame_id"]]
            json_data["frame_id_abs"] = [frame["frame_id_abs"]]
            json_data["chunk_serial_data"] = [frame["chunk_serial_data"]]
            json_data["serial_msg"] = [frame["serial_msg"]]

        else:
            # This means we are still writing to the same json file
            json_data["real_times"].append(frame["real_times"])
            local_times.append(frame["local_times"])
            json_data["timestamps"].append(frame["timestamps"])
            json_data["frame_id"].append(frame["frame_id"])
            json_data["frame_id_abs"].append(frame["frame_id_abs"])
            json_data["chunk_serial_data"].append(frame["chunk_serial_data"])
            json_data["serial_msg"].append(frame["serial_msg"])

        json_queue.task_done()

    # write the last json file with the remaining data
    json_file = current_filename + ".json"

    # Get the information from the config file
    json_data["serials"] = frame["camera_serials"]
    json_data["camera_config_hash"] = config_metadata["camera_config_hash"]
    json_data["camera_info"] = config_metadata["camera_info"]
    json_data["meta_info"] = config_metadata["meta_info"]
    json_data["system_info"] = config_metadata["system_info"]
    # Get the current camera settings for each camera before writing
    json_data["exposure_times"] = frame["exposure_times"]
    json_data["frame_rates_requested"] = frame["frame_rates_requested"]
    json_data["frame_rates_binning"] = frame["frame_rates_binning"]

    with open(json_file, "w") as f:
        json.dump(json_data, f)
        f.write("\n")

    max_timespread = calculate_timespread_drift(json_data["timestamps"])

    records_queue.put({"filename": current_filename, "timestamp_spread": max_timespread, "recording_timestamp": local_times[0]})

    json_queue.task_done()



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

        self.pixel_format_conversion = {'BayerRG8': cv2.COLOR_BAYER_RG2RGB, 
                                        'BayerBG8': cv2.COLOR_BAYER_BG2RGB}

    def get_config_hash(self,yaml_content,hash_len=10):

        # Sorting keys to ensure consistent hashing
        file_str = json.dumps(yaml_content,sort_keys=True)
        encoded_config = file_str.encode('utf-8')

        # Create hash of encoded config file and return
        return hashlib.sha256(encoded_config).hexdigest()[:hash_len]
    
    def get_detailed_processor_info(self):
        cpu_info = ""

        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_info = line.split(":")[1].strip()
                        break
        except:
            cpu_info = "CPU information not available for this system"

        return cpu_info
    
    def get_system_info(self):
        info = {
            "system": platform.system(),
            "hostname": platform.node(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": platform.machine(),
            "processor": self.get_detailed_processor_info(),
        }

        return info

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

    def check_queue_sizes(self, queue_dict):
        for name, q in queue_dict.items():
            print(name, q.qsize())

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
            print(self.camera_config)
            # Parse additional parameters from the config file
            exposure_time = self.camera_config["acquisition-settings"]["exposure_time"]
            frame_rate = self.camera_config["acquisition-settings"]["frame_rate"]
            self.gpio_settings = self.camera_config["gpio-settings"]
            self.chunk_data = self.camera_config["acquisition-settings"]["chunk_data"]
            self.camera_info = self.camera_config["camera-info"]

        else:
            # If no config file is passed, use default values
            exposure_time = 15000
            frame_rate = 30
            self.gpio_settings = {'line0': 'Off', 'line2': 'Off', 'line3': 'Off'}
            self.chunk_data = ['FrameID','SerialData']
            self.camera_info = {}

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
            "gpio_settings": self.gpio_settings,
            "chunk_data": self.chunk_data,
            "camera_info": self.camera_info,

        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.cams)) as executor:
            list(executor.map(lambda c: init_camera(c, **config_params), self.cams))

        await self.synchronize_cameras()

        self.cams.sort(key=lambda x: x.DeviceSerialNumber)

        # # Print the pixel format and serial ID for each camera
        # for c in self.cams:
        #     print(f"Camera {c.DeviceSerialNumber}: {c.PixelFormat}")

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

        if self.video_base_file is not None:
            self.video_base_name = self.video_base_file.split("/")[-1]
            self.video_path = "/".join(self.video_base_file.split("/")[:-1])

            # Split the video_base_name to get the root and the date
            # self.video_datetime = "_".join(self.video_base_name.split("_")[-2:])
            self.video_root = "_".join(self.video_base_name.split("_")[:-2])

        

        config_metadata = {}
        if self.camera_config:
            config_metadata["meta_info"] = self.camera_config["meta-info"]
            config_metadata["camera_info"] = self.camera_config["camera-info"]
            camera_config_hash = self.get_config_hash(self.camera_config)
            print("CONFIG HASH",camera_config_hash)
            config_metadata["camera_config_hash"] = camera_config_hash
            acquisition_type = self.camera_config["acquisition-type"]
            video_segment_len = self.camera_config["acquisition-settings"]["video_segment_len"]
        else:
            config_metadata["meta_info"] = "No Config"
            config_metadata["camera_info"] =  [c.DeviceSerialNumber for c in self.cams]
            config_metadata["exposure_times"] = [15000] * len(self.cams)
            config_metadata["frame_rate_requested"] = [30] * len(self.cams)
            config_metadata["camera_config_hash"] = None
            acquisition_type = "max_frames"
            video_segment_len = max_frames

        config_metadata["system_info"] = self.get_system_info()

        # Initializing an image queue for each camera
        self.image_queue_dict = {c.DeviceSerialNumber: Queue(max_frames) for c in self.cams}

        # Initializing a json queue for each camera
        self.json_queue = Queue(max_frames)

        self.records_queue = Queue(max_frames)

        # set up the threads to write videos to disk, if requested
        if self.video_base_file is not None:

            # Start a writing thread for each camera
            for c in self.cams:
                serial = c.DeviceSerialNumber
                threading.Thread(
                    name=f"write_image_{serial}",
                    target=write_image_queue,
                    kwargs={
                        "vid_file": self.video_base_file,
                        "image_queue": self.image_queue_dict[serial],
                        # "json_queue": self.json_queue_dict[serial],
                        "serial": serial,
                        "pixel_format": c.PixelFormat,
                        "acquisition_fps": c.AcquisitionFrameRate,
                        "acquisition_type": acquisition_type,
                        "video_segment_len": video_segment_len,
                    },
                ).start()

            # Start a writing thread for the json queue
            threading.Thread(
                name=f"write_metadata",
                target=write_metadata_queue,
                kwargs={
                    "json_file": self.video_base_file,
                    "json_queue": self.json_queue,
                    "records_queue": self.records_queue,
                    "config_metadata": config_metadata,
                },
            ).start()


        def start_cam(i):
            # this won't truly start them until command is send below
            self.cams[i].start()

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.cams)) as executor:
            l = list(executor.map(start_cam, range(len(self.cams))))

        frame_diff = {}
        prev_timestamp = {}
        timestamp_diff = {}
        initial_timestamp = {}

        print("Acquisition, Resulting, Exposure, DeviceLinkThroughputLimit:")
        for c in self.cams:
            print(f"{c.DeviceSerialNumber}: {c.AcquisitionFrameRate}, {c.AcquisitionResultingFrameRate}, {c.ExposureTime}, {c.DeviceLinkThroughputLimit} ")
            print(f"Frame Size: {c.Width} {c.Height}")

            if self.gpio_settings['line2'] == '3V3_Enable':
                c.LineSelector = 'Line2'
                c.LineMode = 'Input'
                c.V3_3Enable = True
            if self.gpio_settings['line3'] == 'SerialOn':
                print(c.SerialReceiveQueueCurrentCharacterCount)
                print(c.SerialReceiveQueueMaxCharacterCount)
                c.SerialReceiveQueueClear()
                print(c.SerialReceiveQueueCurrentCharacterCount)

            frame_diff[c.DeviceSerialNumber] = 0
            prev_timestamp[c.DeviceSerialNumber] = 0
            timestamp_diff[c.DeviceSerialNumber] = 0
            initial_timestamp[c.DeviceSerialNumber] = 0

        # schedule a command to start in 250 ms in the future
        self.cams[0].TimestampLatch()
        value = self.cams[0].TimestampLatchValue
        latchValue = int(value + 0.250 * 1e9)
        self.iface.TLInterface.GevActionTime.SetValue(latchValue)
        self.iface.TLInterface.GevActionGroupKey.SetValue(1)  # these group/mask/device numbers should match above
        self.iface.TLInterface.GevActionGroupMask.SetValue(1)
        self.iface.TLInterface.GevActionDeviceKey.SetValue(0)
        self.iface.TLInterface.ActionCommand()

        frame_idx = 0

        if acquisition_type == "continuous":
            total_frames = video_segment_len
        else:
            total_frames = max_frames
        
        prog = tqdm(total=total_frames)

        while acquisition_type == "continuous" or frame_idx < max_frames:

            # Get the current real time
            real_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            local_time = datetime.now()

            # Use thread safe checking of semaphore to determine whether to stop recording
            if self.stop_recording.is_set():
                self.stop_recording.clear()
                print("Stopping recording")
                break

            # Update progress for max frame recording
            if acquisition_type == "continuous":

                self.set_progress(frame_idx / total_frames)
                prog.update(1)
                
                # Reset the progress bar after each video segment
                if frame_idx % total_frames == 0:
                    prog = tqdm(total=total_frames)
                    frame_idx = 0

                    if self.video_base_file is not None:
                        # Create a new video_base_filename for the new video segment
                        # video_base_name looks like 'data/t111/20240501/t111_20240501_130531'
                        # we just need to replace the date and time parts of the filename
                        # First get the current date and time
                        now = datetime.now()
                        time_str = now.strftime("%Y%m%d_%H%M%S")

                        # Update the video_base_name with the new time_str
                        self.video_base_name = "_".join([self.video_root, time_str])

                        # Update the video_base_file with the new filename
                        self.video_base_file = os.path.join(self.video_path, self.video_base_name)

            else:
                self.set_progress(frame_idx / max_frames)
                prog.update(1)

            # get the image raw data
            # for each camera, get the current frame and assign it to
            # the corresponding camera
            real_time_images = []

            frame_metadata = {"real_times": real_time, "local_times": local_time, "base_filename": self.video_base_file}

            frame_metadata["timestamps"] = []
            frame_metadata["frame_id"] = []
            frame_metadata["frame_id_abs"] = []
            frame_metadata["chunk_serial_data"] = []
            frame_metadata["serial_msg"] = []
            frame_metadata["camera_serials"] = []
            frame_metadata["exposure_times"] = []
            frame_metadata["frame_rates_requested"] = []
            frame_metadata["frame_rates_binning"] = []

            for c in self.cams:
                im_ref = c.get_image()
                timestamp = im_ref.GetTimeStamp()

                if frame_idx == 0:
                    initial_timestamp[c.DeviceSerialNumber] = timestamp

                chunk_data = im_ref.GetChunkData()
                frame_id = im_ref.GetFrameID()
                frame_id_abs = chunk_data.GetFrameID()

                serial_msg = []
                
                frame_count = -1
                if self.gpio_settings['line3'] == 'SerialOn':
                    # We expect only 5 bytes to be sent
                    if c.ChunkSerialDataLength == 5:
                        chunk_serial_data = c.ChunkSerialData
                        serial_msg = chunk_serial_data
                        split_chunk = [ord(c) for c in chunk_serial_data]

                        # Reconstruct the current count from the chunk serial data
                        frame_count = 0
                        for i, b in enumerate(split_chunk):
                            frame_count |= (b & 0x7F) << (7 * i)

                frame_metadata["timestamps"].append(timestamp)
                frame_metadata["frame_id"].append(frame_id)
                frame_metadata["frame_id_abs"].append(frame_id_abs)
                frame_metadata["chunk_serial_data"].append(frame_count)
                frame_metadata["serial_msg"].append(serial_msg)
                frame_metadata["camera_serials"].append(c.DeviceSerialNumber)
                frame_metadata["exposure_times"].append(15000)
                frame_metadata["frame_rates_binning"].append(c.BinningHorizontal * 30)
                frame_metadata["frame_rates_requested"].append(c.AcquisitionFrameRate)

                curr_frame_diff = (frame_idx+1) - frame_id
                if curr_frame_diff != frame_diff[c.DeviceSerialNumber] and curr_frame_diff != 0:
                    print(f"{c.DeviceSerialNumber}: Frame ID mismatch: {frame_idx + 1} {frame_id}")
                    # self.check_queue_sizes(self.image_queue_dict)
                    frame_diff[c.DeviceSerialNumber] = (frame_idx+1) - frame_id
                    
                    if timestamp != 0 and prev_timestamp[c.DeviceSerialNumber] != 0:
                        cur_timestamp_diff = timestamp - prev_timestamp[c.DeviceSerialNumber]

                        print(f"{c.DeviceSerialNumber}: timestamp diff {cur_timestamp_diff * 1e-6}")

                    print(f"{c.DeviceSerialNumber}: frame_idx based on timestamps {(timestamp - initial_timestamp[c.DeviceSerialNumber]) * 1e-9 * 29.08}")
                    # timestamp_diff[c.DeviceSerialNumber] = cur_timestamp_diff


                # get the data array
                # Using try/except to handle frame tearing
                try:
                    im = im_ref.GetNDArray()
                    im_ref.Release()

                    if self.preview_callback is not None:
                        # if preview is enabled, save the size of the first image
                        # and append the image from each camera to a list
                        real_time_images.append((im,self.pixel_format_conversion[c.PixelFormat]))

                except Exception as e:
                    tqdm.write("Bad frame")
                    continue

                if self.video_base_file is not None:
                    # Writing the frame information for the current camera to its queue
                    self.image_queue_dict[c.DeviceSerialNumber].put(
                        {"im": im, "real_times": real_time, "timestamps": timestamp,  "base_filename": self.video_base_file}
                    )

                prev_timestamp[c.DeviceSerialNumber] = timestamp

            if self.video_base_file is not None:
                # put the frame metadata into the json queue
                self.json_queue.put(frame_metadata)

            if self.preview_callback:
                self.preview_callback(real_time_images)

            frame_idx += 1

        if self.preview_callback:
            self.preview_callback(None)

        print("Finished recording")

        exposure_times = []
        frame_rates = []
        camera_ids = []
        for c in self.cams:
            # Recording the final exposure times and requested frame rates for each camera
            # Actual frame rate can be calculated from the timestamps in the output json
            exposure_times.append(c.ExposureTime)
            frame_rates.append(c.BinningHorizontal * 30)

            camera_ids.append(c.DeviceSerialNumber)

            if self.gpio_settings['line2'] == '3V3_Enable':
                c.LineSelector = 'Line2'
                c.V3_3Enable = False
                c.LineMode = 'Output'
            # Stopping each camera
            c.stop()

        records = []
        if self.video_base_file is not None:
            # stop video writing threads and output json file

            # to allow each queue to be processed before moving on
            for c in self.cams:
                self.image_queue_dict[c.DeviceSerialNumber].put(None)
                self.image_queue_dict[c.DeviceSerialNumber].join()

            # to allow the json queue to be processed before moving on
            self.json_queue.put(None)
            self.json_queue.join()

            # go through the records queue and add the records to a list
            for i in range(self.records_queue.qsize()):
                records.append(self.records_queue.get())
                self.records_queue.task_done()

            self.records_queue.join()

        self.set_status("Idle")

        return records

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
