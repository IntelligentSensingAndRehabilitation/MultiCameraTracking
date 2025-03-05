import PySpin
import simple_pyspin
from simple_pyspin import Camera
import numpy as np
from tqdm import tqdm
import datetime
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

    # c.DeviceLinkThroughputLimit = throughput_limit #94371840 # 106954752 # 
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

        print(f"IN WRITE QUEUE {serial} {frame_num} {frame['base_filename']}")

        if pixel_format == "BayerRG8":
            im = cv2.cvtColor(im, cv2.COLOR_BAYER_RG2RGB)
        elif pixel_format == "BayerBG8":
            im = cv2.cvtColor(im, cv2.COLOR_BAYER_BG2RGB)

        # need to collect two frames to track the FPS
        if out_video is None and len(real_times) == 1:
            last_im = im

        elif out_video is None and len(real_times) > 1:
            # Get the video file for the current frame
            base_filename = frame["base_filename"]
            vid_file = frame["base_filename"] + f".{serial}.mp4"

            vid_writer_print = f"starting new video file {vid_file}"
            print(vid_writer_print)

            tqdm.write(f"Writing FPS: {acquisition_fps}")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            out_video = cv2.VideoWriter(vid_file, fourcc, acquisition_fps, (im.shape[1], im.shape[0]))
            out_video.write(last_im)

        # elif frame['base_filename'] != base_filename:
        #     # This means a new file should be started
        #     print(f"NEW VIDEO START {frame['base_filename']} {base_filename} {frame_num}")
        #     # video_segment_num += 1

        #     out_video.release()

        #     # Get the video file for the current frame
        #     vid_file = frame["base_filename"] + f".{serial}.mp4"
        #     print(f"starting new continuous video file {vid_file} {frame_num}")

        #     tqdm.write(f"Writing FPS: {acquisition_fps}")

        #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        #     print(f"writing to {vid_file}")
        #     out_video = cv2.VideoWriter(vid_file, fourcc, acquisition_fps, (im.shape[1], im.shape[0]))
        #     out_video.write(im)

        else:

            # Check if the base_filename passed is the same as the previous one
            # This means we are still writing to the same video file
            if frame["base_filename"] != base_filename:
                # This means a new file should be started
                # release the previous video file
                print("releasing video file")
                out_video.release()

                base_filename = frame["base_filename"]
                # Get the video file for the current frame
                vid_file = frame["base_filename"] + f".{serial}.mp4"

                print(f"starting new video file {vid_file} {frame_num}")

                tqdm.write(f"Writing FPS: {acquisition_fps}")

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                print(f"writing to {vid_file}")
                out_video = cv2.VideoWriter(vid_file, fourcc, acquisition_fps, (im.shape[1], im.shape[0]))

            out_video.write(im)

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

    max = np.max(spread) * 1000

    # if max is nan or infinity, set to -1
    if np.isnan(max) or np.isinf(max):
        max = -1

    return max

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

    chunk_data = config_metadata["chunk_data"]

    json_data = {}
    json_data["real_times"] = []
    json_data["timestamps"] = []
    json_data["frame_id"] = []

    if chunk_data:
        json_data["frame_id_abs"] = []
        json_data["chunk_serial_data"] = []
        json_data["serial_msg"] = []

    bad_frame = 0

    for frame_num, frame in enumerate(iter(json_queue.get, None)):
        if frame is None:
            break

        if 'first_bad_frame' in frame:
            bad_frame = frame["first_bad_frame"]

        if current_filename != frame["base_filename"]:
            print(f"starting new json file {current_filename}, {frame['base_filename']}")
            
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

            if bad_frame != 0:
                json_data["first_bad_frame"] = bad_frame

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

            if chunk_data:
                json_data["frame_id_abs"] = [frame["frame_id_abs"]]
                json_data["chunk_serial_data"] = [frame["chunk_serial_data"]]
                json_data["serial_msg"] = [frame["serial_msg"]]

        else:
            # This means we are still writing to the same json file
            json_data["real_times"].append(frame["real_times"])
            local_times.append(frame["local_times"])
            json_data["timestamps"].append(frame["timestamps"])
            json_data["frame_id"].append(frame["frame_id"])

            if chunk_data:
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

    if bad_frame != 0:
        json_data["first_bad_frame"] = bad_frame

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
        self.stop_frame_set = threading.Event()

        # Set up thread safe counter for frame count
        self.frame_counter_lock = threading.Lock()
        self.stop_frame = 0

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

            if self.chunk_data:
                print(f"Extracting the following variables from chunk data: {self.chunk_data}")

        else:
            # If no config file is passed, use default values
            exposure_time = 15000
            frame_rate = 30
            self.gpio_settings = {'line0': 'Off', 'line2': 'Off', 'line3': 'Off'}
            self.chunk_data = []
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

    def _prepare_config_metadata(self, max_frames: int) -> dict:
        """Prepare configuration metadata."""
        if self.camera_config:
            self.acquisition_type = self.camera_config["acquisition-type"]
            self.video_segment_len = self.camera_config["acquisition-settings"]["video_segment_len"]
            return {
                "meta_info": self.camera_config["meta-info"],
                "camera_info": self.camera_config["camera-info"],
                "camera_config_hash": self.get_config_hash(self.camera_config),
                "acquisition_type": self.acquisition_type,
                "video_segment_len": self.video_segment_len,
                "system_info": self.get_system_info(),
                "chunk_data": self.chunk_data
            }
        
        self.acquisition_type = "max_frames"
        self.video_segment_len = max_frames
        
        return {
            "meta_info": "No Config",
            "camera_info": [c.DeviceSerialNumber for c in self.cams],
            "exposure_times": [15000] * len(self.cams),
            "frame_rate_requested": [30] * len(self.cams),
            "camera_config_hash": None,
            "acquisition_type": "max_frames",
            "video_segment_len": max_frames,
            "system_info": self.get_system_info(),
            "chunk_data": self.chunk_data
        }
    
    def update_filename(self, current_filename):
        print(f"CURRENT FILENAME: {current_filename}")
        if current_filename is not None:
            
            current_filename = self.calculate_next_filename(current_filename)
            
            print(f"NEW/NEXT FILENAME: {current_filename}")

        return current_filename

    def calculate_next_filename(self, current_filename):

        base_name = current_filename.split("/")[-1]

        # First get the YYYYMMDD_HHMMSS from base_name (which is formatted as {self.video_root}_{YYYYMMDD}_{HHMMSS})
        time_str = "_".join(base_name.split("_")[-2:])

        # Then add the acquisition video segment length / frame_rate to the time_str
        # This is done by converting the time_str to a datetime object, adding the time delta, and then converting it back to a string
        # Get the current datetime object
        time_datetime = datetime.datetime.strptime(time_str, "%Y%m%d_%H%M%S")
        # Calculate the time delta
        time_delta = datetime.timedelta(seconds=round(self.video_segment_len / self.acquisition_frame_rate))

        # Add the time delta to the current datetime object
        new_time_datetime = time_datetime + time_delta

        # Convert the new datetime object back to a string
        new_time_str = new_time_datetime.strftime("%Y%m%d_%H%M%S")

        # Then create the new filename by joining the video_root and the new time_str
        new_filename = "_".join([self.video_root, new_time_str])

        new_file = os.path.join(self.video_path, new_filename)

        return new_file

    def update_progress(self, frame_idx, total_frames):
        self.set_progress(frame_idx / total_frames)

        if self.acquisition_type == "continuous":
            # Reset the progress bar after each video segment
            if frame_idx != 0 and frame_idx % total_frames == 0:
                frame_idx = 0
                # self.update_filename()

        return frame_idx
    
    def initialize_frame_metadata(self):
            
        # Get the current real time
        real_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        local_time = datetime.datetime.now()

        frame_metadata = {"real_times": real_time, "local_times": local_time}

        frame_metadata["timestamps"] = []
        frame_metadata["frame_id"] = []
        frame_metadata["camera_serials"] = []
        frame_metadata["exposure_times"] = []
        frame_metadata["frame_rates_requested"] = []
        frame_metadata["frame_rates_binning"] = []

        if self.chunk_data:
            frame_metadata["frame_id_abs"] = []
            frame_metadata["chunk_serial_data"] = []
            frame_metadata["serial_msg"] = []

        return frame_metadata
    
    def process_serial_data(self, c):
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

        return serial_msg, frame_count
    
    def monitor_frames(self, frame_idx, frame_id, timestamp, camera_serial):
        curr_frame_diff = (frame_idx+1) - frame_id
        if curr_frame_diff != self.frame_diff[camera_serial] and curr_frame_diff != 0:
            print(f"{camera_serial}: Frame ID mismatch | loop: {frame_idx + 1} | cam: {frame_id}")
            print("checking image queue sizes")
            # self.check_queue_sizes(self.image_queue_dict)
            print("checking acquisition queue sizes")
            # self.check_queue_sizes(self.acquisition_queue)
            self.frame_diff[camera_serial] = (frame_idx+1) - frame_id
            
            if timestamp != 0 and self.prev_timestamp[camera_serial] != 0:
                cur_timestamp_diff = timestamp - self.prev_timestamp[camera_serial]

                print(f"{camera_serial}: timestamp diff {cur_timestamp_diff * 1e-6}")

            print(f"{camera_serial}: frame_idx based on timestamps {int((timestamp - self.initial_timestamp[camera_serial]) * 1e-9 * 29.08)}")

            if self.first_bad_frame[camera_serial] == -1:
                self.first_bad_frame[camera_serial] = {'loop_frame_idx': frame_idx, 'cam_frame_id': frame_id}

                self.frame_metadata["first_bad_frame"] = self.first_bad_frame
            # timestamp_diff[c.DeviceSerialNumber] = cur_timestamp_diff

    def increment_frame_counter(self):
        with self.frame_counter_lock:
            self.frame_counter += 1

    def get_frame_count(self):
        with self.frame_counter_lock:
            return self.frame_counter
        
    def set_stop_frame(self, cleanup_frames):
        self.stop_frame = self.get_frame_count() + cleanup_frames
        self.stop_frame_set.set()
        print("stop_frame", self.stop_frame)

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
            print(f"video_root: {self.video_root}")
            print(f"video_path: {self.video_path}")
            print(f"video_base_name: {self.video_base_name}")

        
        config_metadata = self._prepare_config_metadata(max_frames)

        # Set max_frames = self.video_segment_len. self.video_segment_len is either set to max_frames or 
        # a value from the config file.
        max_frames = self.video_segment_len

        # Initializing an image queue for each camera
        self.image_queue_dict = {c.DeviceSerialNumber: Queue(max_frames) for c in self.cams}

        # Initialize intermediate queues for the reading threads
        self.acquisition_queue = {c.DeviceSerialNumber: Queue(max_frames) for c in self.cams}

        # Initializing a json queue for each camera
        self.json_queue = Queue(max_frames)

        self.records_queue = Queue(max_frames)

        # Set up video writing threads if recording is enabled
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
                        "acquisition_type": self.acquisition_type,
                        "video_segment_len": self.video_segment_len,
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

        self.frame_diff = {}
        self.prev_timestamp = {}
        self.timestamp_diff = {}
        self.initial_timestamp = {}
        self.first_bad_frame = {}

        self.cam_serials = []

        print("Acquisition, Resulting, Exposure, DeviceLinkThroughputLimit:")
        for c in self.cams:
            self.acquisition_frame_rate = c.AcquisitionFrameRate
            print(f"{c.DeviceSerialNumber}: {self.acquisition_frame_rate}, {c.AcquisitionResultingFrameRate}, {c.ExposureTime}, {c.DeviceLinkThroughputLimit} ")
            print(f"Frame Size: {c.Width} {c.Height}")
            print(f"{c.GevSCPSPacketSize}, {c.GevSCPD}")

            if self.gpio_settings['line2'] == '3V3_Enable':
                c.LineSelector = 'Line2'
                c.LineMode = 'Input'
                c.V3_3Enable = True
            if self.gpio_settings['line3'] == 'SerialOn':
                print(c.SerialReceiveQueueCurrentCharacterCount)
                print(c.SerialReceiveQueueMaxCharacterCount)
                c.SerialReceiveQueueClear()
                print(c.SerialReceiveQueueCurrentCharacterCount)

            self.frame_diff[c.DeviceSerialNumber] = 0
            self.prev_timestamp[c.DeviceSerialNumber] = 0
            self.timestamp_diff[c.DeviceSerialNumber] = 0
            self.initial_timestamp[c.DeviceSerialNumber] = 0
            self.first_bad_frame[c.DeviceSerialNumber] = -1

            self.cam_serials.append(c.DeviceSerialNumber)

        # if self.video_base_file is not None:
        #     # Create the next filename as well by adding the video_segment_len / acquisition_frame_rate  to the current filename
        #     self.calculate_next_filename()
        #     print(f"video_base_file_next: {self.video_base_file_next}")

        # schedule a command to start in 250 ms in the future
        self.cams[0].TimestampLatch()
        value = self.cams[0].TimestampLatchValue
        latchValue = int(value + 0.250 * 1e9)
        self.iface.TLInterface.GevActionTime.SetValue(latchValue)
        self.iface.TLInterface.GevActionGroupKey.SetValue(1)  # these group/mask/device numbers should match above
        self.iface.TLInterface.GevActionGroupMask.SetValue(1)
        self.iface.TLInterface.GevActionDeviceKey.SetValue(0)
        self.iface.TLInterface.ActionCommand()

        # Function to get image from a single camera
        def get_camera_image(camera):

            camera_serial = camera.DeviceSerialNumber
            pixel_format = camera.PixelFormat
            max_block_count = camera.TransferQueueMaxBlockCount

            frame_idx = 0

            prev_count = 0
            prev_overflow = 0
            prev_frame_id = 0
            processed_frames = 0

            current_filename = self.video_base_file

            while self.acquisition_type == "continuous" or frame_idx < max_frames:

                # print(f"VIDEO BASE FILENAME {camera_serial} {self.video_base_file}")

                if self.stop_frame_set.is_set():
                    # Check if the camera frame count is equal to the stop_frame
                    if frame_idx >= self.stop_frame:

                        print(f"Stopping {camera_serial} recording\n{frame_idx},{self.stop_frame}")
                        self.acquisition_queue[camera_serial].put(None)
                        break

                # Check if a new video segment should be started
                if current_filename is not None and frame_idx % self.video_segment_len == 0 and frame_idx != 0:
                    
                    print(f"Starting new video segment for {camera_serial} at frame {frame_idx}")
                    current_filename = self.update_filename(current_filename)

                try:
                    im_ref = camera.get_image()
                    
                    if im_ref.IsIncomplete():
                        
                        im_stat = im_ref.GetImageStatus()
                        print(f"{camera_serial}: Image incomplete\n{PySpin.Image.GetImageStatusDescription(im_stat)}")
                        im_ref.Release()
                        continue
                    
                    timestamp = im_ref.GetTimeStamp()
                    frame_id = im_ref.GetFrameID()

                    if self.chunk_data:
                        chunk_data = im_ref.GetChunkData()
                        frame_id_abs = chunk_data.GetFrameID()
                        serial_msg, frame_count = self.process_serial_data(camera)

                    # Check if the frame_id has incremented by 1
                    if frame_id != prev_frame_id + 1:
                        # If it has not, print the frame id, previous frame id, and buffer count
                        print(f"{camera_serial}: Frame ID mismatch, current frame id: {frame_id}, previous frame id: {prev_frame_id}")

                    prev_frame_id = frame_id
                    processed_frames += 1

                except Exception as e:
                    print(f"{e}*************{camera_serial}***************************** FAILED TO GET IMAGE ******************************************")
                    time.sleep(0.1)
                    continue

                # Check if the frame is none
                if im_ref is None:
                    print("############################################### IMAGE IS NONE ******************************************")
                    continue
                
                try:
                    im = im_ref.GetNDArray()

                    if current_filename is not None:
                        self.image_queue_dict[camera_serial].put({
                            "im": im,
                            "real_times": 0,
                            "timestamps": timestamp,
                            "base_filename": current_filename
                        })

                    frame_data = {
                        'image': im,
                        'timestamp': timestamp,
                        'frame_id': frame_id,
                        'camera_serial': camera_serial,
                        'pixel_format': pixel_format,
                        'base_filename': current_filename,
                    }

                    if self.chunk_data:
                        frame_data['frame_id_abs'] = frame_id_abs
                        frame_data['serial_msg'] = serial_msg
                        frame_data['frame_count'] = frame_count
                        
                    # put the frame data into the acquisition queue
                    self.acquisition_queue[camera_serial].put(frame_data)

                except Exception as e:
                    print(e)
                    tqdm.write("Bad frame")
                    continue
                finally:
                    # always be sure to release the image reference
                    if im_ref is not None:
                        im_ref.Release()
                    
                    im_ref = None
                    im = None
                    frame_data = None
                
                frame_idx += 1

        def process_synchronized_metadata():
            frame_idx = 0
            self.frame_counter = 0
            cleanup_frames = 10

            while self.acquisition_type == "continuous" or frame_idx < max_frames:

                if self.stop_recording.is_set():
                    
                    # Set the current stop_frame
                    if not self.stop_frame_set.is_set():
                        print("setting stop frame")
                        self.set_stop_frame(cleanup_frames)
                    else:
                        print("cleaning_up", frame_idx, self.stop_frame, max_frames)
                        # print(f"{self.cam_serials}\n{[self.acquisition_queue[c].empty() for c in self.cam_serials]}")

                        # break out if frame_idx == stop_frame
                        if frame_idx == self.stop_frame:
                            print("exiting metadata loop")
                            break

                empty_queues = [self.acquisition_queue[c].empty() for c in self.cam_serials]
                # Check if all acquisition queues have at least one item
                if any(empty_queues):
                    time.sleep(0.1)
                    continue

                # Wait until all queues have at least one item
                # acquisition_frames = [self.acquisition_queue[c].get() for c in self.cam_serials] 
                acquisition_frames = []
                for c in self.cam_serials:
                    acquisition_frames.append(self.acquisition_queue[c].get())
                    self.acquisition_queue[c].task_done()

                frame_idx = self.update_progress(frame_idx, max_frames)
                self.increment_frame_counter()

                real_time_images = []
                self.frame_metadata = self.initialize_frame_metadata()

                for frame_data in acquisition_frames:

                    camera_serial = frame_data['camera_serial']
                    
                    self.frame_metadata['timestamps'].append(frame_data['timestamp'])
                    self.frame_metadata['frame_id'].append(frame_data['frame_id'])
                    self.frame_metadata['camera_serials'].append(camera_serial)
                    self.frame_metadata['exposure_times'].append(15000)
                    self.frame_metadata['frame_rates_binning'].append(30)
                    self.frame_metadata['frame_rates_requested'].append(30)

                    if self.chunk_data:
                        self.frame_metadata['frame_id_abs'].append(frame_data['frame_id_abs'])
                        self.frame_metadata['chunk_serial_data'].append(frame_data['frame_count'])
                        self.frame_metadata['serial_msg'].append(frame_data['serial_msg'])

                    if self.preview_callback:
                        real_time_images.append((frame_data['image'],self.pixel_format_conversion[frame_data['pixel_format']]))

                    if frame_idx == 0:
                        self.initial_timestamp[camera_serial] = frame_data['timestamp']
                    
                    # self.monitor_frames( 
                    #     frame_idx,
                    #     frame_data['frame_id'],
                    #     frame_data['timestamp'],
                    #     camera_serial
                    # )

                    self.prev_timestamp[camera_serial] = frame_data['timestamp']

                # Put the frame metadata into the json queue
                if self.video_base_file is not None:
                    self.frame_metadata['base_filename'] = frame_data['base_filename']
                    self.json_queue.put(self.frame_metadata)

                # Handle preview callback
                if self.preview_callback: 
                    self.preview_callback(real_time_images)

                frame_idx += 1

        # Start threads for acquisition
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.cams)) as camera_executor, \
            concurrent.futures.ThreadPoolExecutor(max_workers=1) as metadata_executor:
            
            # Start all camera captures in parallel
            future_to_camera = {camera_executor.submit(get_camera_image, camera): camera for camera in self.cams}

            # Start metadata processing in parallel
            metadata_future = metadata_executor.submit(process_synchronized_metadata)

        if self.preview_callback:
            self.preview_callback(None)

        self.stop_frame_set.clear()
        self.stop_recording.clear()
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
    now = datetime.datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S")
    filename = f"{args.vid_file}_{time_str}.mp4"

    # install a signal handler to call stop acquisition on Ctrl-C
    signal.signal(signal.SIGINT, lambda sig, frame: acquisition.stop_acquisition())

    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(acquisition.start_acquisition(recording_path=filename, max_frames=args.max_frames))
    acquisition.start_acquisition(recording_path=filename, max_frames=args.max_frames)

    acquisition.close()
