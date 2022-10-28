import copy

import PySpin
import simple_pyspin
from simple_pyspin import Camera
from PIL import Image
import numpy as np
from tqdm import tqdm
from datetime import datetime
from queue import Queue
import concurrent.futures
import threading
import curses
import json
import time
import cv2
import os
import _thread
import yaml

# Defining window size based on number
# of cameras(key)
window_sizes = {
    1: np.array([1, 1, 1]),
    2: np.array([1, 2, 1]),
    3: np.array([2, 2, 1]),
    4: np.array([2, 2, 1]),
    5: np.array([2, 3, 1]),
    6: np.array([2, 3, 1]),
    7: np.array([2, 4, 1]),
    8: np.array([2, 4, 1]),
}


def record_dual(vid_file, max_frames=100, num_cams=4, preview=True, resize=0.5, config=""):
    # Initializing dict to hold each image queue (from each camera)
    image_queue_dict = {}
    if preview:
        visualization_queue = Queue(1)

    system = PySpin.System.GetInstance()
    iface_list = system.GetInterfaces()

    # Check if a config file has been provided
    if config != "":
        with open(config, "r") as file:
            camera_config = yaml.safe_load(file)
            # Updating interface_cameras if a config file is passed
            # with the camera IDs passed
            iface_cameras = list(camera_config["camera-info"].keys())
    else:
        iface_cameras = num_cams

    def select_interface(interface,cameras):
        # This method takes in an interface and list of cameras (if a config
        # file is provided) or number of cameras. It checks if the current
        # interface has cameras and returns a list of valid camera IDs or
        # number of cameras

        # Check the current interface to see if it has cameras
        interface_cams = interface.GetCameras()
        # Get the number of cameras on the current interface
        num_interface_cams = interface_cams.GetSize()

        if num_interface_cams > 0:
            # If camera list is passed, confirm all SNs are valid
            if isinstance(cameras,list):
                camera_id_list = [str(c) for c in cameras if interface_cams.GetBySerial(str(c)).IsValid()]

                invalid_ids = [c for c in cameras if str(c) not in camera_id_list]

                if invalid_ids:
                    # if len(camera_id_list) != len(cameras):
                    print(f'The following camera ID(s) from {config} are invalid: {invalid_ids}')

                return camera_id_list
            # If num_cams is passed, confirm it is less than size
            # of interface_cams and return the correct num_cams
            if isinstance(cameras,int):
                # Set num_cams to the # of available cameras
                # if input arg is larger (num_cams = min(num_cams,len(camera_list))
                num_cams = min(cameras, num_interface_cams)
                print(f"No config file passed. Selecting the first {num_cams} cameras in the list.")

                return num_cams
        # If there are no cameras on the interface, return None
        return None

    # Identify the interface we are going to send a command for synchronous recording
    iface_idx = []
    for i,current_iface in enumerate(iface_list):
        current_iface_cams = select_interface(current_iface,iface_cameras)

        # If the value returned from select_interface is not None,
        # select the current interface
        if current_iface_cams is not None:
            iface_idx.append(i)
            iface = current_iface
            iface_cams = current_iface_cams

    # Confirm that cameras were only found on 1 interface
    assert len(iface_idx) == 1, "Unable to automatically pick interface as cameras found on multiple"

    iface.TLInterface.GevActionDeviceKey.SetValue(0)
    iface.TLInterface.GevActionGroupKey.SetValue(1)
    iface.TLInterface.GevActionGroupMask.SetValue(1)

    if config != "":
        # if config is passed then use the config list
        # of cameras to select
        cams = [Camera(i, lock=True) for i in iface_cams]
    else:
        # otherwise just select the first num_cams cameras
        cams = [Camera(i, lock=True) for i in range(iface_cams)]

    def init_camera(c):
        # Initialize each available camera
        c.init()

        c.PixelFormat = "BayerRG8"  # BGR8 Mono8
        # c.BinningHorizontal = 1
        # c.BinningVertical = 1

        if False:
            c.GainAuto = "Continuous"
            c.ExposureAuto = "Continuous"
            # c.IspEnable = True

        c.GevSCPSPacketSize = 9000
        if num_cams > 2:
            c.DeviceLinkThroughputLimit = 85000000
            c.GevSCPD = 25000
        else:
            c.DeviceLinkThroughputLimit = 125000000
            c.GevSCPD = 25000
        # c.StreamPacketResendEnable = True

        # set up masks for triggering
        c.ActionDeviceKey = 0
        c.ActionGroupKey = 1
        c.ActionGroupMask = 1

        # set up trigger setting
        c.TriggerMode = 'Off'
        c.TriggerSelector = 'AcquisitionStart'   # Need to select AcquisitionStart for real time clock
        c.TriggerSource = 'Action0'
        c.TriggerMode = 'On'

        # Initializing an image queue for each camera
        image_queue_dict[c.DeviceSerialNumber] = Queue(max_frames)

        print(
            c.DeviceSerialNumber,
            c.PixelSize,
            c.PixelColorFilter,
            c.PixelFormat,
            c.Width,
            c.Height,
            c.WidthMax,
            c.HeightMax,
            c.BinningHorizontal,
            c.BinningVertical,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cams)) as executor:
        l = list(executor.map(init_camera, cams))

    cams.sort(key=lambda x: x.DeviceSerialNumber)

    # print(cams[0].get_info('PixelFormat'))
    pixel_format = cams[0].PixelFormat

    if not all([c.GevIEEE1588 for c in cams]):
        print("Cameras not synchronized. Enabling IEEE1588 (takes 10 seconds)")
        for c in cams:
            c.GevIEEE1588 = True

        time.sleep(10)

    for c in cams:
        c.GevIEEE1588DataSetLatch()
        print('Primary' if c.GevIEEE1588StatusLatched == 'Master' else 'Secondary', c.GevIEEE1588OffsetFromMasterLatched)

    def acquire():

        def start_cam(i):
            # this won't truly start them until command is send below
            cams[i].start()

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(cams)) as executor:
            l = list(executor.map(start_cam, range(len(cams))))

        # schedule a command to start in 250 ms in the future
        cams[0].TimestampLatch()
        value = cams[0].TimestampLatchValue
        latchValue = int(value + 0.250 * 1e9)
        iface.TLInterface.GevActionTime.SetValue(latchValue)
        iface.TLInterface.GevActionGroupKey.SetValue(1)   # these group/mask/device numbers should match above
        iface.TLInterface.GevActionGroupMask.SetValue(1)
        iface.TLInterface.GevActionDeviceKey.SetValue(0)
        iface.TLInterface.ActionCommand()

        try:
            for _ in tqdm(range(max_frames)):

                # get the image raw data
                # for each camera, get the current frame and assign it to
                # the corresponding camera
                real_times = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                size_flag = 0
                real_time_images = []
                for c in cams:
                    im = c.get_image()
                    timestamps = im.GetTimeStamp()

                    # get the data array
                    # Using try/except to handle frame tearing
                    try:
                        im = im.GetNDArray()

                        if preview:
                            # if preview is enabled, save the size of the first image
                            # and append the image from each camera to a list
                            real_time_images.append(im)

                    except Exception as e:
                        # print(e)
                        tqdm.write("Bad frame")
                        continue

                    # Writing the frame information for the current camera to its queue
                    image_queue_dict[c.DeviceSerialNumber].put(
                        {"im": im, "real_times": real_times, "timestamps": timestamps}
                    )

                if preview:
                    # Add image list to queue if empty
                    if visualization_queue.empty():
                        visualization_queue.put({"im": real_time_images}, block=False)

        except KeyboardInterrupt:
            tqdm.write("Ctrl-C detected")

        for c in cams:
            c.stop()

            image_queue_dict[c.DeviceSerialNumber].put(None)

        if preview:
            visualization_queue.put(None, block=True)

    def visualize(image_queue):

        # Getting the image list from the queue
        for frame in iter(image_queue.get, None):

            if frame is None:
                cv2.destroyAllWindows()
                return

            real_time_images = frame["im"]

            preview_images = []

            for i in real_time_images:

                # Go through the list of images, and convert them all to color
                preview_im = cv2.cvtColor(i, cv2.COLOR_BAYER_RG2RGB)

                # Check if the image should be resized and resize accordingly
                if (0.0 < resize <= 1.0) and isinstance(resize, float):
                    preview_im = cv2.resize(preview_im, dsize=None, fx=resize, fy=resize)

                # Add the color(/resized) images to a new list
                preview_images.append(preview_im)

            if len(preview_images) < np.prod(window_sizes[num_cams]):
                # Add extra square to fill in empty space if there are
                # not enough images to fit the current grid size
                preview_images.extend(
                    [
                        np.zeros(preview_images[0].shape, dtype=np.uint8)
                        for i in range(np.prod(window_sizes[num_cams]) - len(preview_images))
                    ]
                )

            h, w, d = preview_images[0].shape

            # Initializing the full output grid
            preview = np.zeros(np.array(preview_images[0].shape) * window_sizes[num_cams], dtype=np.uint8)

            # removing padding code for now, making assumption that all cameras
            # will have same sized images

            # Filling in the full grid with individual images
            im_counter = 0
            w_offset = 0
            h_offset = 0
            for r in range(window_sizes[num_cams][0]):
                for c in range(window_sizes[num_cams][1]):
                    preview[h_offset : h_offset + h, w_offset : w_offset + w] = preview_images[im_counter]
                    im_counter += 1
                    w_offset += w

                h_offset += h
                w_offset = 0

            # Display the preview image
            cv2.imshow("Preview", preview)
            cv2.waitKey(1)

    def write_queue(vid_file, image_queue, json_queue, serial):
        now = datetime.now()
        time_str = now.strftime("%Y%m%d_%H%M%S")
        vid_file = os.path.splitext(vid_file)[0] + f"_{time_str}.{serial}.mp4"

        print(vid_file)

        timestamps = []
        real_times = []

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

                ts = np.asarray(timestamps)
                delta = np.mean(np.diff(ts, axis=0)) * 1e-9
                fps = 1.0 / delta
                tqdm.write(f"Computed FPS: {fps}")

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_video = cv2.VideoWriter(vid_file, fourcc, fps, (im.shape[1], im.shape[0]))
                out_video.write(last_im)

            else:
                out_video.write(im)

            image_queue.task_done()

        out_video.release()

        # Adding the json info corresponding to the current camera to its own queue
        json_queue.put({"serial": serial, "timestamps": timestamps, "real_times": real_times, "time_str": time_str})

        # average frame time from ns to s
        ts = np.asarray(timestamps)
        delta = np.mean(np.diff(ts, axis=0)) * 1e-9
        fps = 1.0 / delta

        print(f"Finished writing images. Final fps: {fps}")

        # indicate the last None event is handled
        image_queue.task_done()

    # initializing dictionary to hold json queue for each camera
    json_queue = {}
    # Start a writing thread for each camera
    for c in cams:
        serial = c.DeviceSerialNumber
        # Initializing queue to store json info for each camera
        json_queue[c.DeviceSerialNumber] = Queue(max_frames)
        threading.Thread(
            target=write_queue,
            kwargs={
                "vid_file": vid_file,
                "image_queue": image_queue_dict[serial],
                "json_queue": json_queue[c.DeviceSerialNumber],
                "serial": serial,
            },
        ).start()

    if preview:
        import pynput

        # Starting a daemon thread that hosts the OpenCV visualization (cv2.imshow())
        threading.Thread(target=visualize, kwargs={"image_queue": visualization_queue}, daemon=visualize).start()

        # Defining method to listen to keyboard input
        def on_press(key):
            if (
                key == pynput.keyboard.Key.esc
                or key == pynput.keyboard.KeyCode.from_char("q")
                or key == pynput.keyboard.KeyCode.from_char("c")
            ):
                # Stop listener
                _thread.interrupt_main()
                return False

        # Collect events until released
        listener = pynput.keyboard.Listener(on_press=on_press, suppress=True)
        listener.start()

    acquire()

    # Joining the image queues for each camera
    # to allow each queue to be processed before moving on
    for c in cams:
        image_queue_dict[c.DeviceSerialNumber].join()

    # Creating a dictionary to hold the contents of each camera's json queue
    output_json = {}
    all_json = {}

    for j in json_queue:
        time_str = json_queue[j].queue[0]["time_str"]
        real_times = json_queue[j].queue[0]["real_times"]

        all_json[json_queue[j].queue[0]["serial"]] = json_queue[j].queue[0]

    # defining the filename for the json file
    json_file = os.path.splitext(vid_file)[0] + f"_{time_str}.json"

    # combining the json information from each camera's queue
    all_serials = [all_json[key]["serial"] for key in all_json]
    all_timestamps = [all_json[key]["timestamps"] for key in all_json]

    output_json["serials"] = all_serials
    output_json["timestamps"] = [list(t) for t in zip(*all_timestamps)]
    output_json["real_times"] = real_times

    if config != "":
        output_json["meta_info"] = camera_config["meta-info"]
        output_json["camera_info"] = camera_config["camera-info"]

    # writing the json file for the current recording session
    json.dump(output_json, open(json_file, "w"))

    ts = np.array(output_json["timestamps"])
    dt = (ts - ts[0, 0]) / 1e9
    spread = np.max(dt, axis=1) - np.min(dt, axis=1)
    if np.all(spread < 1e-6):
        print('Timestamps well aligned and clean')
    else:
        print(f'Timestamps showed a maximum spread of {np.max(spread) * 1000} ms')

    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record video from GigE FLIR cameras")
    parser.add_argument("vid_file", help="Video file to write")
    parser.add_argument("-m", "--max_frames", type=int, default=10000, help="Maximum frames to record")
    parser.add_argument("-n", "--num_cams", type=int, default=4, help="Number of input cameras")
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

    record_dual(
        vid_file=args.vid_file,
        max_frames=args.max_frames,
        num_cams=args.num_cams,
        preview=args.preview,
        resize=args.scaling,
        config=args.config,
    )
