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

    # Get the available cameras
    camera_list = simple_pyspin.list_cameras()

    # Check if a config file has been provided
    if config != "":
        with open(config, "r") as file:
            camera_config = yaml.safe_load(file)
        print(f"Selecting cameras defined in {config}.")
        # First create list of all available cameras
        cams = [Camera(i, lock=True) for i in range(camera_list.GetSize())]

    else:
        # Set num_cams to the # of available cameras
        # if input arg is larger (num_cams = min(num_cams,len(camera_list))
        num_cams = min(num_cams,len(camera_list))
        print(f"No config file passed. Selecting the first {num_cams} cameras in the list.")
        # First create list of first n cameras in camera_list where n=num_cams
        cams = [Camera(i, lock=True) for i in range(num_cams)]

    def init_camera(c):
        # Initialize each available camera
        c.init()

        # check if the current camera is in the list defined by config
        if config != "":
            if int(c.DeviceSerialNumber) not in camera_config["camera-info"].keys():
                print(f"{c.DeviceSerialNumber} not listed in config file.")
                return

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

        return c

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cams)) as executor:
        l = list(executor.map(init_camera, cams))

    # Creating a list of only cameras present in the config file
    cams = [cam for cam in l if cam != None]

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
        print(c.GevIEEE1588StatusLatched, c.GevIEEE1588OffsetFromMasterLatched)

    def acquire():
        def start_cam(i):
            cams[i].start()

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(cams)) as executor:
            l = list(executor.map(start_cam, range(len(cams))))

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
            tqdm.write("Crtl-C detected")

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
    if np.all(spread < 4e-3):
        print("Timestamps well aligned and clean")
    else:
        print(f"Timestamps showed a maximum spread of {np.max(spread) * 1000} ms")

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
