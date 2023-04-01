from fastapi import FastAPI, Request, HTTPException, APIRouter, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.applications import Starlette
from uvicorn.main import Server
from pathlib import Path
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List
from typing_extensions import Annotated
import concurrent.futures
import numpy as np
import logging
import socketio
import signal
import math
import datetime
import cv2
import os
import asyncio


# file templates directory, which is located relative to this file location
templates = os.path.split(__file__)[0]
templates = os.path.join(templates, "templates")
templates = Jinja2Templates(directory=templates)

logger = logging.getLogger("uvicorn.server")

# sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
# app_asgi = socketio.ASGIApp(sio, app)
preview_queue = asyncio.Queue()

# Replace this with your actual acquisition library import
from multi_camera.acquisition.flir_recording_api import FlirRecorder, CameraStatus

RECORDING_BASE = "data"
CONFIG_PATH = "/home/cbm/MultiCameraTracking/multi_camera_configs/"
DEFAULT_CONFIG = os.path.join(CONFIG_PATH, "cotton_lab_config_20221109.yaml")

acquisition = None
camera_status = []

# Create a thread pool executor with 1 thread
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
loop = asyncio.get_event_loop()

recording_status = ""
recording_status_queue = asyncio.Queue()


def receive_status(status):
    global recording_status
    recording_status = status

    print(f"Status: {status}")
    # Put the status in the queue using asyncio from a synchronous function
    loop.call_soon_threadsafe(recording_status_queue.put_nowait, {"status": status})


# @asynccontextmanager
async def lifespan(app: FastAPI):
    global acquisition
    global camera_status

    # Perform startup tasks
    acquisition = FlirRecorder(receive_status)
    # camera_status = acquisition.configure_cameras(DEFAULT_CONFIG)

    yield

    # Perform shutdown tasks
    acquisition.close()
    logger.info("Acquisition system closed")


app = FastAPI(lifespan=lifespan)
api_router = APIRouter(prefix="/api/v1")

# ugly patch, but the websocket user space code doesn't know when
# they are closed (even though they are in response to sigterm)
# so we need them to monitor this flag to exit
original_handler = Server.handle_exit


class AppStatus:
    should_exit = False

    @staticmethod
    def handle_exit(*args, **kwargs):
        print("Exiting!!!")
        AppStatus.should_exit = True
        original_handler(*args, **kwargs)


Server.handle_exit = AppStatus.handle_exit

# Add a middleware to handle CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})


class NewTrialData(BaseModel):
    recording_dir: str
    recording_filename: str
    comment: str


class ChannelData(BaseModel):
    value: int


class ConfigFileData(BaseModel):
    config: str


class PriorRecordings(BaseModel):
    filename: str
    comment: str


prior_recordings = []


@api_router.post("/set_channel")
async def set_channel(data: ChannelData):
    global channel_selector_value
    channel_selector_value = data.value


@api_router.get("/camera_status")
async def get_camera_status():
    camera_status = acquisition.get_camera_status()
    return camera_status


@api_router.post("/stop")
async def stop_recording():
    acquisition.stop_acquisition()
    return {"status": "recording_stopped"}


@api_router.post("/set_channel")
async def set_channel(value: int):
    global channel_selector_value
    channel_selector_value = value
    print(f"Setting channel to {value}")
    return {"status": "channel_set"}


@api_router.post("/new_session")
async def new_session(subject_id: str):
    date = datetime.date.today().strftime("%Y%m%d")
    session_dir = os.path.join(RECORDING_BASE, subject_id, date)
    os.makedirs(session_dir, exist_ok=True)
    return {"recording_dir": session_dir, "recording_filename": f"{subject_id}"}


@api_router.post("/new_trial")
async def new_trial(data: NewTrialData):
    recording_dir = data.recording_dir
    recording_filename = data.recording_filename
    comment = data.comment

    # Build the recording file name from the components
    now = datetime.datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S")
    recording_path = os.path.join(recording_dir, f"{recording_filename}_{time_str}")

    def receive_frames_wrapper(frames):
        loop.create_task(receive_frames(frames))

    # run acquisition in a separate thread
    import threading
    from functools import partial

    start_acquisition = partial(
        acquisition.start_acquisition, recording_path=recording_path, preview_callback=receive_frames_wrapper
    )
    threading.Thread(target=start_acquisition).start()

    # add this recording entry to the list of prior recordings
    prior_recordings.append(PriorRecordings(filename=recording_path, comment=comment))

    return {"status": "Recording Started", "recording_file_name": recording_path}


@api_router.post("/preview")
async def preview():
    def receive_frames_wrapper(frames):
        loop.create_task(receive_frames(frames))

    # run acquisition in a separate thread
    import threading
    from functools import partial

    start_acquisition = partial(acquisition.start_acquisition, preview_callback=receive_frames_wrapper)
    threading.Thread(target=start_acquisition).start()

    return {"status": "Preview Started"}


@api_router.post("/stop")
async def stop():
    acquisition.stop_acquisition()
    return {"status": "Acquisition Stopped"}


@api_router.get("/prior_recordings", response_model=List[PriorRecordings])
async def get_prior_recordings() -> List[PriorRecordings]:
    return prior_recordings


@api_router.get("/configs")
async def get_configs():
    config_files = os.listdir(CONFIG_PATH)
    config_files = [""] + [f for f in config_files if f.endswith(".yaml")]
    return JSONResponse(content=config_files)


@api_router.get("/current_config", response_model=str)
async def get_current_config() -> str:
    config = acquisition.config_file
    if config is None:
        return ""
    return os.path.split(config)[-1]


@api_router.post("/update_config")
async def update_config(config: ConfigFileData):
    print("Received config: ", config.config)
    acquisition.configure_cameras(os.path.join(CONFIG_PATH, config.config))
    return {"status": "success", "config": config.config}


@api_router.post("/reset_cameras")
async def reset_cameras():
    acquisition.reset_cameras()
    return {"status": "success"}


# create an endpoint that exposes the camera statuses
@api_router.get("/camera_status")
async def get_camera_status() -> List[CameraStatus]:
    camera_status = acquisition.get_camera_status()
    return camera_status


@api_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("Websocket request received")
    await websocket.accept()
    print("Websocket connected")
    try:
        while not AppStatus.should_exit:
            try:
                status = await asyncio.wait_for(recording_status_queue.get(), timeout=0.5)
                print("Sending status: ", status)
                await websocket.send_json(status)
            except asyncio.TimeoutError:
                # need this to monitor for exit
                pass
        print("Exit flag detected")
    except WebSocketDisconnect:
        print("Websocket disconnected")


@api_router.get("/recording_status", response_model=str)
async def get_recording_status() -> str:
    return recording_status


def downsample_image(image, new_width, new_height):
    return cv2.resize(image, (new_width, new_height))


def convert_rgb_to_jpeg(frame):
    ret, jpeg_image = cv2.imencode(".jpg", frame)
    return jpeg_image.tobytes()


async def receive_frames(frames):
    if not preview_queue.empty():
        # If the queue is not empty, then we are not keeping up with the frames
        logger.warn("Dropping frame")
        return

    num_frames = len(frames)
    grid_width = math.ceil(math.sqrt(num_frames))
    grid_height = math.ceil(num_frames / grid_width)

    # Convert each frame to RGB and store in a list
    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BAYER_RG2RGB) for frame in frames]

    # Calculate the size of each frame to fit in the grid
    frame_height, frame_width, _ = rgb_frames[0].shape
    grid_frame_width = frame_width // grid_width
    grid_frame_height = frame_height // grid_height

    # Initialize an empty grid image
    grid_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Fill the grid with the frames
    for i, frame in enumerate(rgb_frames):
        row = i // grid_width
        col = i % grid_width

        # Resize the frame to fit the grid
        resized_frame = cv2.resize(frame, (grid_frame_width, grid_frame_height))

        # Calculate the position in the grid
        y_start = row * grid_frame_height
        y_end = y_start + grid_frame_height
        x_start = col * grid_frame_width
        x_end = x_start + grid_frame_width

        # Place the resized frame in the grid
        grid_image[y_start:y_end, x_start:x_end] = resized_frame

    # preserve aspect ratio of original images and make the total height 480 pixels
    frame_ratio = frame_width / frame_height
    grid_ratio = grid_width / grid_height
    ratio = frame_ratio * grid_ratio
    width = 1080

    downsampled_frame = downsample_image(grid_image, int(ratio * width), width)
    jpeg_image = convert_rgb_to_jpeg(downsampled_frame)

    print("Putting frame on queue")

    await preview_queue.put(jpeg_image)


# @api_router.get("/video")
async def video_endpoint():
    async def generate_frames():
        while True:
            # Wait for the next frame to become available
            try:
                frame = await asyncio.wait_for(preview_queue.get(), timeout=2.5)
                print("Received JPEG")
            except asyncio.TimeoutError:
                # TODO: if uvicorn lifecycle let us check a flag knowing that shutdown was in process then
                # we could exit here. Not really possble right now, though.
                continue
            except asyncio.exceptions.CancelledError:
                print("Disconnecting generator as client disconnected.")
                return
            if frame is None:
                break

            # Write the boundary frame to the response
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")

    return StreamingResponse(generate_frames(), status_code=206, media_type="multipart/x-mixed-replace; boundary=frame")


@api_router.websocket("/video_ws")
async def video_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Video Websocket connected")
    try:
        while not AppStatus.should_exit:
            try:
                frame = await asyncio.wait_for(preview_queue.get(), timeout=2.5)
                print("Sending frame")
                if frame is None:
                    break
                await websocket.send_bytes(frame)
            except asyncio.TimeoutError:
                print("No frame")
    except WebSocketDisconnect:
        logger.info("Websocket disconnected")


app.include_router(api_router)


def websocket_test():
    # web socket testing code
    import aiohttp
    import asyncio
    import threading

    async def socket_test():
        await asyncio.sleep(10)
        print("Testing")
        conn = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=conn) as session:
            print("Connected to server")
            async with session.ws_connect("ws://localhost:8000/api/v1/ws") as ws:
                print("Connected to server WebSocket")
                # await for messages and send messages
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        print(f"SERVER says - {msg.data}")
                        text = input("Enter a message: ")
                        await ws.send_str(text)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        break

    def wrapper():
        asyncio.run(socket_test())

    # run websocket_test in a new thread
    threading.Thread(target=wrapper).start()


if __name__ == "__main__":
    import uvicorn

    # Start the server
    uvicorn.run(
        "multi_camera.acquisition.fastapi_gui.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="trace",
    )
