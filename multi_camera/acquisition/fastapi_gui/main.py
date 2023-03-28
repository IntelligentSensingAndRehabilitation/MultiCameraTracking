from fastapi import FastAPI, Request, HTTPException, APIRouter, Body
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
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

app = FastAPI()
api_router = APIRouter(prefix="/api/v1")

if True:
    templates = Jinja2Templates(directory="templates")
else:
    # Possibly in future move to react frontend, but for now just use the static files
    app.mount("/static", StaticFiles(directory="multi-camera-video-acquisition/build/static"), name="static")

    @app.get("/")
    async def serve_frontend():
        build_dir = Path("multi-camera-video-acquisition/build")
        index_html = build_dir / "index.html"
        if not index_html.is_file():
            raise HTTPException(status_code=404, detail="Index file not found")
        return FileResponse(str(index_html), media_type="text/html")


logger = logging.getLogger("uvicorn.server")

# sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
# app_asgi = socketio.ASGIApp(sio, app)
preview_queue = asyncio.Queue()

# Replace this with your actual acquisition library import
from multi_camera.acquisition.flir_recording_api import FlirRecorder, CameraStatus

RECORDING_BASE = "data"
CONFIG_PATH = "/home/cbm/MultiCameraTracking/multi_camera_configs/"
DEFAULT_CONFIG = os.path.join(CONFIG_PATH, "cotton_lab_config_20221109.yaml")

acquisition = FlirRecorder()
camera_status = acquisition.configure_cameras(DEFAULT_CONFIG)

# Create a thread pool executor with 1 thread
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
loop = asyncio.get_event_loop()

channel_selector_value = 2


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
        if frames is not None:
            loop.create_task(receive_frames(frames))

    routine = acquisition.start_acquisition(recording_path=recording_path, preview_callback=receive_frames_wrapper)

    loop = asyncio.get_event_loop()
    loop.create_task(routine)

    # add this recording entry to the list of prior recordings
    prior_recordings.append(PriorRecordings(filename=recording_path, comment=comment))

    return {"status": "Recording Started", "recording_file_name": recording_path}


@api_router.get("/prior_recordings")
async def get_prior_recordings() -> List[PriorRecordings]:
    return prior_recordings


@api_router.get("/configs")
async def get_configs():
    config_files = os.listdir(CONFIG_PATH)
    config_files = [f for f in config_files if f.endswith(".yaml")]
    return JSONResponse(content=config_files)


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


@api_router.get("/recordings")
async def get_recordings():
    recordings = []
    for root, dirs, files in os.walk("./data"):
        for file in files:
            if file.endswith(".avi"):
                recording_path = os.path.join(root, file)
                participant, date = os.path.basename(os.path.dirname(recording_path)).split("/")
                comment_path = recording_path[:-4] + ".txt"
                comment = ""
                if os.path.exists(comment_path):
                    with open(comment_path) as f:
                        comment = f.read()
                recordings.append({"participant": participant, "filename": file, "comment": comment})
    return JSONResponse(content=recordings)


def downsample_image(image, new_width, new_height):
    return cv2.resize(image, (new_width, new_height))


def convert_rgb_to_jpeg(frame):
    ret, jpeg_image = cv2.imencode(".jpg", frame)
    return jpeg_image.tobytes()


async def receive_frames(frames):
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
    width = 480

    downsampled_frame = downsample_image(grid_image, int(ratio * width), width)
    jpeg_image = convert_rgb_to_jpeg(downsampled_frame)

    await preview_queue.put(jpeg_image)


@api_router.get("/video")
async def video_endpoint():
    async def generate_frames():
        while True:
            # Wait for the next frame to become available
            frame = await preview_queue.get()
            if frame is None:
                break

            # Write the boundary frame to the response
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")

    return StreamingResponse(generate_frames(), status_code=206, media_type="multipart/x-mixed-replace; boundary=frame")


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Closing the video endpoint")
    await preview_queue.put(None)


app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn

    # Start the server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
