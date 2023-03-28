from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import concurrent.futures
import socketio
import glob
from datetime import datetime
import cv2
import os
import asyncio
import queue

app = FastAPI()
templates = Jinja2Templates(directory="templates")
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app_asgi = socketio.ASGIApp(sio, app)
preview_queue = queue.Queue()

# Replace this with your actual acquisition library import
from multi_camera.acquisition.flir_recording_api import FlirRecorder

acquisition = FlirRecorder("/home/cbm/MultiCameraTracking/multi_camera_configs/cotton_lab_config_20221109.yaml")

# Create a thread pool executor with 1 thread
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

channel_selector_value = 2


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})


class StartData(BaseModel):
    recording_dir: str
    recording_filename: str
    config_file: str


class ChannelData(BaseModel):
    value: int


@app.post("/set_channel")
async def set_channel(data: ChannelData):
    global channel_selector_value
    channel_selector_value = data.value


@app.post("/start")
async def start_recording(data: StartData):
    print(f"Starting recording {data.recording_dir} {data.recording_filename} {data.config_file}")

    loop = asyncio.get_event_loop()

    def receive_frames_wrapper(frames):
        if frames is not None:
            loop.create_task(receive_frames(frames))

    acquisition.configure(recording_path="./data/test_app_video", preview_callback=receive_frames_wrapper)
    await loop.run_in_executor(executor, acquisition.start_acquisition)
    return {"status": "success"}


@app.post("/stop")
async def stop_recording():
    acquisition.stop_acquisition()
    return {"status": "recording_stopped"}


@app.post("/set_channel")
async def set_channel(value: int):
    global channel_selector_value
    channel_selector_value = value
    print(f"Setting channel to {value}")
    return {"status": "channel_set"}


@app.post("/new_session")
async def new_session(subject_id: str):
    date = datetime.date.today().strftime("%Y%m%d")
    session_dir = f"{subject_id}/{date}"
    os.makedirs(f"./data/{session_dir}", exist_ok=True)
    return {"recording_dir": session_dir, "recording_filename": f"{subject_id}_{date}"}


@app.post("/new_trial")
async def new_trial(recording_dir: str, recording_filename: str, config_file: str):
    acquisition.configure(config_file, os.path.join("./data", recording_dir, recording_filename))
    loop = asyncio.get_event_loop()

    def receive_frames_wrapper(frames):
        if frames is not None:
            loop.create_task(receive_frames(frames))

    await loop.run_in_executor(executor, acquisition.start_acquisition)
    return {"status": "recording_started"}


@app.get("/configs")
async def get_configs():
    config_files = glob.glob("/home/cbm/MultiCameraTracking/multi_camera_configs/*.yaml")
    print(config_files)
    return JSONResponse(content=config_files)


@app.get("/recordings")
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
    frame = frames[channel_selector_value]

    # Convert to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_RG2RGB)

    downsampled_frame = downsample_image(frame, 640, 470)  # Downsample to 640x480
    jpeg_image = convert_rgb_to_jpeg(downsampled_frame)

    preview_queue.put(jpeg_image)


@app.get("/video")
async def video_endpoint():
    def generate_frames():
        print("starting to generate frames")
        while True:
            # Wait for the next frame to become available
            frame = preview_queue.get()

            # Write the boundary frame to the response
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")
        print("finished generating frames")

    return StreamingResponse(generate_frames(), status_code=206, media_type="multipart/x-mixed-replace; boundary=frame")
