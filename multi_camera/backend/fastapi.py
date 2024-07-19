from fastapi import FastAPI, Depends, Request, APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from uvicorn.main import Server
from websockets.exceptions import ConnectionClosedOK
from datetime import date
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List, Dict
from dataclasses import dataclass, field
import base64
import numpy as np
import logging
import math
import datetime
import cv2
import os
import asyncio

from multi_camera.acquisition.flir_recording_api import FlirRecorder, CameraStatus
from multi_camera.backend.recording_db import (
    get_db,
    add_recording,
    get_recordings,
    modify_recording_entry,
    synchronize_to_datajoint,
    push_to_datajoint,
    ParticipantOut,
    SessionOut,
    RecordingOut,
)


# file templates directory, which is located relative to this file location
templates = os.path.split(__file__)[0]
templates = os.path.join(templates, "templates")
templates = Jinja2Templates(directory=templates)

logger = logging.getLogger("uvicorn.server")

import colorlog

# Create a custom log format with colors
log_colors_config = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red,bg_white",
}
log_format = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s (%(name)s):%(reset)s  %(message)s", log_colors=log_colors_config
)
acquisition_logger = logging.getLogger("acquisition")
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(log_format)
acquisition_logger.addHandler(streamHandler)
acquisition_logger.setLevel(logging.DEBUG)


class Session(BaseModel):
    participant_name: str
    session_date: date
    recording_path: str


@dataclass
class GlobalState:
    preview_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    current_session: Session = None
    recording_status: str = ""
    acquisition = None


_global_state = GlobalState()


def get_global_state() -> GlobalState:
    # print("Getting global state: ", _global_state)
    return _global_state


def db_dependency():
    db = get_db()
    try:
        yield db
    finally:
        db.close()


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        logger.debug("Websocket connected")
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        logger.debug("Websocket Disconnected")
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_json(message)


manager = ConnectionManager()

RECORDING_BASE = "data"
CONFIG_PATH = "/configs/"
DEFAULT_CONFIG = os.path.join(CONFIG_PATH, "cotton_lab_config_20230620.yaml")

print(CONFIG_PATH)
config_files = os.listdir(CONFIG_PATH)
print(config_files)
print([""] + [f for f in config_files if f.endswith(".yaml")])
# import socket
# print(socket.gethostname())
# print(os.environ.get("HOSTNAME", "localhost"))
# print(socket.getfqdn())


loop = asyncio.get_event_loop()


def receive_status(status, progress=None):
    """
    Receive status updates from the acquisition system

    This callback is running on a different thread so needs to be handled carefully.
    """
    global_state = get_global_state()
    global_state.recording_status = status

    update = {"status": status}
    if progress is not None:
        update["progress"] = progress
    else:
        acquisition_logger.info(f"Status: {status}")

    # Put the status in the queue using asyncio from a synchronous function
    loop.create_task(manager.broadcast(update))


@asynccontextmanager
async def lifespan(app: FastAPI):
    state: GlobalState = get_global_state()

    # Perform startup tasks
    acquisition_logger.info("Starting acquisition system")
    state.acquisition = FlirRecorder(receive_status)

    state = get_global_state()

    db = get_db()

    # adding a try/except here to catch the case where the database is not available
    # this can happen if the database is not running or if the network is down
    try:
        synchronize_to_datajoint(db)
    except Exception as e:  
        acquisition_logger.error(f"Could not synchronize to datajoint: {e}")
        

    yield

    # Perform shutdown tasks
    state.acquisition.close()
    acquisition_logger.info("Acquisition system closed")


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
        if not AppStatus.should_exit:
            logger.info("Exit signal detected: " + str(args) + " " + str(kwargs))
            AppStatus.should_exit = True
        original_handler(*args, **kwargs)


Server.handle_exit = AppStatus.handle_exit

# base_url = os.environ.get("REACT_APP_BASE_URL", "localhost")

# Add a middleware to handle CORS
app.add_middleware(
    CORSMiddleware,
    # allow_origins=[f"http://{base_url}:3000",f"http://{base_url}:8000"],
    allow_origins=["*"],
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
    max_frames: int

class PreviewData(BaseModel):
    max_frames: int


class ConfigFileData(BaseModel):
    config: str

class PriorRecordings(BaseModel):
    participant: str
    filename: str
    recording_timestamp: datetime.datetime
    comment: str
    config_file: str
    should_process: bool
    timestamp_spread: float


@api_router.get("/camera_status", response_model=List[CameraStatus])
async def get_camera_status() -> List[CameraStatus]:
    state = get_global_state()
    camera_status = await state.acquisition.get_camera_status()
    return camera_status


@api_router.post("/stop")
async def stop_recording():
    state: GlobalState = get_global_state()
    state.acquisition.stop_acquisition()
    return {}


@api_router.post("/session", response_model=Session)
async def set_session(subject_id: str) -> Session:
    """
    Create a new session directory for the participant

    Args:
        subject_id (str): The participant ID
    Returns:
        dict: A dictionary with the recording directory and filename
    """

    date = datetime.date.today()
    session_dir = os.path.join(RECORDING_BASE, subject_id, date.strftime("%Y%m%d"))
    os.makedirs(session_dir, exist_ok=True)

    state: GlobalState = get_global_state()
    state.current_session = Session(participant_name=subject_id, session_date=date, recording_path=session_dir)
    print("New session: ", state.current_session)

    return state.current_session


@api_router.get("/session", response_model=Session)
async def get_session() -> Session:
    state: GlobalState = get_global_state()
    if state.current_session is None:
        raise HTTPException(status_code=404, detail="No current session")
    return state.current_session


@api_router.post("/new_trial")
async def new_trial(data: NewTrialData, db: Session = Depends(db_dependency)):
    recording_dir = data.recording_dir
    recording_filename = data.recording_filename
    comment = data.comment
    max_frames = data.max_frames

    print("New trial: ", data)

    # Build the recording file name from the components
    now = datetime.datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S")
    recording_path = os.path.join(recording_dir, f"{recording_filename}_{time_str}")

    state: GlobalState = get_global_state()
    current_session = state.current_session

    def receive_frames_wrapper(frames):
        loop.create_task(receive_frames(frames))

    # run acquisition in a separate thread
    acquisition_coroutine = run_in_threadpool(
        state.acquisition.start_acquisition,
        recording_path=recording_path,
        preview_callback=receive_frames_wrapper,
        max_frames=max_frames,
    )
    task = asyncio.create_task(acquisition_coroutine)

    config = await get_current_config()

    def task_done_callback(task):
        print("Task completed.")
        # You can retrieve the result (or exception) of the task using `result()` method.
        try:
            result = task.result()
            print(f"Task result: {result}")

            add_recording(
                db,
                participant_name=current_session.participant_name,
                session_date=current_session.session_date,
                session_path=current_session.recording_path,
                filename=recording_path,
                recording_timestamp=now,
                config_file=config,
                comment=comment,
                timestamp_spread=result["timestamp_spread"],
            )
        except Exception as e:
            print(f"Task raised an exception: {e}")

    task.add_done_callback(task_done_callback)

    return {"recording_file_name": recording_path}


@api_router.post("/preview")
async def preview(data: PreviewData):

    max_frames = data.max_frames

    def receive_frames_wrapper(frames):
        loop.create_task(receive_frames(frames))

    # run acquisition in a separate thread
    import threading
    from functools import partial

    state: GlobalState = get_global_state()
    await run_in_threadpool(state.acquisition.start_acquisition, preview_callback=receive_frames_wrapper, max_frames=max_frames)

    return {}


@api_router.post("/stop")
async def stop():
    state: GlobalState = get_global_state()
    state.acquisition.stop_acquisition()
    return {}


@api_router.get("/prior_recordings", response_model=List[PriorRecordings])
async def get_prior_recordings(db=Depends(db_dependency)) -> List[PriorRecordings]:
    state = get_global_state()
    if state.current_session is None:
        participant_name = None
    else:
        participant_name = state.current_session.participant_name
    prior_recordings = []
    db_recordings: ParticipantOut = get_recordings(db, participant_name=participant_name)
    for participant in db_recordings:
        participant: ParticipantOut = participant
        for session in participant.sessions:
            session: SessionOut = session
            for recording in session.recordings:
                recording: RecordingOut = recording
                prior_recordings.append(
                    PriorRecordings(
                        participant=participant.name,
                        filename=recording.filename,
                        recording_timestamp=recording.recording_timestamp,
                        comment=recording.comment,
                        config_file=recording.config_file,
                        should_process=recording.should_process,
                        timestamp_spread=recording.timestamp_spread,
                    )
                )

    # reverse the list before returning to put in chronological order
    prior_recordings.reverse()
    return prior_recordings


@api_router.post("/update_recording")
async def update_recording(recording: PriorRecordings, db=Depends(db_dependency)):
    print("Updating recording: ", recording)

    participant = ParticipantOut(name=recording.participant, sessions=[])
    recording = RecordingOut(
        filename=recording.filename,
        recording_timestamp=recording.recording_timestamp,
        comment=recording.comment,
        config_file=recording.config_file,
        should_process=recording.should_process,
        timestamp_spread=recording.timestamp_spread,
    )

    modify_recording_entry(db, participant, recording)

    return {}


@api_router.post("/calibrate")
async def update_recording(recording: PriorRecordings, charuco_flag, db=Depends(db_dependency)):
    from multi_camera.datajoint.calibrate_cameras import run_calibration

    print("Calibrating recording: ", recording)

    vid_path, vid_base = os.path.split(recording.filename)

    if charuco_flag == True:
        print("Using charuco board")
        checkerboard_size=109.0
        checkerboard_dim=(5, 7)
        charuco = True
    else:
        print("Using checkerboard")
        checkerboard_size=110.0
        checkerboard_dim=(4, 6)
        charuco = False


    calibration_coroutine = run_in_threadpool(
        run_calibration,
        vid_base=vid_base,
        vid_path=vid_path,
        checkerboard_size=checkerboard_size,
        checkerboard_dim=checkerboard_dim,
        charuco=charuco,
    )
    task = asyncio.create_task(calibration_coroutine)

    return {}


class ProcessSession(BaseModel):
    participant_name: str
    session_date: datetime.date
    video_project: str


@api_router.post("/process_session")
async def update_recording(session: ProcessSession, db=Depends(db_dependency)):
    push_to_datajoint(db, session.participant_name, session.session_date, session.video_project)
    return {}


@api_router.get("/recording_db", response_model=List[ParticipantOut])
async def get_recording_db(db=Depends(db_dependency)) -> List[ParticipantOut]:
    return get_recordings(db)


@api_router.get("/configs")
async def get_configs():
    print(CONFIG_PATH)
    config_files = os.listdir(CONFIG_PATH)
    print(config_files)
    config_files = [""] + [f for f in config_files if f.endswith(".yaml")]
    return JSONResponse(content=config_files)


@api_router.get("/current_config", response_model=str)
async def get_current_config() -> str:
    state: GlobalState = get_global_state()

    config = state.acquisition.config_file
    if config is None:
        return ""
    return os.path.split(config)[-1]


@api_router.post("/current_config")
async def update_config(config: ConfigFileData):
    print("Received config: ", config.config)
    state: GlobalState = get_global_state()
    if config.config == "":
        state.acquisition.reset()
    else:
        await state.acquisition.configure_cameras(os.path.join(CONFIG_PATH, config.config))
    return {"status": "success", "config": config.config}


@api_router.post("/reset_cameras")
async def reset_cameras():
    state: GlobalState = get_global_state()
    # await run_in_threadpool(state.acquisition.reset_cameras)
    await state.acquisition.reset_cameras()
    return {"status": "success"}


# create an endpoint that exposes the camera statuses
@api_router.get("/camera_status")
async def get_camera_status() -> List[CameraStatus]:
    state: GlobalState = get_global_state()
    camera_status = state.acquisition.get_camera_status()
    return camera_status


@api_router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    """This websocket endpoint is used to send the recording status to the client"""

    state: GlobalState = get_global_state()
    logger.debug("Websocket request received")

    await manager.connect(websocket)

    logger.debug("Websocket connected")

    try:
        while not AppStatus.should_exit:
            try:
                # other code will broadcast status updates to listening clients
                # this loop just hands out waiting for the websocket to close
                status = await asyncio.wait_for(websocket.receive(), timeout=0.5)
            except asyncio.TimeoutError:
                # need this to monitor for exit
                # print(websocket.client_state, websocket.application_state)
                pass
        logger.debug("Exit flag detected")
    except WebSocketDisconnect as e:
        logger.debug("Websocket disconnected with WebSocketDisconnect: %s", e)
    except RuntimeError as e:
        # This is a cludge, but when the remote websocket is closed then receive throws a RuntimeError.
        # it seems like the WebSocketDisconnect should be thrown, but it is not.
        logger.debug("Websocket disconnected with RuntimeError: %s", e)

    manager.disconnect(websocket)

    logger.info("Regular websocket exited")


@api_router.get("/recording_status", response_model=str)
async def get_recording_status() -> str:
    state: GlobalState = get_global_state()
    return state.recording_status


def downsample_image(image, new_width, new_height):
    return cv2.resize(image, (new_width, new_height))


def convert_rgb_to_jpeg(frame):
    ret, jpeg_image = cv2.imencode(".jpg", frame)
    return jpeg_image.tobytes()


async def receive_frames(frames):
    state: GlobalState = get_global_state()
    if not state.preview_queue.empty():
        # If the queue is not empty, then we are not keeping up with the frames
        # logger.warn("Dropping frame")
        return

    if frames is None:
        print("Received empty frame")
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

    await state.preview_queue.put(jpeg_image)


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
    state: GlobalState = get_global_state()
    await websocket.accept()
    logger.info("Video Websocket connected")
    try:
        while not AppStatus.should_exit:
            try:
                frame = await asyncio.wait_for(state.preview_queue.get(), timeout=2.5)
                logger.debug("Sending frame")
                if frame is None:
                    break
                await websocket.send_bytes(frame)
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect as e:
        logger.info("Video Websocket disconnected with WebSocketDisconnect: %s", e)
    except ConnectionClosedOK as e:
        logger.info("Video  Websocket disconnected with ConnectionClosedOK: %s", e)

    logger.info("Video websocket exited")


class SMPLData(BaseModel):
    ids: List[int]
    frames: int
    type: str
    faces: List[List[int]]
    meshes: str  # base64 encoded


@api_router.get("/mesh", response_model=SMPLData)
async def get_mesh(
    filename: str = Query(None, description="Name of the file to be used."), downsample: int = Query(default=5)
) -> SMPLData:
    from .annotation import get_mesh

    res = get_mesh(filename, downsample)
    return SMPLData(**res)


class UnannotatedRecordings(BaseModel):
    video_base_filenames: List[str]


@api_router.get("/unannotated_recordings")
async def get_unannotated_recordings():
    from .annotation import get_unannotated_recordings

    video_base_filenames = get_unannotated_recordings()
    return UnannotatedRecordings(video_base_filenames=video_base_filenames.tolist())


class Annotation(BaseModel):
    video_base_filename: str
    ids: List[int]


@api_router.post("/annotation")
async def post_annotation(annotation: Annotation):
    from .annotation import annotate_recording

    success = annotate_recording(annotation.video_base_filename, annotation.ids)
    return {"success": success}


# code for single person SMPL reconstructions


class ReconstructionTrials(BaseModel):
    participant_id: str
    session_date: date
    video_base_filename: str


@api_router.get("/smpl_trials", response_model=List[ReconstructionTrials])
async def get_smpl_trials(
    model: str = Query(default="smpl", description="Type of SMPL model to use.")
) -> List[ReconstructionTrials]:
    print(model)
    from .smpl import get_smpl_trials

    trials = get_smpl_trials(model)
    trials = [ReconstructionTrials(**trial) for trial in trials]
    return trials


@api_router.get("/smpl")
async def get_smpl(
    filename: str = Query(None, description="Name of the file to be used."),
    model: str = Query(default="smpl", description="Type of SMPL model to use."),
) -> SMPLData:
    from .smpl import get_smpl_trajectory

    res = get_smpl_trajectory(filename, model)

    return SMPLData(ids=[0], frames=-1, type="smpl", faces=res["faces"], meshes=res["vertices"])


# code for working with biomechanical data


@api_router.get("/biomechanics_trials", response_model=List[ReconstructionTrials])
async def get_biomechanics_trials() -> List[ReconstructionTrials]:
    from .biomechanics import get_biomechanics_trials

    trials = get_biomechanics_trials()
    trials = [ReconstructionTrials(**trial) for trial in trials]
    return trials


class MeshData(BaseModel):
    vertices: List[List[float]]
    faces: List[List[int]]


class TrajectoryData(BaseModel):
    positions: List[List[float]]
    rotations: List[List[float]]


class BiomechanicsData(BaseModel):
    meshes: Dict[str, MeshData]
    trajectories: Dict[str, TrajectoryData]


@api_router.get("/biomechanics")
async def get_biomechanics(filename: str = Query(None, description="Name of the file to be used.")):
    from .biomechanics import get_biomechanics_trajectory

    # fetch the data dictionary for this trial
    res = get_biomechanics_trajectory(filename)

    meshes = res["meshes"]
    traj = res["trajectories"]

    # repackage the data with Pydantic
    meshes = {k: MeshData(vertices=v["vertices"], faces=v["faces"]) for k, v in meshes.items()}
    traj = {k: TrajectoryData(positions=v["positions"], rotations=v["rotations"]) for k, v in traj.items()}

    data = BiomechanicsData(meshes=meshes, trajectories=traj)

    return data


app.include_router(api_router)


def register_exception(app: FastAPI):
    from fastapi import Request, status
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
        # or logger.error(f'{exc}')
        # logger.error(request, exc_str)
        print(request, exc_str)
        content = {"status_code": 10422, "message": exc_str, "data": None}
        return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


register_exception(app)

if __name__ == "__main__":
    import uvicorn

    # Start the server
    uvicorn.run(
        "multi_camera.backend.fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        timeout_keep_alive=30,
        # log_level="debug",
    )
