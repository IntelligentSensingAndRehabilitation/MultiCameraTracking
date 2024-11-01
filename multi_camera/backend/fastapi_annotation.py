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

from multi_camera.backend.recording_db import (
    get_db,
    synchronize_to_datajoint,
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
    acquisition_logger.info("Starting annotation system")

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
        "multi_camera.backend.fastapi_annotation:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        timeout_keep_alive=30,
        # log_level="debug",
    )
