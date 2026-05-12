from fastapi import (
    FastAPI,
    Depends,
    Request,
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Query,
    UploadFile,
    File,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from uvicorn.main import Server
from websockets.exceptions import ConnectionClosedOK
from datetime import date
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import numpy as np
import logging
import math
import datetime
import cv2
import os
import asyncio
import shutil
import re
import time
from pathlib import Path

from multi_camera.acquisition.flir_recording_api import FlirRecorder, CameraStatus
from multi_camera.acquisition.diagnostics.json_parser import (
    diagnose_session_issues,
    diagnose_sync_issues,
    load_session,
)
from multi_camera.acquisition.health import (
    CameraReachabilityReport,
    DhcpServerStatus,
    HealthCheckReport,
    HealthConfig,
    HealthIdlePoller,
    HostNetworkStatus,
    check_camera_reachability,
    run_health_check,
    severity_changed,
)
from multi_camera.backend.recording_db import (
    get_db,
    add_recording,
    add_photo,
    get_recordings,
    modify_recording_entry,
    rename_recording_entry,
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
    "%(log_color)s%(levelname)s (%(name)s):%(reset)s  %(message)s",
    log_colors=log_colors_config,
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
    selected_camera: int | None = None
    health_config: HealthConfig = field(default_factory=HealthConfig)
    _health_cache: HealthCheckReport | None = None
    _health_cache_ts: float = 0.0
    _health_cache_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _health_poller: HealthIdlePoller | None = None
    last_session_insights: set[str] = field(default_factory=set)


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


class DiagnosticsManager:
    """WebSocket fan-out for structured diagnostic envelopes.

    Separate from :class:`ConnectionManager` so bursty diagnostic broadcasts
    can't delay the recording status writer on ``/api/v1/ws``. Per-connection
    asyncio.Lock serializes writes to avoid racing the websockets keepalive
    ping; failed sends drop the connection instead of leaving stale entries.
    """

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._locks: dict[WebSocket, asyncio.Lock] = {}

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        self._locks[websocket] = asyncio.Lock()
        logger.debug("Diagnostics websocket connected")

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self._locks.pop(websocket, None)
        logger.debug("Diagnostics websocket disconnected")

    async def _send_one(self, websocket: WebSocket, message: dict) -> bool:
        lock = self._locks.get(websocket)
        if lock is None:
            return False
        async with lock:
            try:
                await websocket.send_json(message)
                return True
            except Exception as e:  # noqa: BLE001
                acquisition_logger.warning(
                    f"Diagnostics WS send failed, dropping connection: {e}"
                )
                return False

    async def broadcast(self, message: dict) -> None:
        if not self.active_connections:
            return
        results = await asyncio.gather(
            *(self._send_one(c, message) for c in list(self.active_connections)),
            return_exceptions=False,
        )
        for connection, ok in zip(list(self.active_connections), results):
            if not ok:
                self.disconnect(connection)


diagnostics_manager = DiagnosticsManager()


def _on_acquisition_diagnostic(envelope: dict) -> None:
    """Bridge ``FlirRecorder._fire_diagnostic_once`` → ``broadcast_event``.

    Called from acquisition worker threads when the recorder catches a
    Spinnaker error per camera. Forwards the envelope to the existing
    diagnostics WS fan-out as a ``session_insight`` event. If the
    recorder attached a ``remediation`` list (from
    ``classify_spinnaker_error``), the steps are passed through in
    ``details`` so the Diagnostics tab renders them as a numbered list.
    """
    details = dict(envelope.get("details", {}))
    remediation = envelope.get("remediation")
    if remediation:
        details["remediation"] = remediation
    broadcast_event(
        event_type="session_insight",
        level=envelope.get("level", "warn"),
        code=envelope.get("code", "acquisition_error"),
        message=envelope.get("message", ""),
        details=details,
    )


def _on_idle_health_poll(
    new_report: HealthCheckReport,
    previous: HealthCheckReport | None,
) -> None:
    """Broadcast a ``health_report`` envelope when overall severity changes.

    Called from the :class:`HealthIdlePoller` daemon thread after each poll.
    Avoids spamming clients — if severity didn't change since the last poll,
    nothing is broadcast.
    """
    if not severity_changed(new_report, previous):
        return
    top = new_report.findings[0] if new_report.findings else None
    msg = top.message if top else f"Health is now {new_report.overall}"
    broadcast_event(
        event_type="health_report",
        level=new_report.overall,
        code=top.code if top else "health_status_change",
        message=msg,
        details={
            "overall": new_report.overall,
            "missing_cameras": list(new_report.cameras.missing),
            "dhcp_applicable": new_report.dhcp.applicable,
            "dhcp_service_active": new_report.dhcp.service_active,
        },
    )


def broadcast_event(
    event_type: str,
    level: str,
    code: str,
    message: str,
    details: dict | None = None,
) -> None:
    """Dispatch a structured diagnostic envelope. Safe from any thread."""
    envelope = {
        "type": event_type,
        "level": level,
        "code": code,
        "message": message,
        "details": details or {},
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
    }
    coro = diagnostics_manager.broadcast(envelope)
    try:
        asyncio.get_running_loop()
        asyncio.create_task(coro)
    except RuntimeError:
        asyncio.run_coroutine_threadsafe(coro, loop)

RECORDING_BASE = "data"
CONFIG_PATH = "/configs/"
DEFAULT_CONFIG = os.path.join(CONFIG_PATH, "cotton_lab_config_20230620.yaml")
DATA_VOLUME = os.environ.get("DATA_VOLUME", "/data")
DISK_SPACE_WARNING_THRESHOLD_GB = float(
    os.environ.get("DISK_SPACE_WARNING_THRESHOLD_GB", "50")
)


def get_disk_space_info(path: str) -> Dict:
    """
    Get disk space information for the specified path.

    Args:
        path: The directory path to check

    Returns:
        Dictionary containing disk space information:
        - disk_space_gb_remaining: Available space in GB
        - disk_space_percent_remaining: Available space as percentage
        - disk_space_warning: True if below threshold (in GB)
    """
    try:
        stat = shutil.disk_usage(path)
        total_gb = stat.total / (1024**3)
        free_gb = stat.free / (1024**3)
        percent_remaining = (stat.free / stat.total) * 100 if stat.total > 0 else 0

        return {
            "disk_space_gb_remaining": round(free_gb, 2),
            "disk_space_percent_remaining": round(percent_remaining, 2),
            "disk_space_warning": free_gb < DISK_SPACE_WARNING_THRESHOLD_GB,
            "disk_space_total_gb": round(total_gb, 2),
        }
    except Exception as e:
        acquisition_logger.error(f"Error checking disk space for {path}: {e}")
        return {
            "disk_space_gb_remaining": 0,
            "disk_space_percent_remaining": 0,
            "disk_space_warning": True,
            "disk_space_total_gb": 0,
        }



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


def _expected_serials(recorder: FlirRecorder | None) -> list[str]:
    """Pull the list of expected cameras from the currently-loaded YAML config."""
    if recorder is None:
        return []
    config = getattr(recorder, "camera_config", None) or {}
    camera_info = config.get("camera-info", {}) if isinstance(config, dict) else {}
    return [str(s) for s in camera_info.keys()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    state: GlobalState = get_global_state()

    # Perform startup tasks
    acquisition_logger.info("Starting acquisition system")
    state.acquisition = FlirRecorder(
        receive_status,
        diagnostics_callback=_on_acquisition_diagnostic,
    )
    state.health_config = HealthConfig.from_env()
    acquisition_logger.info(
        f"Health config: mode={state.health_config.deployment_mode} "
        f"iface={state.health_config.network_interface} "
        f"idle_poll={state.health_config.idle_poll_s}s"
    )

    state = get_global_state()

    state._health_poller = HealthIdlePoller(
        config=state.health_config,
        run_check=lambda: run_health_check(
            config=state.health_config,
            expected_serials=_expected_serials(state.acquisition),
            recorder=state.acquisition,
            recording_state=state.recording_status or "Idle",
            # During an active recording the recorder owns the camera handles;
            # PySpin GigE enumeration would race with the worker threads and
            # is also slow. Camera-specific issues during recording are
            # surfaced via the recorder's diagnostics_callback instead. DHCP
            # and host-network checks still run.
            skip_camera_enumeration=(state.recording_status in _BUSY_RECORDING_STATES),
        ),
        on_poll=_on_idle_health_poll,
        logger=acquisition_logger,
    )
    state._health_poller.start()

    db = get_db()

    # adding a try/except here to catch the case where the database is not available
    # this can happen if the database is not running or if the network is down
    try:
        synchronize_to_datajoint(db)
    except Exception as e:
        acquisition_logger.error(f"Could not synchronize to datajoint: {e}")

    yield

    # Perform shutdown tasks
    if state._health_poller is not None:
        state._health_poller.stop()
        state._health_poller = None
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
    # Newer Starlette versions require ``request`` as a positional/named arg
    # on TemplateResponse, not as a field in ``context``. Pass it explicitly.
    return templates.TemplateResponse(request=request, name="index.html")


class NewTrialData(BaseModel):
    recording_dir: str
    recording_filename: str
    comment: str
    max_frames: int
    diagnostics_level: int = 1
    frame_skip_recovery: bool = True


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


class ImageUploadResponse(BaseModel):
    saved_path: str
    original_filename: str
    saved_filename: str
    file_size_mb: float
    description: Optional[str] = None
    timestamp: datetime.datetime
    disk_space_gb_remaining: float
    disk_space_percent_remaining: float
    disk_space_warning: bool
    disk_space_total_gb: float


@api_router.get("/camera_status", response_model=List[CameraStatus])
async def get_camera_status() -> List[CameraStatus]:
    state = get_global_state()
    camera_status = await state.acquisition.get_camera_status()
    return camera_status


async def _get_health_report(force_refresh: bool = False) -> HealthCheckReport:
    """Return a HealthCheckReport, using a short TTL cache on GlobalState.

    Camera reachability via PySpin GigE broadcast is the slow part (multiple
    seconds in network mode). Caching the full report behind a TTL means
    REST polling from the GUI is cheap.
    """
    state = get_global_state()
    async with state._health_cache_lock:
        now_mono = time.monotonic()
        if (
            not force_refresh
            and state._health_cache is not None
            and now_mono - state._health_cache_ts < state.health_config.cache_ttl_s
        ):
            return state._health_cache

        expected = _expected_serials(state.acquisition)
        recording_state = state.recording_status or "Idle"
        report = await run_in_threadpool(
            run_health_check,
            config=state.health_config,
            expected_serials=expected,
            recorder=state.acquisition,
            recording_state=recording_state,
        )
        state._health_cache = report
        state._health_cache_ts = now_mono
        return report


@api_router.get("/health", response_model=HealthCheckReport)
async def get_health() -> HealthCheckReport:
    """Structured health snapshot: DHCP + camera + host network."""
    return await _get_health_report(force_refresh=False)


@api_router.get("/health/dhcp", response_model=DhcpServerStatus)
async def get_health_dhcp() -> DhcpServerStatus:
    report = await _get_health_report(force_refresh=False)
    return report.dhcp


@api_router.get("/health/cameras", response_model=CameraReachabilityReport)
async def get_health_cameras() -> CameraReachabilityReport:
    report = await _get_health_report(force_refresh=False)
    return report.cameras


@api_router.get("/health/host_network", response_model=HostNetworkStatus)
async def get_health_host_network() -> HostNetworkStatus:
    report = await _get_health_report(force_refresh=False)
    return report.host_network


@api_router.post("/health/refresh", response_model=HealthCheckReport)
async def refresh_health() -> HealthCheckReport:
    """Force a fresh health check, bypassing the TTL cache."""
    return await _get_health_report(force_refresh=True)


class SessionSummaryReport(BaseModel):
    n_trials: int
    insights: List[str]
    recommendations: List[str]
    trial_findings: List[str] = []
    generated_at: datetime.datetime


def _build_session_summary(session_dir: str) -> SessionSummaryReport:
    """Run all diagnostic detectors over the JSON sidecars in ``session_dir``.

    Combines session-level synthesizers (``diagnose_session_issues``) with
    per-trial detectors (``diagnose_sync_issues``).
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    path = Path(session_dir)
    try:
        report = load_session(path)
    except FileNotFoundError:
        return SessionSummaryReport(
            n_trials=0,
            insights=[],
            recommendations=[],
            trial_findings=[],
            generated_at=now,
        )
    insights, recommendations = diagnose_session_issues(report)
    trial_findings = diagnose_sync_issues(report)
    return SessionSummaryReport(
        n_trials=len(report.trials),
        insights=insights,
        recommendations=recommendations,
        trial_findings=trial_findings,
        generated_at=now,
    )


def _broadcast_new_session_insights() -> None:
    """Run session-level + per-trial detectors, broadcast only NEW findings.

    Called after each trial completes. Diffs against
    ``state.last_session_insights`` (updated in-place) so the same finding
    doesn't re-broadcast when a later trial includes earlier trials in its
    sidecar set. Per-trial detectors typically embed trial identifiers in
    the finding string, so they dedup naturally.

    Failures inside the synthesizers are caught and logged — the trial has
    already finished, so we never want this to raise back into the
    task_done_callback path.
    """
    state = get_global_state()
    if state.current_session is None:
        return
    try:
        report = load_session(Path(state.current_session.recording_path))
        session_insights, _recommendations = diagnose_session_issues(report)
        per_trial_insights = diagnose_sync_issues(report)
        n_trials = len(report.trials)
    except Exception as e:  # noqa: BLE001
        acquisition_logger.error(f"Could not run diagnostics: {e}", exc_info=True)
        return

    session_set = set(session_insights)
    per_trial_set = set(per_trial_insights)
    current = session_set | per_trial_set
    new_insights = current - state.last_session_insights
    state.last_session_insights = current

    for insight in new_insights:
        is_per_trial = insight in per_trial_set and insight not in session_set
        broadcast_event(
            event_type="session_insight",
            level="warn",
            code=(
                "trial_symptom_detected"
                if is_per_trial
                else "session_pattern_detected"
            ),
            message=insight,
            details={
                "n_trials": n_trials,
                "scope": "trial" if is_per_trial else "session",
            },
        )


@api_router.get("/health/session_summary", response_model=SessionSummaryReport)
async def get_session_summary() -> SessionSummaryReport:
    """Run session-level synthesizers over the current session's recordings.

    Returns plain-English insights and recommendations for the diagnostics
    tab. Requires an active session; returns 404 if none is set.
    """
    state = get_global_state()
    if state.current_session is None:
        raise HTTPException(status_code=404, detail="No current session")
    return await run_in_threadpool(
        _build_session_summary, state.current_session.recording_path
    )


@api_router.post("/stop")
async def stop_recording():
    state: GlobalState = get_global_state()
    state.acquisition.stop_acquisition()
    return {}


@api_router.post("/session", response_model=Session)
async def set_session(subject_id: str, fin: Optional[str] = None, db=Depends(db_dependency)) -> Session:
    """
    Create a new session directory for the participant

    Args:
        subject_id (str): The participant ID
        fin (str, optional): Financial Identification Number from patient wristband.
            Stored in the recordings.db alongside the participant. Proper PHI access
            control is deferred to the DataJoint export layer (future work).
    Returns:
        dict: A dictionary with the recording directory and filename
    """

    date = datetime.date.today()
    session_dir = os.path.join(RECORDING_BASE, subject_id, date.strftime("%Y%m%d"))
    os.makedirs(session_dir, exist_ok=True)

    state: GlobalState = get_global_state()
    state.current_session = Session(
        participant_name=subject_id, session_date=date, recording_path=session_dir
    )
    state.last_session_insights = set()
    print("New session: ", state.current_session)

    if fin and fin.strip():
        from multi_camera.backend.recording_db import store_fin
        store_fin(db, participant_name=subject_id, fin=fin.strip())
        print(f"FIN stored for participant {subject_id}")

    return state.current_session


@api_router.get("/session", response_model=Session)
async def get_session() -> Session:
    state: GlobalState = get_global_state()
    if state.current_session is None:
        raise HTTPException(status_code=404, detail="No current session")
    return state.current_session


ALLOWED_IMAGE_TYPES = {'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'}
MAX_UPLOAD_SIZE_MB = 25


@api_router.post("/upload_image", response_model=ImageUploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    db=Depends(db_dependency),
) -> ImageUploadResponse:
    """
    Upload a patient identification image for the current session.

    Images are stored in the session's images/ subdirectory.

    Args:
        file: The image file to upload (multipart/form-data)
        description: Optional label for the image (e.g., "front view")
    """
    state: GlobalState = get_global_state()
    if state.current_session is None:
        raise HTTPException(status_code=404, detail="No active session. Create a session before uploading images.")

    current_session = state.current_session

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed types: JPEG, PNG, GIF, BMP, WebP",
        )

    file_content = await file.read()
    file_size_mb = len(file_content) / (1024 * 1024)

    if file_size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {file_size_mb:.1f}MB. Maximum size: {MAX_UPLOAD_SIZE_MB}MB",
        )

    images_dir = os.path.join(current_session.recording_path, "images")
    os.makedirs(images_dir, exist_ok=True)

    timestamp = datetime.datetime.now()
    time_str = timestamp.strftime("%Y%m%d_%H%M%S")
    original_filename = file.filename or "uploaded_image.jpg"
    safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', os.path.basename(original_filename))
    if not safe_filename or safe_filename.startswith('.'):
        safe_filename = "uploaded_image.jpg"
    saved_filename = f"{time_str}_{safe_filename}"
    saved_path = os.path.join(images_dir, saved_filename)

    with open(saved_path, "wb") as f:
        f.write(file_content)

    add_photo(
        db,
        participant_name=current_session.participant_name,
        session_date=current_session.session_date,
        session_path=current_session.recording_path,
        filename=saved_filename,
        original_filename=original_filename,
        saved_path=saved_path,
        file_size_mb=round(file_size_mb, 3),
        upload_timestamp=timestamp,
        description=description,
    )

    acquisition_logger.info(
        f"Image saved: {saved_filename} ({file_size_mb:.2f}MB) "
        f"for session {current_session.participant_name}/{current_session.session_date}"
    )

    disk_info = get_disk_space_info(DATA_VOLUME)
    if disk_info["disk_space_warning"]:
        acquisition_logger.warning(
            f"Low disk space after image upload: {disk_info['disk_space_percent_remaining']}% remaining "
            f"({disk_info['disk_space_gb_remaining']} GB)"
        )

    return ImageUploadResponse(
        saved_path=saved_path,
        original_filename=original_filename,
        saved_filename=saved_filename,
        file_size_mb=round(file_size_mb, 3),
        description=description,
        timestamp=timestamp,
        **disk_info,
    )


@api_router.get("/status")
async def get_status() -> Dict:
    """
    Get system status including disk space information.

    Returns:
        Dictionary containing:
        - disk_space_gb_remaining: Available space in GB
        - disk_space_percent_remaining: Available space as percentage
        - disk_space_warning: True if below threshold
        - disk_space_total_gb: Total disk space in GB
    """
    disk_info = get_disk_space_info(DATA_VOLUME)
    return disk_info


@api_router.post("/new_trial")
async def new_trial(data: NewTrialData, db: Session = Depends(db_dependency)):
    recording_dir = data.recording_dir
    recording_filename = data.recording_filename
    comment = data.comment
    max_frames = data.max_frames

    print("New trial: ", data)

    # Check disk space before starting recording
    disk_info = get_disk_space_info(DATA_VOLUME)
    if disk_info["disk_space_warning"]:
        acquisition_logger.warning(
            f"Low disk space: {disk_info['disk_space_percent_remaining']}% remaining "
            f"({disk_info['disk_space_gb_remaining']} GB)"
        )

    # Build the recording file name from the components
    now = datetime.datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S")
    recording_path = os.path.join(recording_dir, f"{recording_filename}_{time_str}")

    state: GlobalState = get_global_state()
    state.selected_camera = None
    current_session = state.current_session

    expected = _expected_serials(state.acquisition)
    if expected:
        preflight = await run_in_threadpool(
            check_camera_reachability,
            expected_serials=expected,
            recorder=state.acquisition,
        )
        if preflight.missing:
            # FlirRecorder skips missing cameras and continues; warn the
            # operator without blocking the take.
            broadcast_event(
                event_type="session_insight",
                level="warn",
                code="camera_missing",
                message=(
                    f"Recording started without camera(s) "
                    f"{', '.join(preflight.missing)} — check cables and power."
                ),
                details={"missing": preflight.missing},
            )

    def receive_frames_wrapper(frames):
        loop.create_task(receive_frames(frames))

    # Flip status synchronously before scheduling the acquisition thread, so
    # recovery endpoints (restart/restore/exclude) immediately see the system
    # as busy. Otherwise this handler returns 200 before start_acquisition's
    # set_status("Recording") runs on the worker, leaving a window where a
    # follow-up Restart click would tear down PySpin mid-BeginAcquisition.
    state.recording_status = "Starting"

    # run acquisition in a separate thread
    acquisition_coroutine = run_in_threadpool(
        state.acquisition.start_acquisition,
        recording_path=recording_path,
        preview_callback=receive_frames_wrapper,
        max_frames=max_frames,
        diagnostics_level=data.diagnostics_level,
        frame_skip_recovery=data.frame_skip_recovery,
    )
    task = asyncio.create_task(acquisition_coroutine)

    config = await get_current_config()

    def task_done_callback(task):
        try:
            result = task.result()

            # The result contains a list of all the recordings after acquisition was started
            for record in result:
                add_recording(
                    db,
                    participant_name=current_session.participant_name,
                    session_date=current_session.session_date,
                    session_path=current_session.recording_path,
                    filename=record["filename"],
                    recording_timestamp=record["recording_timestamp"],
                    config_file=config,
                    comment=comment,
                    timestamp_spread=record["timestamp_spread"],
                )
            _broadcast_new_session_insights()
        except Exception as e:
            import traceback
            acquisition_logger.error(f"Trial failed: {e}", exc_info=True)
            # If start_acquisition raised before its set_status("Recording")
            # ran, status is still "Starting" — clear it so recovery
            # endpoints aren't blocked.
            if state.recording_status == "Starting":
                state.recording_status = "Idle"
            broadcast_event(
                event_type="error",
                level="error",
                code="trial_failed",
                message=f"Recording failed: {e}",
                details={
                    "exception_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                },
            )

    task.add_done_callback(task_done_callback)

    return {"recording_file_name": recording_path, **disk_info}


@api_router.post("/preview")
async def preview(data: PreviewData):
    max_frames = data.max_frames

    def receive_frames_wrapper(frames):
        loop.create_task(receive_frames(frames))

    state: GlobalState = get_global_state()
    state.selected_camera = None
    await run_in_threadpool(
        state.acquisition.start_acquisition,
        preview_callback=receive_frames_wrapper,
        max_frames=max_frames,
    )

    return {}


@api_router.post("/stop")
async def stop():
    state: GlobalState = get_global_state()
    state.selected_camera = None
    state.acquisition.stop_acquisition()
    return {}


@api_router.post("/validate_sync")
async def validate_sync():
    state: GlobalState = get_global_state()
    result = await run_in_threadpool(state.acquisition.validate_sync)
    return result


@api_router.get("/prior_recordings", response_model=List[PriorRecordings])
async def get_prior_recordings(db=Depends(db_dependency)) -> List[PriorRecordings]:
    state = get_global_state()
    if state.current_session is None:
        participant_name = None
    else:
        participant_name = state.current_session.participant_name
    prior_recordings = []
    db_recordings: ParticipantOut = get_recordings(
        db, participant_name=participant_name
    )
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


class RenameRecordingData(BaseModel):
    participant: str
    filename: str
    new_filename: str


@api_router.post("/rename_recording")
async def rename_recording(data: RenameRecordingData, db=Depends(db_dependency)):
    print("Renaming recording: ", data)
    try:
        rename_recording_entry(db, data.participant, data.filename, data.new_filename)
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {}


@api_router.post("/calibrate")
async def update_recording(
    recording: PriorRecordings,
    charuco_flag: bool = Query(
        ..., description="True for Charuco, False for Checkerboard"
    ),
    db=Depends(db_dependency),
):
    from multi_camera.datajoint.calibrate_cameras import run_calibration

    print("Calibrating recording: ", recording)

    vid_path, vid_base = os.path.split(recording.filename)

    if charuco_flag == True:
        print("Using charuco board")
        checkerboard_size = 109.0
        checkerboard_dim = (5, 7)
        charuco = True
    else:
        print("Using checkerboard")
        checkerboard_size = 110.0
        checkerboard_dim = (4, 6)
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
    push_to_datajoint(
        db, session.participant_name, session.session_date, session.video_project
    )
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


# States during which recovery endpoints (restart / restore / exclude) must
# refuse to act. "Starting" exists to close the race between new_trial
# returning 200 and FlirRecorder.start_acquisition flipping the status to
# "Recording" on its worker thread — without it, a fast click on
# "Restart acquisition" could tear down PySpin mid-BeginAcquisition.
_BUSY_RECORDING_STATES = frozenset({"Starting", "Recording"})


async def _refresh_health_after_configure() -> None:
    """Force a health re-check after a camera reconfigure so the operator
    immediately sees post-init state (link speeds, throughput, missing
    cameras) without waiting for the next idle poll. Broadcasts the new
    report so the GUI re-fetches.
    """
    try:
        report = await _get_health_report(force_refresh=True)
    except Exception as e:  # noqa: BLE001
        acquisition_logger.warning(f"Post-configure health refresh failed: {e}")
        return
    broadcast_event(
        event_type="health_report",
        level=report.overall,
        code="post_configure_refresh",
        message="Health report updated after camera reconfigure.",
    )


async def _reset_and_configure(saved_config: str | None, action: str) -> None:
    """Tear down PySpin and reconfigure from saved_config. If reconfigure
    fails the system is left with no cams Init'd; broadcast an error
    envelope with recovery instructions so the operator sees it in the
    diagnostics-tab finding list rather than just a small red div under
    the button. Re-raises so the calling endpoint returns 500.

    `action` is the operator-facing verb (e.g. "Restart") used in the
    error message.
    """
    state: GlobalState = get_global_state()
    # Camera close is per-camera over a threadpool inside FlirRecorder; with
    # 8+ cameras it can stall the asyncio event loop for seconds, blocking
    # the very websocket fanout that just queued the "started" envelope.
    await run_in_threadpool(state.acquisition.reset)
    if saved_config is None:
        return
    try:
        await state.acquisition.configure_cameras(saved_config)
    except Exception as e:  # noqa: BLE001
        acquisition_logger.error(
            f"{action}: configure_cameras failed after reset: {e}", exc_info=True
        )
        broadcast_event(
            event_type="session_insight",
            level="error",
            code="reconfigure_failed",
            message=f"{action} failed: cameras did not come back up ({e}).",
            details={
                "remediation": [
                    "Click 'Restart acquisition' to retry.",
                    "If it fails again, power-cycle the camera switch and retry.",
                    "If the issue persists, check that all cameras are powered and connected.",
                ],
            },
        )
        raise


@api_router.post("/current_config")
async def update_config(config: ConfigFileData):
    print("Received config: ", config.config)
    state: GlobalState = get_global_state()
    if config.config == "":
        state.acquisition.reset()
    else:
        await state.acquisition.configure_cameras(
            os.path.join(CONFIG_PATH, config.config)
        )
    await _refresh_health_after_configure()
    return {"status": "success", "config": config.config}


@api_router.post("/reset_cameras")
async def reset_cameras():
    state: GlobalState = get_global_state()
    # await run_in_threadpool(state.acquisition.reset_cameras)
    await state.acquisition.reset_cameras()
    return {"status": "success"}


async def _set_camera_excluded(serial: str, excluded: bool) -> dict:
    """Add/remove a camera from the operator-excluded set, then full
    reconfigure so the change takes effect. Returns the new exclusion list.
    """
    state: GlobalState = get_global_state()
    if state.recording_status in _BUSY_RECORDING_STATES:
        raise HTTPException(
            status_code=409,
            detail="Cannot change camera exclusion while a recording is active.",
        )
    saved_config = getattr(state.acquisition, "config_file", None)
    current = set(getattr(state.acquisition, "excluded_serials", set()))
    if excluded:
        current.add(str(serial))
    else:
        current.discard(str(serial))
    state.acquisition.set_excluded_serials(current)

    action = "Excluded" if excluded else "Included"
    broadcast_event(
        event_type="session_insight",
        level="warn" if excluded else "ok",
        code=("camera_excluded" if excluded else "camera_included"),
        message=f"{action} camera {serial}; reconfiguring…",
    )

    await _reset_and_configure(saved_config, action=action)

    await _refresh_health_after_configure()

    return {
        "status": "success",
        "serial": serial,
        "excluded": excluded,
        "excluded_serials": sorted(current),
    }


@api_router.post("/cameras/{serial}/exclude")
async def exclude_camera(serial: str):
    """Remove a camera from this session's recordings without editing the
    YAML config. Recovery path for one bad camera that's holding up the
    rest of the rig (e.g. stuck-link case from the throughput-outlier
    finding). Excluded cameras are not Init'd, do not appear in the
    camera_status table, and are skipped by configure_cameras until
    re-included via /cameras/{serial}/include or until the system restarts.

    Returns 409 while a recording is active.
    """
    return await _set_camera_excluded(serial, excluded=True)


@api_router.post("/cameras/{serial}/include")
async def include_camera(serial: str):
    """Re-add an operator-excluded camera back into the recording set
    and reconfigure. Returns 409 while a recording is active.
    """
    return await _set_camera_excluded(serial, excluded=False)


@api_router.post("/cameras/{serial}/restore_defaults")
async def restore_camera_defaults(serial: str):
    """Restore one camera to factory defaults via UserSetLoad, then re-init
    the system so every camera picks up the saved config cleanly. Recovery
    path for the 'one camera latched a bad feature value
    (DeviceLinkThroughputLimit, LineMode, etc.) that survives soft restart'
    case — equivalent to SpinView's 'Restore Factory Defaults' followed by
    reconfigure.

    Returns 409 if recording, 404 if the serial isn't currently in cams.
    """
    state: GlobalState = get_global_state()
    if state.recording_status in _BUSY_RECORDING_STATES:
        raise HTTPException(
            status_code=409,
            detail="Cannot restore camera defaults while a recording is active.",
        )

    saved_config = getattr(state.acquisition, "config_file", None)

    broadcast_event(
        event_type="session_insight",
        level="warn",
        code="restore_defaults_started",
        message=f"Restoring factory defaults on camera {serial}…",
    )

    try:
        await run_in_threadpool(
            state.acquisition.restore_camera_defaults, serial
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    await _reset_and_configure(saved_config, action="Restore defaults")

    await _refresh_health_after_configure()

    broadcast_event(
        event_type="session_insight",
        level="ok",
        code="restore_defaults_complete",
        message=f"Camera {serial} restored to factory defaults and re-configured.",
    )

    return {"status": "success", "serial": serial, "config": saved_config}


class ForceIpData(BaseModel):
    ip: Optional[str] = None
    mask: Optional[str] = None
    gateway: Optional[str] = None


@api_router.post("/cameras/{mac}/force_ip")
async def force_camera_ip(mac: str, data: ForceIpData):
    """Force a camera that's on the wrong subnet to a target IP via PySpin's
    GigE Vision ForceIP. Identified by MAC because the serial isn't readable
    until the camera is on a reachable subnet.

    Body fields are optional. With no body the camera goes to Auto-Force-IP
    (next free address on the configured subnet) — the preferred path for
    operators who don't want to pick an IP. Returns 409 while recording,
    404 if no camera with that MAC is on the wrong subnet right now.
    """
    state: GlobalState = get_global_state()
    if state.recording_status in _BUSY_RECORDING_STATES:
        raise HTTPException(
            status_code=409,
            detail="Cannot Force IP while a recording is active.",
        )

    broadcast_event(
        event_type="session_insight",
        level="warn",
        code="force_ip_started",
        message=f"Forcing IP on camera {mac}…",
    )

    kwargs: dict = {"mac": mac}
    if data.ip is not None:
        kwargs["ip"] = data.ip
    if data.mask is not None:
        kwargs["mask"] = data.mask
    if data.gateway is not None:
        kwargs["gateway"] = data.gateway

    try:
        result = await run_in_threadpool(
            lambda: state.acquisition.force_camera_ip(**kwargs)
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # noqa: BLE001
        broadcast_event(
            event_type="session_insight",
            level="error",
            code="force_ip_failed",
            message=f"Force IP on {mac} failed: {e}",
            details={
                "remediation": [
                    "Check that the camera is on a directly-connected interface.",
                    "Run `make reset` to power-cycle the camera.",
                ],
            },
        )
        raise HTTPException(status_code=500, detail=str(e))

    await _refresh_health_after_configure()

    broadcast_event(
        event_type="session_insight",
        level="ok",
        code="force_ip_complete",
        message=(
            f"Camera {mac} is being re-IP'd; it should reappear in the "
            "Diagnostics tab within a few seconds."
        ),
    )

    return {"status": "success", **result}


@api_router.post("/restart_acquisition")
async def restart_acquisition():
    """Tear down the PySpin system and reinitialize from the saved config.

    Distinct from /reset_cameras (which power-cycles each camera via
    DeviceReset) — this only re-creates the software stack and re-runs
    the PTP sync setup in configure_cameras. Recovery path for the
    timestamp-spread > 1.0 / drifted-PTP case.

    Returns 409 while a recording is active.
    """
    state: GlobalState = get_global_state()
    if state.recording_status in _BUSY_RECORDING_STATES:
        raise HTTPException(
            status_code=409,
            detail="Cannot restart acquisition while a recording is active.",
        )

    # config_file gets cleared by close(); capture before reset().
    saved_config = getattr(state.acquisition, "config_file", None)

    broadcast_event(
        event_type="session_insight",
        level="warn",
        code="restart_started",
        message="Restarting acquisition system…",
    )

    await _reset_and_configure(saved_config, action="Restart")

    await _refresh_health_after_configure()

    broadcast_event(
        event_type="session_insight",
        level="ok",
        code="restart_complete",
        message="Acquisition system restarted.",
    )

    return {"status": "success", "config": saved_config}


# create an endpoint that exposes the camera statuses
@api_router.get("/camera_status")
async def get_camera_status() -> List[CameraStatus]:
    state: GlobalState = get_global_state()
    camera_status = state.acquisition.get_camera_status()
    return camera_status


@api_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
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


@api_router.websocket("/ws/diagnostics")
async def diagnostics_websocket_endpoint(websocket: WebSocket):
    """Push channel for structured diagnostic envelopes.

    Separate from /api/v1/ws (which carries recording {status, progress})
    so a bursty diagnostic broadcast cannot delay the recording status writer.
    """
    await diagnostics_manager.connect(websocket)
    try:
        while not AppStatus.should_exit:
            try:
                await asyncio.wait_for(websocket.receive(), timeout=0.5)
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect as e:
        logger.debug("Diagnostics websocket disconnected: %s", e)
    except RuntimeError as e:
        logger.debug("Diagnostics websocket disconnected (RuntimeError): %s", e)
    finally:
        diagnostics_manager.disconnect(websocket)


class SelectCameraData(BaseModel):
    camera_index: int | None = None


@api_router.post("/select_camera")
async def select_camera(data: SelectCameraData):
    state: GlobalState = get_global_state()
    state.selected_camera = data.camera_index
    return {"selected_camera": data.camera_index}


@api_router.get("/recording_status", response_model=str)
async def get_recording_status() -> str:
    state: GlobalState = get_global_state()
    return state.recording_status


def downsample_image(image, new_width, new_height):
    return cv2.resize(image, (new_width, new_height))


def convert_rgb_to_jpeg(frame):
    ret, jpeg_image = cv2.imencode(".jpg", frame)
    return jpeg_image.tobytes()


def draw_serial_label(frame, serial: str):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, w / 600)
    thickness = max(1, int(scale * 2))
    pos = (10, int(30 * scale + 10))
    cv2.putText(frame, serial, pos, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(
        frame, serial, pos, font, scale, (255, 255, 255), thickness, cv2.LINE_AA
    )


async def receive_frames(frames):
    state: GlobalState = get_global_state()
    if not state.preview_queue.empty():
        return

    if frames is None:
        print("Received empty frame")
        return

    if state.selected_camera is not None:
        idx = state.selected_camera
        if 0 <= idx < len(frames):
            frame, conversion, serial = frames[idx]
            rgb_frame = cv2.cvtColor(frame, conversion)
            h, w = rgb_frame.shape[:2]
            target_width = 1080
            target_height = int(target_width * h / w)
            downsampled = downsample_image(rgb_frame, target_width, target_height)
            draw_serial_label(downsampled, serial)
            jpeg_image = convert_rgb_to_jpeg(downsampled)
            await state.preview_queue.put(jpeg_image)
            return

    num_frames = len(frames)
    grid_width = math.ceil(math.sqrt(num_frames))
    grid_height = math.ceil(num_frames / grid_width)

    # Convert each frame to RGB and store in a list
    rgb_frames = []
    serials = []
    for frame, conversion, serial in frames:
        rgb_frames.append(cv2.cvtColor(frame, conversion))
        serials.append(serial)

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
        draw_serial_label(resized_frame, serials[i])

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
    state: GlobalState = get_global_state()
    async def generate_frames():
        while True:
            # Wait for the next frame to become available
            try:
                frame = await asyncio.wait_for(state.preview_queue.get(), timeout=2.5)
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
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")

    return StreamingResponse(
        generate_frames(),
        status_code=206,
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


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
    filename: str = Query(None, description="Name of the file to be used."),
    downsample: int = Query(default=5),
) -> SMPLData:
    from .annotation import get_mesh

    res = get_mesh(filename, downsample)
    return SMPLData(**res)


# code for single person SMPL reconstructions


class ReconstructionTrials(BaseModel):
    participant_id: str
    session_date: date
    video_base_filename: str


@api_router.get("/smpl_trials", response_model=List[ReconstructionTrials])
async def get_smpl_trials(
    model: str = Query(default="smpl", description="Type of SMPL model to use."),
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

    return SMPLData(
        ids=[0], frames=-1, type="smpl", faces=res["faces"], meshes=res["vertices"]
    )


app.include_router(api_router)


def register_exception(app: FastAPI):
    from fastapi import Request, status
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
        # or logger.error(f'{exc}')
        # logger.error(request, exc_str)
        print(request, exc_str)
        content = {"status_code": 10422, "message": exc_str, "data": None}
        return JSONResponse(
            content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )


register_exception(app)

if __name__ == "__main__":
    import uvicorn

    # Start the server
    uvicorn.run(
        "multi_camera.backend.fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        timeout_keep_alive=30,
        # log_level="debug",
    )
