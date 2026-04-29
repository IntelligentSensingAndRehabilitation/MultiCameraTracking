import os

from sqlalchemy import create_engine, Boolean, Column, Float, Integer, String, Date, DateTime, ForeignKey
from sqlalchemy.orm import relationship, declarative_base, joinedload
from typing import Union, Tuple, List, Optional
from pydantic import BaseModel
from datetime import date, datetime

Base = declarative_base()


class Participant(Base):
    __tablename__ = "participants"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)

    sessions = relationship("Session", back_populates="participant")
    fin_record = relationship("ParticipantFIN", uselist=False, back_populates="participant")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_date = Column(Date)
    session_path = Column(String)
    participant_id = Column(Integer, ForeignKey("participants.id"))
    fin = Column(String, nullable=True)

    participant = relationship("Participant", back_populates="sessions")
    recordings = relationship("Recording", back_populates="session")
    photos = relationship("Photo", back_populates="session")
    imported = relationship("Imported", uselist=False, back_populates="session")


class Recording(Base):
    __tablename__ = "recordings"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    filename = Column(String)
    recording_timestamp = Column(DateTime)  # Add recording_timestamp field
    comment = Column(String, nullable=True)  # Add comment field
    config_file = Column(String, nullable=True)  # Add config_file field
    should_process = Column(Boolean, default=True)  # Add should_process field with a default value of True
    timestamp_spread = Column(Integer, nullable=True)  # Add timestamp_spread field

    session = relationship("Session", back_populates="recordings")


class Imported(Base):
    """A row in table indicates that a session has been imported into DataJoint"""

    __tablename__ = "imported"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))

    session = relationship("Session", back_populates="imported")


class Photo(Base):
    """Tracks uploaded participant identification photos for a session"""

    __tablename__ = "photos"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    filename = Column(String)
    original_filename = Column(String)
    saved_path = Column(String)
    description = Column(String, nullable=True)
    file_size_mb = Column(Float)
    upload_timestamp = Column(DateTime)

    session = relationship("Session", back_populates="photos")


class ParticipantFIN(Base):
    """DEPRECATED: FIN is now stored per-session on `Session.fin`.

    Kept for one release to support rollback and to feed the one-shot migration in
    `_ensure_session_fin_column`. Do not write to this table; remove in a follow-up
    once the new path is verified in production.
    """

    __tablename__ = "participant_fin"

    id = Column(Integer, primary_key=True, index=True)
    participant_id = Column(Integer, ForeignKey("participants.id"), unique=True, nullable=False)
    fin = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    participant = relationship("Participant", back_populates="fin_record")


def _get_or_create_participant(db, participant_name: str) -> Participant:
    """Find or create a participant by name."""
    participant = db.query(Participant).filter(Participant.name == participant_name).first()
    if not participant:
        participant = Participant(name=participant_name)
        db.add(participant)
        db.flush()
    return participant


def _get_or_create_session(db: Session, participant_name: str, session_date, session_path: str):
    """Find or create a participant and session, returning the Session object."""
    if isinstance(session_date, str):
        from datetime import datetime
        session_date = datetime.strptime(session_date, "%Y-%m-%d").date()

    participant = _get_or_create_participant(db, participant_name)

    session = (
        db.query(Session).filter(Session.participant_id == participant.id, Session.session_date == session_date).first()
    )
    if not session:
        session = Session(participant_id=participant.id, session_path=session_path, session_date=session_date)
        db.add(session)
        db.flush()

    return session


def store_fin(db, participant_name: str, session_date, session_path: str, fin: str) -> "Session":
    """Set the FIN on the matching session row. Creates participant/session if needed."""
    session = _get_or_create_session(db, participant_name, session_date, session_path)
    session.fin = fin
    db.commit()
    db.refresh(session)
    return session


def get_fin(db, participant_name: str, session_date) -> Optional[str]:
    """Look up the FIN for a (participant, session_date). Returns None if not set."""
    if isinstance(session_date, str):
        session_date = datetime.strptime(session_date, "%Y-%m-%d").date()

    session = (
        db.query(Session)
        .join(Participant, Session.participant_id == Participant.id)
        .filter(Participant.name == participant_name, Session.session_date == session_date)
        .first()
    )
    return session.fin if session and session.fin else None


def add_recording(
    db: Session,
    participant_name: str,
    session_date: Date,
    session_path: str,
    filename: str,
    recording_timestamp: DateTime,
    config_file: Optional[str] = None,
    comment: Optional[str] = None,
    should_process: Optional[bool] = True,
    timestamp_spread: Optional[int] = None,
):
    print(
        "Adding recording to database: ",
        participant_name,
        session_date,
        session_path,
        filename,
        recording_timestamp,
        comment,
        config_file,
        should_process,
        timestamp_spread,
    )

    session = _get_or_create_session(db, participant_name, session_date, session_path)

    new_recording = Recording(
        session_id=session.id,
        filename=filename,
        recording_timestamp=recording_timestamp,
        comment=comment,
        config_file=config_file,
        should_process=should_process,
        timestamp_spread=timestamp_spread,
    )
    db.add(new_recording)
    db.commit()
    db.refresh(new_recording)
    return new_recording


def add_photo(
    db: Session,
    participant_name: str,
    session_date: Date,
    session_path: str,
    filename: str,
    original_filename: str,
    saved_path: str,
    file_size_mb: float,
    upload_timestamp: DateTime,
    description: Optional[str] = None,
):
    session = _get_or_create_session(db, participant_name, session_date, session_path)

    new_photo = Photo(
        session_id=session.id,
        filename=filename,
        original_filename=original_filename,
        saved_path=saved_path,
        description=description,
        file_size_mb=file_size_mb,
        upload_timestamp=upload_timestamp,
    )
    db.add(new_photo)
    db.commit()
    db.refresh(new_photo)
    return new_photo


### Data access API with Pydantic models


class RecordingOut(BaseModel):
    filename: str
    recording_timestamp: datetime
    comment: Optional[str]
    config_file: Optional[str]
    should_process: bool
    timestamp_spread: Optional[float]


class PhotoOut(BaseModel):
    filename: str
    original_filename: str
    saved_path: str
    description: Optional[str]
    file_size_mb: float
    upload_timestamp: datetime


class SessionOut(BaseModel):
    session_date: date
    session_path: str
    recordings: List[RecordingOut]
    photos: List[PhotoOut]
    imported: bool


class ParticipantOut(BaseModel):
    name: str
    sessions: List[SessionOut]


def get_recordings(
    db: Session,
    participant_name: Optional[str] = None,
    filter_by_session_date: Optional[date] = None,
    order_by_date: Optional[bool] = False,
) -> Union[List[ParticipantOut], Tuple[List[SessionOut], List[str]]]:
    query = db.query(Participant).options(
        joinedload(Participant.sessions).joinedload(Session.recordings),
        joinedload(Participant.sessions).joinedload(Session.photos),
    )

    if participant_name:
        query = query.filter(Participant.name == participant_name)

    participants = query.all()

    participant_out_list = []
    for participant in participants:
        session_out_list = []
        for session in participant.sessions:
            if filter_by_session_date and session.session_date != filter_by_session_date:
                continue

            recording_out_list = [
                RecordingOut(**{k: v for k, v in recording.__dict__.items() if k != "_sa_instance_state"})
                for recording in session.recordings
            ]

            photo_out_list = [
                PhotoOut(**{k: v for k, v in photo.__dict__.items() if k != "_sa_instance_state"})
                for photo in session.photos
            ]

            # check if there is an imported entry for this session
            imported = db.query(Imported).filter(Imported.session_id == session.id).first()

            session_out = SessionOut(
                session_date=session.session_date,
                session_path=session.session_path,
                recordings=recording_out_list,
                photos=photo_out_list,
                imported=imported is not None,
            )
            session_out_list.append(session_out)

        participant_out = ParticipantOut(name=participant.name, sessions=session_out_list)
        participant_out_list.append(participant_out)

    if order_by_date:
        session_out_list = []
        participant_names = []
        for participant_out in participant_out_list:
            for session_out in participant_out.sessions:
                session_out_list.append(session_out)
                participant_names.append(participant_out.name)

        # Sort sessions by date and create a tuple with participant names
        sorted_sessions = sorted(zip(session_out_list, participant_names), key=lambda x: x[0].session_date)
        sorted_session_out_list, sorted_participant_names = zip(*sorted_sessions)
        return list(sorted_session_out_list), list(sorted_participant_names)

    return participant_out_list


def modify_recording_entry(db: Session, participant: ParticipantOut, updated_recording: RecordingOut):
    # find the recording entry that matches by participant and file name and update the other fields

    query = db.query(Recording).join(Session).join(Participant)

    # first filter by participant name
    query = query.filter(Participant.name == participant.name)

    # then filter by filename
    query = query.filter(Recording.filename == updated_recording.filename)

    # get the recording entry
    recording = query.first()

    # now update the comment and should process fields
    recording.comment = updated_recording.comment
    recording.should_process = updated_recording.should_process

    # commit the changes
    db.commit()


def rename_recording_entry(db: Session, participant_name: str, old_filename: str, new_filename: str):
    """Rename a recording: update the filename in the database and rename the directory on disk.

    The new filename must share the same parent directory as the old filename
    (only the basename may change). Path traversal segments are rejected.
    """
    # Reject absolute paths and traversal segments
    if os.path.isabs(new_filename):
        raise ValueError("new_filename must not be an absolute path")
    if ".." in new_filename.split(os.sep):
        raise ValueError("new_filename must not contain '..' path segments")

    # Ensure the parent directory hasn't changed (only the basename may differ)
    if os.path.dirname(new_filename) != os.path.dirname(old_filename):
        raise ValueError("new_filename must be in the same directory as the original recording")

    query = db.query(Recording).join(Session).join(Participant)
    query = query.filter(Participant.name == participant_name)
    query = query.filter(Recording.filename == old_filename)

    recording = query.first()
    if recording is None:
        raise ValueError(f"Recording not found: participant={participant_name}, filename={old_filename}")

    # Check for collisions
    existing = db.query(Recording).filter(Recording.filename == new_filename).first()
    if existing is not None:
        raise FileExistsError(f"A recording with filename '{new_filename}' already exists in the database")
    if os.path.exists(new_filename):
        raise FileExistsError(f"Path already exists on disk: {new_filename}")

    # Rename directory on disk if it exists
    if os.path.exists(old_filename):
        os.rename(old_filename, new_filename)

    recording.filename = new_filename
    db.commit()


### DataJoint bridge


def normalize_participant_id(participant_id):
    # historically, my experiments use an identifier like pXXX where XXX is a number
    # the subject_id field was purely numeric. Now, we are allowing alphanumeric
    # identifiers (distinguished by the more modern term participant). We also use
    # tXXXX for test users. Right now having some stay purely numeric provides
    # easier consistency with data collected with the portable system.

    # detect if particpant_name has the format p## or t## where ## is numeric, in which
    # case strip off the first character
    if participant_id[0] in ("p", "t") and participant_id[1:].isnumeric():
        participant_id = participant_id[1:]

    return participant_id


def synchronize_to_datajoint(db: Session):
    """Synchronize the recording database with DataJoint"""

    from multi_camera.datajoint.sessions import Session as SessionDJ

    sessions = db.query(Session).all()

    for session in sessions:
        # see if there is an imported entry for this session or if imported is False

        participant_id = session.participant.name
        session_date = session.session_date

        participant_id = normalize_participant_id(participant_id)

        in_datajoint = len(SessionDJ & {"participant_id": participant_id, "session_date": session_date}) == 1
        imported = db.query(Imported).filter(Imported.session_id == session.id).first()

        print(
            "participant_id: ",
            participant_id,
            "session_date: ",
            session_date,
            "in_datajoint: ",
            in_datajoint,
            "imported: ",
            imported,
        )
        if not imported and in_datajoint:
            print("Session already imported", participant_id, session_date)

            # need to make an imported entry in SQLlite for this session
            imported_entry = Imported(session_id=session.id)
            db.add(imported_entry)
            db.commit()

        elif imported and not in_datajoint:
            print("Session must have been removed from DJ", participant_id, session_date)

            # need to delete the imported entry in SQLlite for this session
            db.delete(imported)
            db.commit()

def get_datajoint_external_path():
    import datajoint as dj

    return dj.config['stores']['localattach']['location']

def check_datajoint_external_mounted(mount_path, sentinel_file_name=".multi_cam_mount_check"):
    import os
    """Check if the external drive is mounted and has the sentinel file"""

    # Sentinel file can be used for other checks like comparing the files last modified date
    # or checking the contents of the sentinel file

    if not os.path.exists(mount_path):
        raise ValueError(f"External drive not mounted at {mount_path}. Please mount the external drive before starting the Multi Camera System.")

    sentinel_file = os.path.join(mount_path, sentinel_file_name)
    if not os.path.exists(sentinel_file):
        raise ValueError(f"{sentinel_file} not found. Please ensure the external drive is mounted before starting the Multi Camera System.")

    print("External drive mounted and sentinel file found.")

def push_to_datajoint(db: Session, participant_id: str, session_date: date, video_project: str):
    from multi_camera.datajoint.sessions import import_session, PhotoSpec

    # get the list of recordings from the database with their comments
    # that match the participant and session date
    db_recordings: ParticipantOut = get_recordings(
        db, participant_name=participant_id, filter_by_session_date=session_date
    )
    assert len(db_recordings) == 1, "Did not find exactly one participant for this name."
    sessions: SessionOut = db_recordings[0].sessions
    assert len(sessions) == 1, "Did not find exactly one session for this participant and date."
    recordings: List[RecordingOut] = sessions[0].recordings

    # SQLite stores the un-normalized participant name (Participant.name == subject_id
    # passed to /session), so look up FIN before normalization below.
    fin = get_fin(db, participant_name=participant_id, session_date=session_date)

    photos = [
        PhotoSpec(
            saved_path=p.saved_path,
            filename=p.filename,
            original_filename=p.original_filename,
            upload_timestamp=p.upload_timestamp,
            description=p.description,
        )
        for p in sessions[0].photos
    ]

    # filter out the recordings that should not be processed and retain the
    # filename and comment
    recordings = [
        (rec.filename, rec.comment) for rec in recordings if rec.should_process and rec.comment != "calibration" and rec.comment != "charuco"
    ]

    print("Processing recordings: ", recordings)

    datajoint_external_path = get_datajoint_external_path()
    check_datajoint_external_mounted(datajoint_external_path)

    # TODO: confirm calibration has been performed

    participant_id = normalize_participant_id(participant_id)

    import_session(
        participant_id,
        session_date,
        video_project=video_project,
        recordings=recordings,
        fin=fin,
        photos=photos,
    )

    synchronize_to_datajoint(db)


def _ensure_session_fin_column(engine):
    """Add `sessions.fin` column on existing DBs that pre-date #15.

    SQLAlchemy's create_all only creates missing tables, not missing columns.
    Best-effort backfill: copy ParticipantFIN.fin to the most recent session
    per participant. Older sessions are left NULL because the legacy
    ParticipantFIN table only kept the latest FIN per participant.
    """
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    if "sessions" not in insp.get_table_names():
        return
    cols = [c["name"] for c in insp.get_columns("sessions")]
    if "fin" in cols:
        return

    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE sessions ADD COLUMN fin VARCHAR"))
        if "participant_fin" in insp.get_table_names():
            conn.execute(text("""
                UPDATE sessions
                SET fin = (
                    SELECT pf.fin FROM participant_fin pf
                    WHERE pf.participant_id = sessions.participant_id
                )
                WHERE sessions.id IN (
                    SELECT MAX(id) FROM sessions GROUP BY participant_id
                )
            """))


def get_db():
    from sqlalchemy.orm import Session, sessionmaker

    DATABASE_URL = "sqlite:///data/recordings.db"

    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    _ensure_session_fin_column(engine)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = SessionLocal()

    return db


def test():
    from sqlalchemy.orm import Session, sessionmaker

    DATABASE_URL = "sqlite:///./test.db"

    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(bind=engine)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = SessionLocal()

    add_recording(db, "test_subject", "2023-04-01", "sesion_path", "testfile.mp4")
    add_recording(db, "test_subject2", "2023-04-01", "sesion_path", "testfile.mp4")
    add_recording(db, "test_subject", "2023-03-01", "sesion_path", "testfile.mp4")

    print("\n")
    print(get_recordings(db, participant_name="test_subject"))
    print("\n")
    print(get_recordings(db, participant_name="test_subject2"))
    print("\n")
    print(get_recordings(db, order_by_date=True))


if __name__ == "__main__":
    test()
