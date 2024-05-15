from sqlalchemy import create_engine, Boolean, Column, Integer, String, Date, DateTime, ForeignKey
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


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_date = Column(Date)
    session_path = Column(String)
    participant_id = Column(Integer, ForeignKey("participants.id"))

    participant = relationship("Participant", back_populates="sessions")
    recordings = relationship("Recording", back_populates="session")
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

    # if date is a string, convert to a python date type
    if isinstance(session_date, str):
        from datetime import datetime

        session_date = datetime.strptime(session_date, "%Y-%m-%d").date()

    # Create or update the participant
    participant = db.query(Participant).filter(Participant.name == participant_name).first()
    if not participant:
        participant = Participant(name=participant_name)
        db.add(participant)
        db.flush()

    # Create or update the session
    session = (
        db.query(Session).filter(Session.participant_id == participant.id, Session.session_date == session_date).first()
    )
    if not session:
        session = Session(participant_id=participant.id, session_path=session_path, session_date=session_date)
        db.add(session)
        db.flush()

    # Create the recording
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


### Data access API with Pydantic models


class RecordingOut(BaseModel):
    filename: str
    recording_timestamp: datetime
    comment: Optional[str]
    config_file: Optional[str]
    should_process: bool
    timestamp_spread: Optional[float]


class SessionOut(BaseModel):
    session_date: date
    session_path: str
    recordings: List[RecordingOut]
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
    query = db.query(Participant).options(joinedload(Participant.sessions).joinedload(Session.recordings))

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

            # check if there is an imported entry for this session
            imported = db.query(Imported).filter(Imported.session_id == session.id).first()

            session_out = SessionOut(
                session_date=session.session_date,
                session_path=session.session_path,
                recordings=recording_out_list,
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
    from multi_camera.datajoint.sessions import import_session

    # get the list of recordings from the database with their comments
    # that match the participant and session date
    db_recordings: ParticipantOut = get_recordings(
        db, participant_name=participant_id, filter_by_session_date=session_date
    )
    assert len(db_recordings) == 1, "Did not find exactly one participant for this name."
    sessions: SessionOut = db_recordings[0].sessions
    assert len(sessions) == 1, "Did not find exactly one session for this participant and date."
    recordings: List[RecordingOut] = sessions[0].recordings

    # filter out the recordings that should not be processed and retain the
    # filename and comment
    recordings = [
        (rec.filename, rec.comment) for rec in recordings if rec.should_process and rec.comment != "calibration"
    ]

    print("Processing recordings: ", recordings)

    datajoint_external_path = get_datajoint_external_path()
    check_datajoint_external_mounted(datajoint_external_path)

    # TODO: confirm calibration has been performed

    participant_id = normalize_participant_id(participant_id)

    import_session(participant_id, session_date, video_project=video_project, recordings=recordings)

    synchronize_to_datajoint(db)


def get_db():
    from sqlalchemy.orm import Session, sessionmaker

    DATABASE_URL = "sqlite:///data/recordings.db"

    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(bind=engine)

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
