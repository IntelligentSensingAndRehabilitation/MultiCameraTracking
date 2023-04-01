from sqlalchemy import create_engine, Boolean, Column, Integer, String, Date, ForeignKey
from sqlalchemy.orm import relationship, declarative_base, joinedload
from typing import Union, Tuple, List, Optional
from pydantic import BaseModel
from datetime import date

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


class Recording(Base):
    __tablename__ = "recordings"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    filename = Column(String)
    comment = Column(String, nullable=True)  # Add comment field
    config_file = Column(String, nullable=True)  # Add config_file field
    should_process = Column(Boolean, default=True)  # Add should_process field with a default value of True

    session = relationship("Session", back_populates="recordings")


def add_recording(
    db: Session,
    participant_name: str,
    session_date: Date,
    session_path: str,
    filename: str,
    config_file: Optional[str] = None,
    comment: Optional[str] = None,
    should_process: Optional[bool] = True,
):
    print(
        "Adding recording to database: ",
        participant_name,
        session_date,
        session_path,
        filename,
        comment,
        config_file,
        should_process,
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
        comment=comment,
        config_file=config_file,
        should_process=should_process,
    )
    db.add(new_recording)
    db.commit()
    db.refresh(new_recording)
    return new_recording


### Data access API with Pydantic models


class RecordingOut(BaseModel):
    filename: str
    comment: Optional[str]
    config_file: Optional[str]
    should_process: bool


class SessionOut(BaseModel):
    session_date: date
    session_path: str
    recordings: List[RecordingOut]


class ParticipantOut(BaseModel):
    name: str
    sessions: List[SessionOut]


def get_recordings(
    db: Session, participant_name: Optional[str] = None, order_by_date: Optional[bool] = False
) -> Union[List[ParticipantOut], Tuple[List[SessionOut], List[str]]]:
    query = db.query(Participant).options(joinedload(Participant.sessions).joinedload(Session.recordings))

    if participant_name:
        query = query.filter(Participant.name == participant_name)

    participants = query.all()

    participant_out_list = []
    for participant in participants:
        session_out_list = []
        for session in participant.sessions:
            recording_out_list = [
                RecordingOut(**{k: v for k, v in recording.__dict__.items() if k != "_sa_instance_state"})
                for recording in session.recordings
            ]
            session_out = SessionOut(
                session_date=session.session_date, session_path=session.session_path, recordings=recording_out_list
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
