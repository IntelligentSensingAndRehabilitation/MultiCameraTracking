"""
Restricted-access PHI database for storing Financial Identification Numbers (FIN).

This module manages a separate SQLite database (data/phi.db) that is isolated from the
main recording database (data/recordings.db). The separation provides filesystem-level
access control for PHI data — the file can be given different permissions, encryption,
or backup policies than the main database.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

PHI_DATABASE_URL = "sqlite:///data/phi.db"

Base = declarative_base()


class ParticipantFIN(Base):
    """Maps a participant to their Financial Identification Number (FIN).

    participant_name matches the name column in the main recording_db participants table.
    One FIN per participant — upserted on conflict.
    """

    __tablename__ = "participant_fin"

    id = Column(Integer, primary_key=True, index=True)
    participant_name = Column(String, nullable=False, unique=True)
    fin = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


def get_phi_db():
    """Create the PHI database engine, ensure tables exist, and return a session."""
    engine = create_engine(PHI_DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def store_fin(db, participant_name: str, fin: str) -> ParticipantFIN:
    """Upsert a FIN for the given participant. Overwrites if one already exists."""
    existing = db.query(ParticipantFIN).filter(
        ParticipantFIN.participant_name == participant_name
    ).first()

    if existing:
        existing.fin = fin
        existing.updated_at = datetime.now()
    else:
        existing = ParticipantFIN(participant_name=participant_name, fin=fin)
        db.add(existing)

    db.commit()
    db.refresh(existing)
    return existing


def get_fin(db, participant_name: str) -> Optional[str]:
    """Look up the FIN for a participant. Returns None if not found."""
    record = db.query(ParticipantFIN).filter(
        ParticipantFIN.participant_name == participant_name
    ).first()
    return record.fin if record else None
