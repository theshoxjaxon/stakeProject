"""Database engine helpers for the football prediction system."""

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import DATABASE_URL
from src.models import Base


def get_engine(database_url: str = DATABASE_URL):
    """Create and return a SQLAlchemy engine for the configured PostgreSQL database."""
    return create_engine(database_url, echo=False)


SessionLocal = sessionmaker(bind=get_engine())


@contextmanager
def get_session() -> Iterator[Session]:
    """Context-managed session bound to the default engine."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db(database_url: str = DATABASE_URL):
    """Create all tables in the database if they do not already exist."""
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return engine
