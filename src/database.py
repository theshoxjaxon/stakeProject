"""Database engine helpers for the football prediction system."""

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import DATABASE_PATH
from src.models import Base


def get_engine(database_path: Path | str = DATABASE_PATH):
    """Create and return a SQLAlchemy engine for the configured SQLite database."""
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", echo=False)


SessionLocal = sessionmaker(bind=get_engine())


@contextmanager
def get_session() -> Iterator[Session]:
    """Context-managed session bound to the default engine."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db(database_path: Path | str = DATABASE_PATH):
    """Create all tables in the database if they do not already exist."""
    engine = get_engine(database_path)
    Base.metadata.create_all(engine)
    return engine
