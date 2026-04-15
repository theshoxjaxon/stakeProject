"""Database engine helpers for the football prediction system."""

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import DATABASE_PATH
from src.models import Base

# 1. Define the Engine Helper so Sessionmaker can use it


def get_engine(database_path: Path | str = DATABASE_PATH):
    """Create and return a SQLAlchemy engine for the configured SQLite database."""
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", echo=False)


# 3. Define SessionLocal NOW that get_engine exists
SessionLocal = sessionmaker(bind=get_engine())

# 4. Database Management Functions


def init_db(database_path: Path | str = DATABASE_PATH):
    """Create all tables in the database if they do not already exist."""
    engine = get_engine(database_path)
    Base.metadata.create_all(engine)
    return engine
