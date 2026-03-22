"""Database engine helpers for the football prediction system."""

from pathlib import Path
from typing import Iterable

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.config import DATABASE_PATH
from src.models import Base, TeamRating

# Re-export models for existing imports: `from src.database import Match, ...`
from src.models import Match, Odds, Team  # noqa: F401

__all__ = [
    "Base",
    "Team",
    "Match",
    "Odds",
    "TeamRating",
    "get_engine",
    "init_db",
    "ensure_team_ratings",
]


def get_engine(database_path: Path | str = DATABASE_PATH):
    """Create and return a SQLAlchemy engine for the configured SQLite database."""
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", echo=False)


def init_db(database_path: Path | str = DATABASE_PATH):
    """Create all tables in the database if they do not already exist."""
    engine = get_engine(database_path)
    Base.metadata.create_all(engine)
    return engine


def ensure_team_ratings(
    session: Session, team_names: Iterable[str], default_elo: float = 1500.0
) -> None:
    """Ensure a TeamRating row exists for every team name in the iterable."""
    cleaned = {t for t in team_names if t}
    if not cleaned:
        return

    existing = {
        name
        for (name,) in session.query(TeamRating.team_name)
        .filter(TeamRating.team_name.in_(cleaned))
        .all()
    }

    for name in cleaned - existing:
        session.add(TeamRating(team_name=name, elo_rating=default_elo))
