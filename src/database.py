"""Database models and engine helpers for the football prediction system."""

from datetime import datetime
from pathlib import Path
from typing import Iterable

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from src.config import DATABASE_PATH


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class Team(Base):
    """Legacy team model (kept for compatibility)."""

    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    current_elo: Mapped[float] = mapped_column(Float, default=1500.0, nullable=False)


class Match(Base):
    """Match model for scheduled and completed games."""

    __tablename__ = "matches"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # Odds API event id
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    home_team: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    away_team: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="scheduled", nullable=False, index=True)

    odds: Mapped[list["Odds"]] = relationship(
        "Odds",
        back_populates="match",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_matches_date_home_away", "date", "home_team", "away_team"),
    )


class Odds(Base):
    """Odds model for bookmaker lines (1X2: home, draw, away)."""

    __tablename__ = "odds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("matches.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    bookmaker: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    h_odds: Mapped[float] = mapped_column(Float, nullable=False)
    d_odds: Mapped[float] = mapped_column(Float, nullable=False)
    a_odds: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    match: Mapped["Match"] = relationship("Match", back_populates="odds")

    __table_args__ = (
        Index("ix_odds_match_bookmaker_time", "match_id", "bookmaker", "timestamp"),
    )


class TeamRating(Base):
    """Rating container for Elo and Poisson strengths."""

    __tablename__ = "team_ratings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    elo_rating: Mapped[float] = mapped_column(Float, nullable=False, default=1500.0)
    attack_strength: Mapped[float | None] = mapped_column(Float, nullable=True)
    defense_strength: Mapped[float | None] = mapped_column(Float, nullable=True)


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


def ensure_team_ratings(session: Session, team_names: Iterable[str], default_elo: float = 1500.0) -> None:
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
