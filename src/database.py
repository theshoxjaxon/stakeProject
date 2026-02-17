"""SQLAlchemy database setup and models for the betting value detection system."""

from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class Team(Base):
    """Team model with ELO rating."""

    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    current_elo: Mapped[float] = mapped_column(Float, default=1500.0, nullable=False)


class Match(Base):
    """Match model for scheduled and completed games."""

    __tablename__ = "matches"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # Odds API event id
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    home_team: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    away_team: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="scheduled", nullable=False)

    odds: Mapped[list["Odds"]] = relationship("Odds", back_populates="match", cascade="all, delete-orphan")


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
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    match: Mapped["Match"] = relationship("Match", back_populates="odds")


def get_engine(database_path: Path | str):
    """Create and return a SQLAlchemy engine."""
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", echo=False)


def init_db(database_path: Path | str):
    """Create all tables in the database."""
    engine = get_engine(database_path)
    Base.metadata.create_all(engine)
    return engine
