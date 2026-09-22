"""SQLAlchemy ORM models — single source of truth for schema (Alembic + runtime)."""

import uuid as uuid_pkg
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import CITEXT, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class Team(Base):
    """Legacy team model (kept for compatibility)."""

    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
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
    status: Mapped[str] = mapped_column(
        String(50), default="scheduled", nullable=False, index=True
    )
    sport_key: Mapped[str | None] = mapped_column(
        String(50), nullable=True, index=True
    )

    odds: Mapped[list["Odds"]] = relationship(
        "Odds",
        back_populates="match",
        cascade="all, delete-orphan",
    )
    
    # Add relationship to predictions
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction",
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
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )

    match: Mapped["Match"] = relationship("Match", back_populates="odds")

    __table_args__ = (
        Index("ix_odds_match_bookmaker_time", "match_id", "bookmaker", "timestamp"),
    )


class TeamRating(Base):
    """Rating container for Elo and Poisson strengths."""

    __tablename__ = "team_ratings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_name: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    elo_rating: Mapped[float] = mapped_column(Float, nullable=False, default=1500.0)
    attack_strength: Mapped[float | None] = mapped_column(Float, nullable=True)
    defense_strength: Mapped[float | None] = mapped_column(Float, nullable=True)


# ============= NEW TABLES FOR PREDICTION TRACKING =============

class Prediction(Base):
    """Store model predictions for future tracking."""
    
    __tablename__ = "predictions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("matches.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )

    # Denormalized copies of Match.date / Match.sport_key — lets the "today's
    # predictions" query hit a single index instead of joining matches.
    kickoff: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    league: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Model probabilities (0-1)
    home_prob: Mapped[float] = mapped_column(Float, nullable=False)
    draw_prob: Mapped[float] = mapped_column(Float, nullable=False)
    away_prob: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Market odds (from the odds table, stored here for snapshot)
    market_home: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_draw: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_away: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Fair odds (1 / probability)
    fair_home: Mapped[float | None] = mapped_column(Float, nullable=True)
    fair_draw: Mapped[float | None] = mapped_column(Float, nullable=True)
    fair_away: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Edges (percentage)
    edge_home: Mapped[float | None] = mapped_column(Float, nullable=True)
    edge_draw: Mapped[float | None] = mapped_column(Float, nullable=True)
    edge_away: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Bet recommendation
    recommended_selection: Mapped[str | None] = mapped_column(String(10), nullable=True)  # 'home', 'draw', 'away'
    recommended_stake_percent: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # % of bankroll
    recommended_stake_amount: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # actual currency
    edge_used: Mapped[float | None] = mapped_column(Float, nullable=True)  # edge that triggered bet
    
    # Results tracking
    actual_outcome: Mapped[str | None] = mapped_column(String(10), nullable=True)  # 'home', 'draw', 'away'
    was_win: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    profit: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    result_settled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    match: Mapped["Match"] = relationship("Match", back_populates="predictions")
    bet: Mapped["Bet | None"] = relationship("Bet", back_populates="prediction", uselist=False, cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("ix_predictions_match_created", "match_id", "created_at"),
        Index("ix_predictions_result_settled", "result_settled"),
        Index("ix_predictions_kickoff_league", "kickoff", "league"),
    )


class Bet(Base):
    """Store actual bets placed (for tracking profit/loss)."""
    
    __tablename__ = "bets"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,  # One bet per prediction
        index=True,
    )
    placed_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )
    
    # Bet details
    selection: Mapped[str] = mapped_column(String(10), nullable=False)  # 'home', 'draw', 'away'
    odds: Mapped[float] = mapped_column(Float, nullable=False)
    stake_amount: Mapped[float] = mapped_column(Float, nullable=False)  # actual currency
    stake_percent: Mapped[float] = mapped_column(Float, nullable=False)  # % of bankroll
    
    # Settlement
    status: Mapped[str] = mapped_column(String(20), default='pending', nullable=False, index=True)  # 'pending', 'won', 'lost', 'void'
    profit: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    prediction: Mapped["Prediction"] = relationship("Prediction", back_populates="bet")

    __table_args__ = (
        Index("ix_bets_status_settled", "status", "settled_at"),
    )

# ============= NEW TABLES FOR ADVANCED ANALYTICS =============

class TeamStats(Base):
    """Store rolling aggregates of advanced team statistics."""

    __tablename__ = "team_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    
    # Rolling window (e.g., last 5, 10 matches)
    window_games: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    
    # Aggregate stats over the window
    avg_xg_for: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_xg_against: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_shots_for: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_shots_against: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_team_stats_team_window", "team_name", "window_games", unique=True),
    )


class MatchAdvanced(Base):
    """Store advanced per-match statistics (e.g., from FBref)."""

    __tablename__ = "match_advanced"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("matches.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,  # One record per match
        index=True,
    )
    
    home_xg: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_xg: Mapped[float | None] = mapped_column(Float, nullable=True)
    home_xgot: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_xgot: Mapped[float | None] = mapped_column(Float, nullable=True)
    home_shots: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_shots: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_possession: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_possession: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="fbref") # e.g. 'fbref', 'understat'
    
    match: Mapped["Match"] = relationship("Match")


class PlayerInjury(Base):
    """Track key player injuries and their expected return."""

    __tablename__ = "player_injuries"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    team_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    
    injury_description: Mapped[str] = mapped_column(String(255), nullable=False)
    expected_return: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Player importance (e.g., 1-5, 5 being critical)
    importance: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    
    # Status (e.g., 'out', 'doubtful', 'suspended')
    status: Mapped[str] = mapped_column(String(50), nullable=False, default='out')
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_player_injuries_team_status", "team_name", "status"),
    )


# ============= MULTI-TENANT (B2B) SCHEMA =============
#
# organizations are the tenant. Every user belongs to exactly one org
# (role is per-user within that org). API keys and usage events are
# org-scoped, not user-scoped, so a whole org shares one quota/plan.
#
# `role` (per-user permissions) and `plan` (org billing tier) are deliberately
# separate columns on separate tables — never combine them into one field.


class Organization(Base):
    """Tenant. Owns users, api keys, and usage events; billed on `plan`."""

    __tablename__ = "organizations"

    id: Mapped[uuid_pkg.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid_pkg.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    plan: Mapped[str] = mapped_column(
        String(20), nullable=False, default="free", server_default="free"
    )
    plan_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    users: Mapped[list["User"]] = relationship(
        "User", back_populates="organization", cascade="all, delete-orphan"
    )
    api_keys: Mapped[list["ApiKey"]] = relationship(
        "ApiKey", back_populates="organization", cascade="all, delete-orphan"
    )
    usage_events: Mapped[list["UsageEvent"]] = relationship(
        "UsageEvent", back_populates="organization", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("plan IN ('free', 'pro', 'premium')", name="ck_organizations_plan"),
    )


class User(Base):
    """A person belonging to exactly one organization. `role` is per-user, not per-org."""

    __tablename__ = "users"

    id: Mapped[uuid_pkg.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid_pkg.uuid4
    )
    org_id: Mapped[uuid_pkg.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    email: Mapped[str] = mapped_column(CITEXT, nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(
        String(20), nullable=False, default="member", server_default="member"
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=text("true")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    organization: Mapped["Organization"] = relationship("Organization", back_populates="users")
    refresh_tokens: Mapped[list["RefreshToken"]] = relationship(
        "RefreshToken", back_populates="user", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("role IN ('owner', 'admin', 'member')", name="ck_users_role"),
    )


class ApiKey(Base):
    """Org-scoped API key. Only the SHA-256 hash is ever stored — never the raw key."""

    __tablename__ = "api_keys"

    id: Mapped[uuid_pkg.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid_pkg.uuid4
    )
    org_id: Mapped[uuid_pkg.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    prefix: Mapped[str] = mapped_column(String(12), nullable=False)
    scopes: Mapped[list] = mapped_column(JSONB, nullable=False, default=list, server_default="[]")
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    organization: Mapped["Organization"] = relationship("Organization", back_populates="api_keys")


class RefreshToken(Base):
    """User-scoped refresh token. Only the hash is stored; rotation tracked via `replaced_by`."""

    __tablename__ = "refresh_tokens"

    id: Mapped[uuid_pkg.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid_pkg.uuid4
    )
    user_id: Mapped[uuid_pkg.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    replaced_by: Mapped[uuid_pkg.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("refresh_tokens.id", ondelete="SET NULL"),
        nullable=True,
    )

    user: Mapped["User"] = relationship("User", back_populates="refresh_tokens")


class UsageEvent(Base):
    """Append-only, org-scoped API usage log. High write volume — integer PK, not UUID."""

    __tablename__ = "usage_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    org_id: Mapped[uuid_pkg.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )
    endpoint: Mapped[str] = mapped_column(String(255), nullable=False)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    organization: Mapped["Organization"] = relationship(
        "Organization", back_populates="usage_events"
    )

    __table_args__ = (Index("ix_usage_events_org_ts", "org_id", "ts"),)
