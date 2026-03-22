"""Shared SQLAlchemy selections for matches shown in prediction and reporting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import Select, or_, select

from src.models import Match

# Terminal / non-upcoming statuses — excluded from prediction lists
_TERMINAL_STATUSES = ("completed", "finished", "cancelled", "postponed", "abandoned")


def utc_now() -> datetime:
    """Wall-clock comparison time in UTC (matches Odds API / stored kickoff times)."""
    return datetime.now(timezone.utc)


def matches_for_prediction(horizon_days: int | None = None) -> Select[tuple[Match]]:
    """
    Matches that have not started yet: kickoff strictly after now.

    ``horizon_days`` (when set) adds an upper bound: kickoff on or before
    now + horizon. When ``None``, only the lower bound (future-only) applies.
    """
    from src.config import PREDICTION_HORIZON_DAYS

    if horizon_days is None:
        horizon_days = PREDICTION_HORIZON_DAYS

    now = utc_now()
    conditions = [
        Match.date > now,
        or_(
            Match.status.is_(None),
            ~Match.status.in_(_TERMINAL_STATUSES),
        ),
    ]
    if horizon_days is not None:
        conditions.append(Match.date <= now + timedelta(days=horizon_days))

    # No .limit() — return every future, non-terminal match (optionally bounded by horizon)
    return select(Match).where(*conditions).order_by(Match.date.asc())
