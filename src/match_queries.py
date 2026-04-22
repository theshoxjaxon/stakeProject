"""Shared SQLAlchemy selections for matches shown in prediction and reporting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum

from sqlalchemy import Select, asc, desc, or_, select

from src.models import Match, Prediction

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


# ---------------------------------------------------------------------------
# Trading Terminal — bet history with dynamic DB-side sorting
# ---------------------------------------------------------------------------


class BetSortField(str, Enum):
    """Columns available for sorting the bet-history view."""

    DATE = "date"    # Match kick-off date
    PNL = "pnl"      # Realised profit / loss on this prediction
    RESULT = "result" # Win (True) or loss (False)


class SortDir(str, Enum):
    """Sort direction — values are intentionally JSON-safe lowercase strings."""

    ASC = "asc"
    DESC = "desc"


# Maps each sort field to the SQLAlchemy column used in order_by.
# Match.date requires the JOIN that get_bet_history already adds.
_SORT_COLUMN = {
    BetSortField.DATE:   Match.date,
    BetSortField.PNL:    Prediction.profit,
    BetSortField.RESULT: Prediction.was_win,
}


def get_bet_history(
    sort_by: BetSortField = BetSortField.DATE,
    sort_dir: SortDir = SortDir.DESC,
    settled_only: bool = True,
) -> Select[tuple[Prediction]]:
    """
    DB-side sorted selection of predictions for the Trading Terminal history view.

    Sorting is fully delegated to SQLite via ``order_by`` — no Python-side
    re-sorting required in the UI layer.

    Parameters
    ----------
    sort_by:
        Primary sort column (DATE, PNL, or RESULT).
    sort_dir:
        ASC or DESC.
    settled_only:
        When True (default), only returns predictions where result_settled=True.
        Pass False to include pending/unsettled bets (e.g. the "live" terminal view).

    Returns
    -------
    A SQLAlchemy ``Select`` ready to be executed with ``session.execute()``.
    The query yields ``Prediction`` ORM objects; access ``prediction.match``
    for kick-off date, team names, and final score.
    """
    direction = desc if sort_dir == SortDir.DESC else asc
    primary_col = _SORT_COLUMN[sort_by]

    # Always add Match.date DESC as a stable tiebreaker so identical PNL / result
    # rows are deterministically ordered by most-recent fixture first.
    order_clauses = [direction(primary_col)]
    if sort_by != BetSortField.DATE:
        order_clauses.append(desc(Match.date))

    stmt = (
        select(Prediction)
        .join(Match, Prediction.match_id == Match.id)
        .order_by(*order_clauses)
    )

    if settled_only:
        stmt = stmt.where(Prediction.result_settled.is_(True))

    return stmt
