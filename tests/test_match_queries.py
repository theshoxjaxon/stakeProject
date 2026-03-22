"""Tests for future-only match selection used in predictions."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.models import Base
from src.match_queries import matches_for_prediction, utc_now
from src.models import Match


@pytest.fixture()
def memory_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
        session.rollback()


def _insert(
    session: Session,
    match_id: str,
    kickoff: datetime,
) -> None:
    session.add(
        Match(
            id=match_id,
            date=kickoff,
            home_team="A",
            away_team="B",
            status="scheduled",
        )
    )
    session.commit()


def test_matches_for_prediction_excludes_past_kickoffs(memory_session: Session) -> None:
    now = utc_now()
    past = now - timedelta(days=7)
    future = now + timedelta(days=3)

    _insert(memory_session, "m1", past)
    _insert(memory_session, "m2", future)

    rows = (
        memory_session.execute(matches_for_prediction(horizon_days=365)).scalars().all()
    )
    ids = {m.id for m in rows}
    assert ids == {"m2"}


def test_matches_for_prediction_horizon_upper_bound(memory_session: Session) -> None:
    now = utc_now()
    _insert(memory_session, "near", now + timedelta(days=1))
    _insert(memory_session, "far", now + timedelta(days=30))

    rows = (
        memory_session.execute(matches_for_prediction(horizon_days=7)).scalars().all()
    )
    ids = {m.id for m in rows}
    assert ids == {"near"}


def test_completed_status_excluded_even_if_future(memory_session: Session) -> None:
    """Terminal statuses should not appear in prediction list."""
    now = utc_now()
    future = now + timedelta(days=2)
    memory_session.add(
        Match(
            id="done",
            date=future,
            home_team="X",
            away_team="Y",
            home_score=1,
            away_score=0,
            status="completed",
        )
    )
    memory_session.add(
        Match(
            id="soon",
            date=future + timedelta(hours=3),
            home_team="A",
            away_team="B",
            status="scheduled",
        )
    )
    memory_session.commit()
    rows = (
        memory_session.execute(matches_for_prediction(horizon_days=365)).scalars().all()
    )
    assert {r.id for r in rows} == {"soon"}


def test_naive_utc_datetime_still_filtered(memory_session: Session) -> None:
    """SQLite often stores naive datetimes; comparison still separates past vs future."""
    now = utc_now()
    past_naive = (now - timedelta(days=14)).replace(tzinfo=None)
    future_naive = (now + timedelta(days=14)).replace(tzinfo=None)
    _insert(memory_session, "old", past_naive)
    _insert(memory_session, "new", future_naive)

    rows = (
        memory_session.execute(matches_for_prediction(horizon_days=365)).scalars().all()
    )
    ids = {m.id for m in rows}
    assert ids == {"new"}
