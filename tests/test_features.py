"""Tests for form / H2H feature helpers."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.database import Base
from src.feature_engineering import compute_h2h, compute_team_form, days_since_last_match
from src.models import Match


@pytest.fixture()
def fx_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _add(
    session: Session,
    mid: str,
    dt: datetime,
    home: str,
    away: str,
    hs: int,
    as_: int,
) -> None:
    session.add(
        Match(
            id=mid,
            date=dt,
            home_team=home,
            away_team=away,
            home_score=hs,
            away_score=as_,
            status="completed",
        )
    )
    session.commit()


def test_team_form_points(fx_session: Session) -> None:
    t0 = datetime(2024, 1, 1, 15, 0, 0)
    _add(fx_session, "1", t0, "A", "B", 2, 0)
    _add(fx_session, "2", t0 + timedelta(days=7), "C", "A", 1, 1)
    before = datetime(2025, 1, 1, 12, 0, 0)
    f = compute_team_form(fx_session, "A", window=10, before=before)
    assert f.games == 2
    assert f.points == 4  # win + draw


def test_days_since_last_match_no_timedelta_tzinfo_bug(fx_session: Session) -> None:
    """Regression: subtract datetimes only; never read .tzinfo on a timedelta."""
    t0 = datetime(2024, 1, 1, 15, 0, 0)
    _add(fx_session, "r1", t0, "A", "B", 1, 0)
    before = datetime(2024, 1, 10, 12, 0, 0)
    d = days_since_last_match(fx_session, "A", before=before)
    assert d == 8  # calendar days between 2024-01-01 15:00 and 2024-01-10 12:00


def test_h2h_counts_wins_for_fixture_home(fx_session: Session) -> None:
    t0 = datetime(2024, 1, 1, 15, 0, 0)
    _add(fx_session, "1", t0, "Liv", "Ars", 2, 1)
    _add(fx_session, "2", t0 + timedelta(days=3), "Ars", "Liv", 0, 0)
    before = datetime(2025, 1, 1, 12, 0, 0)
    h = compute_h2h(fx_session, "Liv", "Ars", limit=5, before=before)
    assert h.meetings == 2
    assert h.home_wins == 1
    assert h.draws == 1
    assert h.away_wins == 0
