#!/usr/bin/env python3
"""Insert minimal development rows into the SQLite database (idempotent-ish)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from src.config import DATABASE_PATH, PROJECT_ROOT
from src.database import get_engine
from src.models import Match, Odds, Team, TeamRating


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def seed(session: Session) -> None:
    """Add sample teams, ratings, one match, and one odds row if missing."""
    if session.query(Team).filter(Team.name == "Dev United").first() is None:
        session.add(Team(name="Dev United", current_elo=1520.0))
    if session.query(Team).filter(Team.name == "Test City").first() is None:
        session.add(Team(name="Test City", current_elo=1480.0))

    for name, elo in (("Dev United", 1520.0), ("Test City", 1480.0)):
        row = session.query(TeamRating).filter(TeamRating.team_name == name).first()
        if row is None:
            session.add(TeamRating(team_name=name, elo_rating=elo, attack_strength=1.1, defense_strength=0.95))

    match_id = "seed_dev_001"
    if session.query(Match).filter(Match.id == match_id).first() is None:
        kickoff = _utc(datetime(2025, 8, 15, 15, 0, 0))
        session.add(
            Match(
                id=match_id,
                date=kickoff,
                home_team="Dev United",
                away_team="Test City",
                home_score=None,
                away_score=None,
                status="scheduled",
            )
        )

    if session.query(Odds).filter(Odds.match_id == match_id, Odds.bookmaker == "samplebook").first() is None:
        session.add(
            Odds(
                match_id=match_id,
                bookmaker="samplebook",
                h_odds=2.10,
                d_odds=3.40,
                a_odds=3.50,
                timestamp=_utc(datetime(2025, 8, 14, 12, 0, 0)),
            )
        )

    session.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed development data into SQLite.")
    parser.add_argument(
        "--database",
        type=Path,
        default=DATABASE_PATH,
        help=f"SQLite file (default: {DATABASE_PATH})",
    )
    args = parser.parse_args()
    db_path = args.database
    if not db_path.is_absolute():
        db_path = (PROJECT_ROOT / db_path).resolve()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = get_engine(db_path)
    with Session(engine) as session:
        seed(session)
    print(f"Seed complete: {db_path}")


if __name__ == "__main__":
    main()
