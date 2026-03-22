"""Burn in Elo ratings using historical completed match results."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import DATABASE_PATH
from src.database import get_engine, init_db
from src.models import Match, Team
from src.elo_model import init_ratings, update_ratings


def _result_from_scores(home_score: int, away_score: int) -> str:
    """Determine result: 'H' (home win), 'D' (draw), 'A' (away win)."""
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def ensure_teams_from_matches(session: Session, matches: list[Match]) -> None:
    """Ensure all teams from matches exist in the teams table."""
    team_names: set[str] = set()
    for m in matches:
        team_names.add(m.home_team)
        team_names.add(m.away_team)
    for name in team_names:
        existing = session.execute(
            select(Team).where(Team.name == name)
        ).scalar_one_or_none()
        if existing is None:
            session.add(Team(name=name, current_elo=1500.0))


def run_backfill_elo() -> int:
    """
    Load completed matches, update Elo ratings in chronological order.
    Returns count of matches processed.
    """
    init_db(DATABASE_PATH)
    engine = get_engine(DATABASE_PATH)

    with Session(engine) as session:
        stmt = (
            select(Match)
            .where(
                Match.status.in_(["completed", "finished"]),
                Match.home_score.isnot(None),
                Match.away_score.isnot(None),
            )
            .order_by(Match.date.asc())
        )
        matches = list(session.execute(stmt).scalars().all())

        if not matches:
            print("No completed matches with scores found.")
            return 0

        ensure_teams_from_matches(session, matches)

        # Initialize all teams to base rating before burn-in
        init_ratings(session)
        session.flush()

        count = 0
        for match in matches:
            home_score = int(match.home_score) if match.home_score is not None else 0
            away_score = int(match.away_score) if match.away_score is not None else 0
            result = _result_from_scores(home_score, away_score)
            update_ratings(session, match.home_team, match.away_team, result)
            count += 1

        session.commit()

        # Team Power Ranking: Top 10 by Elo
        ranking_stmt = (
            select(Team.name, Team.current_elo)
            .order_by(Team.current_elo.desc())
            .limit(10)
        )
        rows = session.execute(ranking_stmt).all()

        print("\n--- Team Power Ranking (Top 10) ---")
        for i, (name, elo) in enumerate(rows, 1):
            print(f"  {i:2}. {name}: {elo:.0f}")

        return count
