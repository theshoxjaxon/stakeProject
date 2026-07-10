"""Elo rating system for match outcome prediction."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models import Team, TeamRating

HOME_ADVANTAGE = 50
K_FACTOR = 20
BASE_RATING = 1500
LEAGUE_AVG_GOALS = 1.35


def expected_score(r_a: float, r_b: float) -> float:
    """
    Elo expected score for team A vs team B.
    E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    """
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400))


def init_ratings(session: Session) -> int:
    """Initialize all teams with base rating 1500. Returns count updated."""
    teams = session.execute(select(Team)).scalars().all()
    for team in teams:
        team.current_elo = BASE_RATING
    return len(teams)


def _rating_for(session: Session, team_name: str) -> float:
    """
    Elo for one team: prefer team_ratings (written by the Elo backfill),
    fall back to the legacy teams.current_elo column, else the base rating.
    """
    rating = session.execute(
        select(TeamRating.elo_rating).where(TeamRating.team_name == team_name)
    ).scalar_one_or_none()
    if rating is not None:
        return float(rating)
    legacy = session.execute(
        select(Team.current_elo).where(Team.name == team_name)
    ).scalar_one_or_none()
    return float(legacy) if legacy is not None else float(BASE_RATING)


def get_elo_ratings(
    session: Session, home_team: str, away_team: str
) -> tuple[float, float]:
    """
    Get Elo ratings from DB. Returns (r_home, r_away) raw ratings.
    Use elo_to_xg or expected_score with HOME_ADVANTAGE for calculations.
    """
    return (_rating_for(session, home_team), _rating_for(session, away_team))


def update_ratings(
    session: Session,
    home_team: str,
    away_team: str,
    result: str,
) -> None:
    """
    Update Elo ratings after a match.
    result: "H" (home win), "D" (draw), "A" (away win).
    K=20.
    """
    home = session.execute(
        select(Team).where(Team.name == home_team)
    ).scalar_one_or_none()
    away = session.execute(
        select(Team).where(Team.name == away_team)
    ).scalar_one_or_none()
    if not home or not away:
        return

    r_home_adj = home.current_elo + HOME_ADVANTAGE
    r_away = away.current_elo

    e_home = expected_score(r_home_adj, r_away)
    e_away = expected_score(r_away, r_home_adj)

    if result == "H":
        s_home, s_away = 1.0, 0.0
    elif result == "A":
        s_home, s_away = 0.0, 1.0
    else:  # "D"
        s_home, s_away = 0.5, 0.5

    home.current_elo += K_FACTOR * (s_home - e_home)
    away.current_elo += K_FACTOR * (s_away - e_away)


def elo_to_xg(
    r_home: float,
    r_away: float,
    league_avg: float = LEAGUE_AVG_GOALS,
) -> tuple[float, float]:
    """
    Convert Elo difference into Expected Goals (xG) for each team.
    Uses league average as baseline. Stronger team gets higher xG.
    """
    r_home_adj = r_home + HOME_ADVANTAGE
    e_home = expected_score(r_home_adj, r_away)
    e_away = 1.0 - e_home
    # Scale: when e_home=0.5, both get league_avg; when e_home>0.5, home gets more
    lambda_home = league_avg * 2.0 * e_home
    lambda_away = league_avg * 2.0 * e_away
    return (lambda_home, lambda_away)
