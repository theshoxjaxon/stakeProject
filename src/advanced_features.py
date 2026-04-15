"""
Advanced feature engineering using xG and other advanced stats.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from sqlalchemy import func

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from src.config import XG_WEIGHT, XG_WINDOW, INJURY_WEIGHT
from src.models import Match, MatchAdvanced, PlayerInjury, TeamStats


def utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _normalize_before(before: datetime | None) -> datetime:
    if before is None:
        return utc_now_naive()
    if before.tzinfo is not None:
        return before.astimezone(timezone.utc).replace(tzinfo=None)
    return before


@dataclass
class RollingStats:
    """Rolling stats over a window of completed matches."""

    team: str
    games: int
    avg_xg_for: float
    avg_xg_against: float
    avg_goals_for: float
    avg_goals_against: float


def get_rolling_stats(
    session: Session,
    team_name: str,
    window: int = 5,
    before: datetime | None = None,
) -> RollingStats:
    """
    Calculate rolling average xG and goals for/against for a given team.
    """
    before = _normalize_before(before)

    past_matches_stmt = (
        select(Match.id)
        .where(
            and_(
                Match.date < before,
                Match.status.in_(["completed", "finished"]),
                or_(Match.home_team == team_name, Match.away_team == team_name),
            )
        )
        .order_by(Match.date.desc())
        .limit(window)
    ).subquery()

    stmt = (
        select(Match, MatchAdvanced)
        .join(MatchAdvanced, Match.id == MatchAdvanced.match_id)
        .where(Match.id.in_(select(past_matches_stmt)))
    )

    results = session.execute(stmt).all()

    if not results:
        return RollingStats(team=team_name, games=0, avg_xg_for=0.0, avg_xg_against=0.0, avg_goals_for=0.0, avg_goals_against=0.0)

    total_xg_for = 0.0
    total_xg_against = 0.0
    total_goals_for = 0.0
    total_goals_against = 0.0
    game_count = len(results)
    valid_games = 0

    for match, advanced_stats in results:
        if advanced_stats.home_xg is None or advanced_stats.away_xg is None or match.home_score is None or match.away_score is None:
            continue
        
        valid_games += 1
        if match.home_team == team_name:
            total_xg_for += advanced_stats.home_xg
            total_xg_against += advanced_stats.away_xg
            total_goals_for += match.home_score
            total_goals_against += match.away_score
        else:
            total_xg_for += advanced_stats.away_xg
            total_xg_against += advanced_stats.home_xg
            total_goals_for += match.away_score
            total_goals_against += match.home_score
            
    if valid_games == 0:
        return RollingStats(team=team_name, games=0, avg_xg_for=0.0, avg_xg_against=0.0, avg_goals_for=0.0, avg_goals_against=0.0)

    return RollingStats(
        team=team_name,
        games=valid_games,
        avg_xg_for=total_xg_for / valid_games,
        avg_xg_against=total_xg_against / valid_games,
        avg_goals_for=total_goals_for / valid_games,
        avg_goals_against=total_goals_against / valid_games,
    )


def get_xg_multiplier(team_name: str, db_session: Session) -> float:
    """
    Calculates a multiplier based on the difference between a team's
    average goals scored and their average expected goals (xG).
    This is based on the idea of regression to the mean.
    """
    rolling_stats = get_rolling_stats(db_session, team_name, window=XG_WINDOW)
    if rolling_stats.games == 0:
        return 1.0
    
    # Positive if underperforming xG, negative if overperforming
    xg_performance = rolling_stats.avg_xg_for - rolling_stats.avg_goals_for
    
    multiplier = 1.0 + (xg_performance * XG_WEIGHT)
    
    # Clamp the multiplier to a reasonable range to avoid extreme effects
    return max(0.8, min(1.2, multiplier))


def get_injury_penalty(team_name: str, db_session: Session) -> float:
    """
    Calculates a penalty multiplier based on the number of key players
    who are injured or suspended for a team.
    """
    unavailable_players_count = db_session.execute(
        select(func.count(PlayerInjury.id)).where(
            PlayerInjury.team_name == team_name,
            PlayerInjury.status.in_(['out', 'suspended'])
        )
    ).scalar_one_or_none()

    if unavailable_players_count is None or unavailable_players_count == 0:
        return 1.0

    # Apply a compounding penalty for each unavailable player
    penalty = INJURY_WEIGHT ** unavailable_players_count
    
    return penalty
