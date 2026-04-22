"""
Rolling form, head-to-head, rest, and home/away context from completed matches.

All kickoff times are compared in naive UTC where possible; ``before`` should match
how ``Match.date`` is stored (typically naive UTC from SQLite).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from src.advanced_features import get_injury_penalty, get_xg_multiplier
from src.config import (
    FORM_WEIGHT,
    H2H_WEIGHT,
    MIDWEEK_FATIGUE_FACTOR,
    REST_FATIGUE_FACTOR,
    REST_SHORT_DAYS,
)
from src.models import Match

logger = logging.getLogger(__name__)


def utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _normalize_before(before: datetime | None) -> datetime:
    if before is None:
        return utc_now_naive()
    if before.tzinfo is not None:
        return before.astimezone(timezone.utc).replace(tzinfo=None)
    return before


@dataclass
class TeamForm:
    """Last N completed games for one team before ``before``."""

    team: str
    games: int
    points: int
    goals_for: float
    goals_against: float
    goals_for_pg: float
    goals_against_pg: float
    points_per_game: float
    form_string: str  # e.g. "WWDLW" from most recent first


def _points_for_team(m: Match, team: str) -> int:
    hs = int(m.home_score or 0)
    as_ = int(m.away_score or 0)
    if m.home_team == team:
        if hs > as_:
            return 3
        if hs == as_:
            return 1
        return 0
    if m.away_team == team:
        if as_ > hs:
            return 3
        if as_ == hs:
            return 1
        return 0
    return 0


def _result_letter(m: Match, team: str) -> str:
    hs = int(m.home_score or 0)
    as_ = int(m.away_score or 0)
    if m.home_team == team:
        if hs > as_:
            return "W"
        if hs < as_:
            return "L"
        return "D"
    if m.away_team == team:
        if as_ > hs:
            return "W"
        if as_ < hs:
            return "L"
        return "D"
    return "?"


def _gf_ga(m: Match, team: str) -> tuple[float, float]:
    hs = float(m.home_score or 0)
    as_ = float(m.away_score or 0)
    if m.home_team == team:
        return hs, as_
    if m.away_team == team:
        return as_, hs
    return 0.0, 0.0


def compute_team_form(
    session: Session,
    team: str,
    window: int,
    before: datetime | None = None,
) -> TeamForm:
    """Rolling form over last ``window`` completed games for ``team``."""
    before = _normalize_before(before)
    stmt = (
        select(Match)
        .where(
            Match.status.in_(["completed", "finished"]),
            Match.home_score.isnot(None),
            Match.date < before,
            (Match.home_team == team) | (Match.away_team == team),
        )
        .order_by(Match.date.desc())
        .limit(window)
    )
    rows = list(session.execute(stmt).scalars().all())
    if not rows:
        return TeamForm(team, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, "")

    pts = 0
    gf = 0.0
    ga = 0.0
    letters: list[str] = []
    for m in rows:
        pts += _points_for_team(m, team)
        g, g_ = _gf_ga(m, team)
        gf += g
        ga += g_
        letters.append(_result_letter(m, team))

    n = len(rows)
    return TeamForm(
        team=team,
        games=n,
        points=pts,
        goals_for=gf,
        goals_against=ga,
        goals_for_pg=gf / n,
        goals_against_pg=ga / n,
        points_per_game=pts / n,
        form_string="".join(letters),
    )


@dataclass
class H2HSummary:
    meetings: int
    home_wins: int  # wins for current fixture home side
    draws: int
    away_wins: int
    avg_total_goals: float


def compute_h2h(
    session: Session,
    home_team: str,
    away_team: str,
    limit: int = 5,
    before: datetime | None = None,
) -> H2HSummary:
    """Last ``limit`` completed meetings between the two clubs (any venue)."""
    before = _normalize_before(before)
    stmt = (
        select(Match)
        .where(
            Match.status.in_(["completed", "finished"]),
            Match.home_score.isnot(None),
            Match.date < before,
            or_(
                (Match.home_team == home_team) & (Match.away_team == away_team),
                (Match.home_team == away_team) & (Match.away_team == home_team),
            ),
        )
        .order_by(Match.date.desc())
        .limit(limit)
    )
    rows = list(session.execute(stmt).scalars().all())
    if not rows:
        return H2HSummary(0, 0, 0, 0, 0.0)

    hw = dr = aw = 0
    goals = 0.0
    for m in rows:
        hs = int(m.home_score or 0)
        as_ = int(m.away_score or 0)
        goals += hs + as_
        if hs == as_:
            dr += 1
            continue
        if hs > as_:
            winner = m.home_team
        else:
            winner = m.away_team
        if winner == home_team:
            hw += 1
        else:
            aw += 1

    n = len(rows)
    return H2HSummary(
        meetings=n,
        home_wins=hw,
        draws=dr,
        away_wins=aw,
        avg_total_goals=goals / n,
    )


def days_since_last_match(
    session: Session, team: str, before: datetime | None = None
) -> int | None:
    """Calendar days since team's last completed match before ``before``."""
    before = _normalize_before(before)
    stmt = (
        select(Match)
        .where(
            Match.status.in_(["completed", "finished"]),
            Match.home_score.isnot(None),
            Match.date < before,
            (Match.home_team == team) | (Match.away_team == team),
        )
        .order_by(Match.date.desc())
        .limit(1)
    )
    m = session.execute(stmt).scalar_one_or_none()
    if m is None or m.date is None:
        return None
    # before is already naive UTC; normalize last match the same way (timedelta has no tzinfo)
    last_naive = _normalize_before(m.date)
    delta = before - last_naive
    return max(0, delta.days)


def is_midweek_kickoff(kickoff: datetime) -> bool:
    """Tue–Thu (UTC) — proxy for congested fixture lists / European midweek games."""
    if kickoff.tzinfo is not None:
        dt = kickoff.astimezone(timezone.utc)
    else:
        dt = kickoff.replace(tzinfo=timezone.utc)
    return dt.weekday() in (1, 2, 3)


@dataclass
class LambdaMultipliers:
    home: float
    away: float


def compute_lambda_multipliers(
    session: Session,
    home_team: str,
    away_team: str,
    kickoff: datetime,
    form_window: int,
) -> LambdaMultipliers:
    """
    Combine form, H2H, rest, and midweek into multipliers applied to Poisson λ_home / λ_away.

    Neutral baseline is (1.0, 1.0). Small deviations (~few %) keep the model stable.
    """
    before = _normalize_before(kickoff)

    fh = compute_team_form(session, home_team, form_window, before)
    fa = compute_team_form(session, away_team, form_window, before)
    h2h = compute_h2h(session, home_team, away_team, limit=5, before=before)

    # Form: map PPG (0–3) around league-ish mean ~1.5
    def form_mult(form: TeamForm) -> float:
        if form.games == 0:
            return 1.0
        ppg = form.points_per_game
        return 1.0 + FORM_WEIGHT * ((ppg - 1.5) / 1.5)

    mh = form_mult(fh)
    ma = form_mult(fa)

    # H2H: tilt toward side that historically dominated in this pairing
    if h2h.meetings > 0:
        denom = h2h.home_wins + h2h.draws + h2h.away_wins
        if denom > 0:
            diff = (h2h.home_wins - h2h.away_wins) / denom
            mh *= 1.0 + H2H_WEIGHT * diff
            ma *= 1.0 - H2H_WEIGHT * diff
            g = (mh * ma) ** 0.5
            if g > 0:
                mh /= g
                ma /= g

    def rest_mult(team: str) -> float:
        d = days_since_last_match(session, team, before)
        if d is None:
            return 1.0
        if d < REST_SHORT_DAYS:
            return REST_FATIGUE_FACTOR
        return 1.0

    mh *= rest_mult(home_team)
    ma *= rest_mult(away_team)

    if is_midweek_kickoff(kickoff):
        mh *= MIDWEEK_FATIGUE_FACTOR
        ma *= MIDWEEK_FATIGUE_FACTOR

    # xG regression-to-mean: teams over/under-performing their xG are nudged back.
    try:
        mh *= get_xg_multiplier(home_team, session)
        ma *= get_xg_multiplier(away_team, session)
    except Exception as exc:
        logger.debug("xG multiplier unavailable (%s) — skipping for this fixture.", exc)

    # Injury/suspension availability penalty: compounds per unavailable key player.
    try:
        mh *= get_injury_penalty(home_team, session)
        ma *= get_injury_penalty(away_team, session)
    except Exception as exc:
        logger.debug("Injury penalty unavailable (%s) — skipping for this fixture.", exc)

    return LambdaMultipliers(
        home=max(0.75, min(1.25, mh)), away=max(0.75, min(1.25, ma))
    )
