"""Historical backfill for matches and Elo ratings.

NOTE: The Odds API scores endpoint only supports a small look-back window
via ``daysFrom`` (1–3). Fully backfilling 2023–2025 seasons for EPL, La Liga,
and Bundesliga would require an external results data source. This module
is structured to support that, and currently uses the scores endpoint to
prime the database with recent results before running Elo updates.
"""

from __future__ import annotations

from datetime import datetime

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import DEFAULT_SPORTS, ODDS_API_BASE_URL, ODDS_API_KEY, SCORES_DAYS_FROM
from src.database import Match, TeamRating, get_engine, init_db
from src.elo import EloEngine


def _fetch_scores_for_sport(
    sport_key: str, days_from: int = SCORES_DAYS_FROM
) -> list[dict]:
    """Fetch recent scores for a sport from The Odds API scores endpoint."""
    if not ODDS_API_KEY:
        raise ValueError("ODDS_API_KEY is not set in the environment/.env.")

    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/scores"
    resp = requests.get(
        url,
        params={"apiKey": ODDS_API_KEY, "daysFrom": days_from},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _upsert_match_from_event(session: Session, event: dict) -> bool:
    """Upsert a Match row from a scores event; return True if written."""
    if not event.get("completed") or not event.get("scores"):
        return False

    match_id = event.get("id")
    home_team = event.get("home_team") or ""
    away_team = event.get("away_team") or ""
    if not match_id or not home_team or not away_team:
        return False

    home_score = away_score = None
    for s in event["scores"]:
        name = s.get("name")
        raw = s.get("score")
        if name is None or raw is None:
            continue
        try:
            val = int(raw)
        except (TypeError, ValueError):
            continue
        if name == home_team:
            home_score = val
        elif name == away_team:
            away_score = val

    if home_score is None or away_score is None:
        return False

    match = session.get(Match, match_id)
    if not match:
        commence_str = event.get("commence_time", "")
        try:
            dt = datetime.fromisoformat(commence_str.replace("Z", "+00:00"))
        except Exception:
            dt = datetime.utcnow()
        match = Match(
            id=match_id,
            date=dt,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            status="completed",
        )
        session.add(match)
        return True

    match.home_score = home_score
    match.away_score = away_score
    match.status = "completed"
    return True


def _rebuild_elo_from_matches(session: Session) -> None:
    """Run EloEngine over all completed matches in chronological order and persist team_ratings."""
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
        return

    engine = EloEngine()
    for m in matches:
        engine.update_ratings(
            m.home_team,
            m.away_team,
            (int(m.home_score), int(m.away_score)),
        )

    # Persist final Elo ratings into team_ratings table
    for team_name, rating in engine.ratings.items():
        tr = (
            session.execute(select(TeamRating).where(TeamRating.team_name == team_name))
            .scalars()
            .first()
        )
        if tr is None:
            tr = TeamRating(team_name=team_name, elo_rating=rating)
            session.add(tr)
        else:
            tr.elo_rating = rating


def run_backfill() -> int:
    """
    Backfill historical results (within API limits) and rebuild Elo ratings.

    Returns
    -------
    int
        Number of matches updated or created.
    """
    init_db()
    engine = get_engine()

    updated = 0
    with Session(engine) as session:
        for sport in DEFAULT_SPORTS:
            events = _fetch_scores_for_sport(sport, days_from=SCORES_DAYS_FROM)
            for ev in events:
                if _upsert_match_from_event(session, ev):
                    updated += 1

        _rebuild_elo_from_matches(session)
        session.commit()

    return updated
