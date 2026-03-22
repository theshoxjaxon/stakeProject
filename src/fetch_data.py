"""Fetch upcoming matches and odds from The Odds API."""

from datetime import datetime

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import (
    DATABASE_PATH,
    ODDS_API_BASE_URL,
    ODDS_API_KEY,
    DEFAULT_SPORTS,
    SCORES_DAYS_FROM,
)
from src.database import get_engine, init_db
from src.models import Match, Odds, Team


def _american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal format."""
    if american_odds >= 100:
        return round(1 + (american_odds / 100), 4)
    return round(1 + (100 / abs(american_odds)), 4)


def _parse_h2h_outcomes(
    outcomes: list[dict], home_team: str, away_team: str
) -> tuple[float, float, float] | None:
    """
    Parse h2h market outcomes into (h_odds, d_odds, a_odds).
    Soccer h2h typically has 3 outcomes: home, draw, away.
    """
    h_odds = d_odds = a_odds = None
    for outcome in outcomes:
        name = outcome.get("name", "")
        price = outcome.get("price")
        if price is None:
            continue
        decimal_price = _american_to_decimal(float(price))
        if name == home_team:
            h_odds = decimal_price
        elif name == away_team:
            a_odds = decimal_price
        elif name.lower() == "draw":
            d_odds = decimal_price
    if h_odds is not None and d_odds is not None and a_odds is not None:
        return (h_odds, d_odds, a_odds)
    return None


def fetch_odds_for_sport(
    sport_key: str, api_key: str, regions: str = "uk,eu"
) -> list[dict]:
    """
    Fetch upcoming events and odds from The Odds API for a given sport.

    The v4 ``/odds`` endpoint returns only not-yet-started fixtures (future kickoffs).
    Uses uk,eu regions for soccer 1X2 (home/draw/away) markets.
    """
    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": "h2h",
        "oddsFormat": "american",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def upsert_match(session: Session, event: dict, sport_key: str) -> Match | None:
    """
    Insert or update a match. Returns the Match if it has valid 1X2 odds, else None.
    """
    match_id = event.get("id")
    if not match_id:
        return None

    commence_time_str = event.get("commence_time", "")
    try:
        commence_time = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None

    home_team = event.get("home_team") or ""
    away_team = event.get("away_team") or ""
    if not home_team or not away_team:
        return None

    existing = session.get(Match, match_id)
    if existing:
        match = existing
    else:
        match = Match(
            id=match_id,
            date=commence_time,
            home_team=home_team,
            away_team=away_team,
            home_score=None,
            away_score=None,
            status="scheduled",
        )
        session.add(match)
        session.flush()

    return match


def upsert_odds(session: Session, match: Match, event: dict) -> int:
    """
    Insert odds for each bookmaker. Skips if we already have recent odds for this match+bookmaker.
    Returns count of new odds rows inserted.
    """
    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")
    inserted = 0

    for bookmaker in event.get("bookmakers", []):
        bookmaker_key = bookmaker.get("key", "unknown")
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = market.get("outcomes", [])
            parsed = _parse_h2h_outcomes(outcomes, home_team, away_team)
            if parsed is None:
                continue

            h_odds, d_odds, a_odds = parsed

            # Check for existing odds (avoid duplicates)
            existing = session.execute(
                select(Odds).where(
                    Odds.match_id == match.id,
                    Odds.bookmaker == bookmaker_key,
                )
            ).scalar_one_or_none()

            if existing is None:
                session.add(
                    Odds(
                        match_id=match.id,
                        bookmaker=bookmaker_key,
                        h_odds=h_odds,
                        d_odds=d_odds,
                        a_odds=a_odds,
                    )
                )
                inserted += 1

    return inserted


def ensure_teams(session: Session, team_names: set[str]) -> None:
    """Ensure all team names exist in the teams table."""
    for name in team_names:
        existing = session.execute(
            select(Team).where(Team.name == name)
        ).scalar_one_or_none()
        if existing is None:
            session.add(Team(name=name, current_elo=1500.0))


def fetch_scores_for_sport(
    sport_key: str, api_key: str, days_from: int = SCORES_DAYS_FROM
) -> list[dict]:
    """
    Fetch scores for a sport from The Odds API scores endpoint.
    days_from: 1-3 (API max). Returns live + recently completed games with scores.
    """
    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/scores"
    params = {
        "apiKey": api_key,
        "daysFrom": days_from,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_historical_scores() -> dict[str, int]:
    """
    Fetch historical scores for DEFAULT_SPORTS (last SCORES_DAYS_FROM days).
    Updates matches table with home_score, away_score, status='completed'.
    Returns {"matches_updated": N, "sports_processed": N}.
    """
    if not ODDS_API_KEY:
        raise ValueError(
            "ODDS_API_KEY not set. Add it to a .env file or set the environment variable."
        )

    init_db(DATABASE_PATH)
    engine = get_engine(DATABASE_PATH)
    matches_updated = 0

    with Session(engine) as session:
        team_names: set[str] = set()

        for sport_key in DEFAULT_SPORTS:
            try:
                events = fetch_scores_for_sport(sport_key, ODDS_API_KEY)
            except requests.RequestException as e:
                raise RuntimeError(
                    f"Failed to fetch scores for {sport_key}: {e}"
                ) from e

            for event in events:
                if not event.get("completed") or not event.get("scores"):
                    continue

                match_id = event.get("id")
                if not match_id:
                    continue

                home_team = event.get("home_team") or ""
                away_team = event.get("away_team") or ""
                if not home_team or not away_team:
                    continue

                home_score = away_score = None
                for s in event["scores"]:
                    name = s.get("name")
                    raw = s.get("score")
                    if name is None or raw is None:
                        continue
                    try:
                        score = int(raw)
                    except (ValueError, TypeError):
                        continue
                    if name == home_team:
                        home_score = score
                    elif name == away_team:
                        away_score = score

                if home_score is None or away_score is None:
                    continue

                team_names.add(home_team)
                team_names.add(away_team)

                match = session.get(Match, match_id)
                if match:
                    match.home_score = home_score
                    match.away_score = away_score
                    match.status = "completed"
                    matches_updated += 1
                else:
                    commence_time_str = event.get("commence_time", "")
                    try:
                        commence_time = datetime.fromisoformat(
                            commence_time_str.replace("Z", "+00:00")
                        )
                    except (ValueError, TypeError):
                        continue
                    session.add(
                        Match(
                            id=match_id,
                            date=commence_time,
                            home_team=home_team,
                            away_team=away_team,
                            home_score=home_score,
                            away_score=away_score,
                            status="completed",
                        )
                    )
                    matches_updated += 1

        team_names.discard("")
        ensure_teams(session, team_names)
        session.commit()

    return {
        "matches_updated": matches_updated,
        "sports_processed": len(DEFAULT_SPORTS),
    }


def run_update_cycle() -> dict[str, int]:
    """
    Run a single update cycle: fetch odds for default sports and persist to SQLite.
    Handles duplicates by checking match_id before inserting matches.
    Returns summary: {"matches_added": N, "odds_added": N, "sports_processed": N}
    """
    if not ODDS_API_KEY:
        raise ValueError(
            "ODDS_API_KEY not set. Add it to a .env file or set the environment variable. "
            "Get your key at https://the-odds-api.com/"
        )

    init_db(DATABASE_PATH)
    engine = get_engine(DATABASE_PATH)
    matches_processed = 0
    odds_added = 0
    team_names: set[str] = set()

    with Session(engine) as session:
        for sport_key in DEFAULT_SPORTS:
            try:
                events = fetch_odds_for_sport(sport_key, ODDS_API_KEY)
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to fetch {sport_key}: {e}") from e

            for event in events:
                match = upsert_match(session, event, sport_key)
                if match is None:
                    continue
                team_names.add(event.get("home_team", ""))
                team_names.add(event.get("away_team", ""))
                inserted = upsert_odds(session, match, event)
                odds_added += inserted
                matches_processed += 1

        team_names.discard("")
        ensure_teams(session, team_names)
        session.commit()

    return {
        "matches_processed": matches_processed,
        "odds_added": odds_added,
        "sports_processed": len(DEFAULT_SPORTS),
    }
