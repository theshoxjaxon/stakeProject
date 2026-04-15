"""
Production-grade data fetching module for advanced match statistics (e.g., xG)
from FBref, incorporating resilient web scraping and efficient database operations.
"""

import functools
import logging
import random
import time
from datetime import datetime, timedelta

import pandas as pd
import soccerdata as sd
from requests.exceptions import RequestException
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from src.database import get_engine
from src.models import Match, MatchAdvanced

# Use the project's logger
logger = logging.getLogger(__name__)

# A list of common User-Agents to rotate through for scraping
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:108.0) Gecko/20100101 Firefox/108.0"
]


def retry_with_backoff(retries: int = 3, delay: int = 5, backoff: int = 2):
    """A decorator for retrying a function with exponential backoff."""
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            mtries, mdelay = retries, delay
            while mtries > 1:
                try:
                    return func(*args, **kwargs)
                except RequestException as e:
                    logger.warning(
                        f"Request failed: {e}. Retrying in {mdelay} seconds..."
                    )
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return func(*args, **kwargs)
        return wrapper_retry
    return decorator_retry


def find_match_in_db(
    session: Session, game_date: pd.Timestamp, home_team: str, away_team: str
) -> Match | None:
    """Find a match in the DB based on teams and a +/- 12-hour window."""
    home_team_norm = home_team.lower().replace(" ", "")
    away_team_norm = away_team.lower().replace(" ", "")

    time_window_start = game_date.to_pydatetime() - timedelta(hours=12)
    time_window_end = game_date.to_pydatetime() + timedelta(hours=12)

    stmt = select(Match).where(Match.date.between(time_window_start, time_window_end))

    possible_matches = session.execute(stmt).scalars().all()

    for match in possible_matches:
        db_home_norm = match.home_team.lower().replace(" ", "")
        db_away_norm = match.away_team.lower().replace(" ", "")
        if db_home_norm == home_team_norm and db_away_norm == away_team_norm:
            return match
    return None


@retry_with_backoff(retries=3, delay=10, backoff=2)
def _fetch_fbref_data(leagues: list[str], seasons: list[int]) -> pd.DataFrame:
    """
    Internal function to fetch schedule and team stats from FBref, then merge them.
    Includes User-Agent rotation to avoid being blocked.
    """
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    logger.info(
        f"Fetching FBref data for leagues: {leagues}, seasons: {seasons} with User-Agent: {headers['User-Agent']}"
    )
    
    fbref = sd.FBref(leagues=leagues, seasons=seasons, headers=headers)
    
    schedule = fbref.read_schedule().reset_index()
    
    # Add a delay before the next heavy request
    time.sleep(random.uniform(3, 7))

    shooting_stats = fbref.read_team_match_stats(stat_type="shooting").reset_index()

    if schedule.empty or shooting_stats.empty:
        logger.warning("Could not retrieve schedule or shooting stats from FBref.")
        return pd.DataFrame()

    home_merged = pd.merge(
        schedule,
        shooting_stats[['game', 'team', 'xg']],
        left_on=['game', 'home_team'],
        right_on=['game', 'team'],
        how='left'
    ).rename(columns={'xg': 'home_xg'})

    away_merged = pd.merge(
        schedule,
        shooting_stats[['game', 'team', 'xg']],
        left_on=['game', 'away_team'],
        right_on=['game', 'team'],
        how='left'
    ).rename(columns={'xg': 'away_xg'})

    final_df = pd.merge(
        home_merged[['game', 'date', 'home_team', 'away_team', 'home_xg']],
        away_merged[['game', 'away_xg']],
        on='game',
        how='inner'
    )
    
    logger.info(f"Successfully fetched and merged {len(final_df)} matches from FBref.")
    return final_df


def sync_league_xg(leagues: list[str], seasons: list[int]) -> dict[str, int]:
    """
    Fetches match data from FBref and stores xG data in the MatchAdvanced table.
    """
    logger.info(f"Starting xG sync for leagues: {leagues}, seasons: {seasons}")
    
    # Introduce a randomized delay to mimic human behavior
    sleep_time = random.uniform(5, 12)
    logger.info(f"Waiting for {sleep_time:.2f} seconds before starting scrape.")
    time.sleep(sleep_time)

    try:
        match_data = _fetch_fbref_data(leagues=leagues, seasons=seasons)
    except Exception as e:
        logger.error(
            f"Failed to fetch data from FBref after multiple retries: {e}",
            exc_info=True,
        )
        return {"upserted": 0, "not_found": 0, "skipped": 0, "total": 0}

    if match_data.empty:
        logger.warning("FBref returned no match data for the given selection.")
        return {"upserted": 0, "not_found": 0, "skipped": 0, "total": 0}

    engine = get_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    upserted_count, not_found_count, skipped_count = 0, 0, 0

    with SessionLocal() as session:
        for _, row in match_data.iterrows():
            home_xg = row.get('home_xg')
            away_xg = row.get('away_xg')

            if pd.isna(home_xg) or pd.isna(away_xg):
                skipped_count += 1
                continue

            match = find_match_in_db(
                session, row['date'], row['home_team'], row['away_team']
            )

            if not match:
                not_found_count += 1
                logger.debug(
                    f"Match not found in DB: {row['date'].date()} | {row['home_team']} vs {row['away_team']}"
                )
                continue

            try:
                session.merge(
                    MatchAdvanced(
                        match_id=match.id,
                        home_xg=float(home_xg),
                        away_xg=float(away_xg),
                        source="fbref",
                    )
                )
                upserted_count += 1
            except Exception as e:
                logger.error(
                    f"Failed to upsert advanced stats for match {match.id}: {e}"
                )
                session.rollback()

        if upserted_count > 0:
            session.commit()

    summary = {
        "upserted": upserted_count,
        "not_found": not_found_count,
        "skipped": skipped_count,
        "total": len(match_data),
    }
    logger.info(f"Finished xG sync. Summary: {summary}")
    return summary
