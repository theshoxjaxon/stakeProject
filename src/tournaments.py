"""Season-aware tournament detection via The Odds API ``/sports`` endpoint.

The Odds API marks every competition with ``active: true`` while it is in
season. We intersect that list with a curated set of candidate tournaments
(World Cup, Euros, Champions League, ...) so that e.g. during a World Cup
month the update cycle automatically pulls World Cup fixtures alongside the
regular domestic leagues — no hand-maintained date tables.

Requests to ``/sports`` are free (they do not count against the API quota),
but the result is still cached on disk for SPORTS_CACHE_TTL_HOURS to keep
pipeline start-up fast and to survive transient network failures.
"""

from __future__ import annotations

import json
import logging
import time

import requests

from src.config import (
    DATA_DIR,
    DEFAULT_SPORTS,
    ODDS_API_BASE_URL,
    ODDS_API_KEY,
    SPORTS_CACHE_TTL_HOURS,
    TOURNAMENT_AUTO_DETECT,
    TOURNAMENT_SPORT_KEYS,
)

logger = logging.getLogger(__name__)

_SPORTS_CACHE_FILE = DATA_DIR / "sports_cache.json"

# Friendly display names for console output and the API.
TOURNAMENT_TITLES = {
    "soccer_fifa_world_cup": "FIFA World Cup",
    "soccer_uefa_european_championship": "UEFA Euro",
    "soccer_uefa_champs_league": "UEFA Champions League",
    "soccer_uefa_europa_league": "UEFA Europa League",
    "soccer_conmebol_copa_america": "Copa América",
    "soccer_epl": "Premier League",
    "soccer_spain_la_liga": "La Liga",
    "soccer_germany_bundesliga": "Bundesliga",
    "soccer_italy_serie_a": "Serie A",
    "soccer_france_ligue_one": "Ligue 1",
}


def sport_title(sport_key: str | None) -> str:
    """Human-readable competition name for a sport key."""
    if not sport_key:
        return "Unknown competition"
    return TOURNAMENT_TITLES.get(sport_key, sport_key)


def _read_cache() -> list[dict] | None:
    """Return cached in-season sports if the cache file is fresh, else None."""
    try:
        raw = json.loads(_SPORTS_CACHE_FILE.read_text())
        age_hours = (time.time() - raw["fetched_at"]) / 3600
        if age_hours < SPORTS_CACHE_TTL_HOURS:
            return raw["sports"]
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return None


def _write_cache(sports: list[dict]) -> None:
    try:
        _SPORTS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SPORTS_CACHE_FILE.write_text(
            json.dumps({"fetched_at": time.time(), "sports": sports})
        )
    except OSError as exc:
        logger.warning("Could not write sports cache: %s", exc)


def _fetch_in_season_sports() -> list[dict]:
    """
    Fetch the list of in-season sports from The Odds API.

    Returns a list of {"key", "title", "active"} dicts. This endpoint is free:
    it does not count against the monthly request quota.
    """
    cached = _read_cache()
    if cached is not None:
        return cached

    url = f"{ODDS_API_BASE_URL}/sports"
    response = requests.get(url, params={"apiKey": ODDS_API_KEY}, timeout=30)
    response.raise_for_status()
    sports = [
        {"key": s.get("key"), "title": s.get("title"), "active": s.get("active")}
        for s in response.json()
    ]
    _write_cache(sports)
    return sports


def get_active_tournaments() -> list[dict]:
    """
    Candidate tournaments that are currently in season.

    Returns [{"key": ..., "title": ...}] — empty when detection is disabled,
    the API key is missing, or nothing is in season.
    """
    if not TOURNAMENT_AUTO_DETECT or not ODDS_API_KEY:
        return []

    try:
        sports = _fetch_in_season_sports()
    except requests.RequestException as exc:
        logger.warning("Tournament detection skipped — /sports fetch failed: %s", exc)
        return []

    active = {s["key"]: s for s in sports if s.get("active")}
    result = []
    for key in TOURNAMENT_SPORT_KEYS:
        if key in active:
            result.append(
                {"key": key, "title": active[key].get("title") or sport_title(key)}
            )
    return result


def resolve_sport_keys() -> list[str]:
    """
    Full list of sport keys for the update cycle: the configured default
    leagues plus any candidate tournament currently in season (deduplicated,
    order preserved).
    """
    keys = list(DEFAULT_SPORTS)
    tournaments = get_active_tournaments()
    for t in tournaments:
        if t["key"] not in keys:
            keys.append(t["key"])
    if tournaments:
        logger.info(
            "Active tournaments in season: %s",
            ", ".join(t["title"] for t in tournaments),
        )
    return keys
