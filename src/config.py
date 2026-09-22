"""Configuration for the betting value detection system — load from environment / .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def _float(name: str, default: str) -> float:
    return float(os.getenv(name, default).strip())


def _int(name: str, default: str) -> int:
    return int(os.getenv(name, default).strip())


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _prediction_horizon_days() -> int | None:
    raw = os.getenv("PREDICTION_HORIZON_DAYS", "").strip()
    if not raw:
        return None
    return max(1, int(raw))


_DEFAULT_DATABASE_URL = (
    "postgresql+psycopg2://postgres:postgres@localhost:5432/stakeproject"
)


def _database_url() -> str:
    raw = os.getenv("DATABASE_URL", "").strip() or _DEFAULT_DATABASE_URL
    from sqlalchemy.engine.url import make_url

    u = make_url(raw)
    if u.get_backend_name() != "postgresql":
        raise ValueError("Only postgresql DATABASE_URL is supported in this project.")
    return raw


DATABASE_URL = _database_url()
PREDICTION_HORIZON_DAYS = _prediction_horizon_days()

# Redis (plan-check cache, daily rate limiting) — auxiliary, no fail-loud like JWT_SECRET.
REDIS_URL = os.getenv("REDIS_URL", "").strip() or "redis://localhost:6379/0"

# API (never commit real keys — use .env)
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
ODDS_API_BASE_URL = os.getenv(
    "ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4"
).strip()

DEFAULT_SPORTS = [
    s.strip()
    for s in os.getenv(
        "DEFAULT_SPORTS",
        "soccer_epl,soccer_spain_la_liga,soccer_germany_bundesliga",
    ).split(",")
    if s.strip()
] or ["soccer_epl"]

# Tournament auto-detection: when a candidate competition is in season
# (World Cup, Euros, UCL, ...) its matches join the update cycle automatically.
TOURNAMENT_AUTO_DETECT = _bool("TOURNAMENT_AUTO_DETECT", True)
TOURNAMENT_SPORT_KEYS = [
    s.strip()
    for s in os.getenv(
        "TOURNAMENT_SPORT_KEYS",
        "soccer_fifa_world_cup,"
        "soccer_uefa_european_championship,"
        "soccer_uefa_champs_league,"
        "soccer_uefa_europa_league,"
        "soccer_conmebol_copa_america",
    ).split(",")
    if s.strip()
]
SPORTS_CACHE_TTL_HOURS = max(1, _int("SPORTS_CACHE_TTL_HOURS", "12"))

# xG scrape (FBref via soccerdata) is slow and easily rate-limited — opt in.
XG_SYNC_ENABLED = _bool("XG_SYNC_ENABLED", False)

# Reporting / BTTS
BTTS_VALUE_THRESHOLD = _float("BTTS_VALUE_THRESHOLD", "0.05")

SCORES_DAYS_FROM = max(1, min(3, _int("SCORES_DAYS_FROM", "3")))

# Value / staking
EDGE_THRESHOLD = _float("EDGE_THRESHOLD", "0.05")
KELLY_FRACTION = _float("KELLY_FRACTION", "0.25")
# If true, main.py prints only sides with edge > EDGE_THRESHOLD (legacy compact view)
SHOW_ONLY_VALUE_BETS = _bool("SHOW_ONLY_VALUE_BETS", False)
DEFAULT_BANKROLL = _float("DEFAULT_BANKROLL", "1000.0")

# Feature engineering (Poisson λ adjustments)
FORM_WINDOW = max(3, _int("FORM_WINDOW", "10"))
FEATURES_ENABLED = _bool("FEATURES_ENABLED", True)
FORM_WEIGHT = _float("FORM_WEIGHT", "0.08")
H2H_WEIGHT = _float("H2H_WEIGHT", "0.06")
REST_SHORT_DAYS = _int("REST_SHORT_DAYS", "3")
REST_FATIGUE_FACTOR = _float("REST_FATIGUE_FACTOR", "0.97")
MIDWEEK_FATIGUE_FACTOR = _float("MIDWEEK_FATIGUE_FACTOR", "0.99")
XG_WEIGHT = _float("XG_WEIGHT", "1.05")
INJURY_WEIGHT = _float("INJURY_WEIGHT", "0.95")
XG_WINDOW = _int("XG_WINDOW", "5")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()
LOG_DIR = Path(os.getenv("LOG_DIR", str(DATA_DIR / "logs")))
LOG_FILE = os.getenv("LOG_FILE", "betting_engine.log").strip()
LOG_MAX_BYTES = _int("LOG_MAX_BYTES", str(2 * 1024 * 1024))
LOG_BACKUP_COUNT = _int("LOG_BACKUP_COUNT", "3")
