"""Configuration for the betting value detection system."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "betting.db"

# API configuration
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# Default sports to fetch (soccer leagues - have home/draw/away markets)
DEFAULT_SPORTS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
]

# Scores API: daysFrom 1-3 (API max is 3 days of historical results)
SCORES_DAYS_FROM = 3
