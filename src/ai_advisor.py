"""AI Advisor for complex betting markets and risk management.

Uses Google Gemini to sanity-check quant value bets. The advisor is optional:
when GEMINI_API_KEY is not set (or AI_ADVISOR_ENABLED=false) the pipeline
runs pure-quant and `get_ai_betting_advice` returns a "Skip" verdict, so
callers never need to special-case a missing key.
"""

from __future__ import annotations

import json
import os

import numpy as np
from dotenv import load_dotenv

from src.config import EDGE_THRESHOLD

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview").strip()

_client = None
_client_failed = False


def advisor_enabled() -> bool:
    """True when the advisor is switched on and a Gemini key is present."""
    raw = os.getenv("AI_ADVISOR_ENABLED", "true").strip().lower()
    enabled = raw in ("1", "true", "yes", "on")
    return enabled and bool(os.getenv("GEMINI_API_KEY", "").strip())


def _get_client():
    """Create the Gemini client on first use; never crash at import time."""
    global _client, _client_failed
    if _client is not None or _client_failed:
        return _client
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        _client_failed = True
        return None
    try:
        from google import genai

        _client = genai.Client(api_key=api_key)
    except Exception:
        _client_failed = True
        _client = None
    return _client


def analyze_advanced_markets(matrix) -> dict[str, float]:
    """
    Slice the score-probability matrix into derived market probabilities.

    - BTTS: both teams score >= 1 (drop row 0 and column 0).
    - Away_Win: away goals > home goals (upper triangle).
    - BTTS_Away_Win: away wins with both teams scoring.
    """
    matrix = np.array(matrix)
    if matrix.ndim != 2:
        return {"BTTS": 0.0, "Away_Win": 0.0, "BTTS_Away_Win": 0.0}

    prob_btts = np.sum(matrix[1:, 1:])
    prob_away_win = np.sum(np.triu(matrix, k=1))
    prob_btts_away = np.sum(np.triu(matrix[1:, :], k=1))

    return {
        "BTTS": round(float(prob_btts) * 100, 2),
        "Away_Win": round(float(prob_away_win) * 100, 2),
        "BTTS_Away_Win": round(float(prob_btts_away) * 100, 2),
    }


def _skip_response(reason: str) -> str:
    return json.dumps(
        {"final_command": "Skip", "reasoning": reason, "recommended_side": None}
    )


def get_ai_betting_advice(match_data: dict, matrix: np.ndarray) -> str:
    """
    Ask Gemini for a final risk verdict on a candidate bet.

    Returns a JSON string with keys: final_command ('Bet' | 'Skip' | 'Error'),
    reasoning, recommended_side ('H' | 'D' | 'A' | null).
    """
    client = _get_client()
    if client is None:
        return _skip_response("AI advisor disabled (no GEMINI_API_KEY).")

    advanced = analyze_advanced_markets(matrix)

    prompt = f"""
    ROLE: Professional Sports Betting Risk Manager.

    You are analyzing a football match using a statistical model.
    The model has generated a **Poisson Matrix** of score probabilities.
    Based on this matrix and market odds, provide a **Final Decision**.

    MATCH: {match_data['home_team']} vs {match_data['away_team']}

    MODEL PREDICTIONS (from Poisson Matrix):
    - Home Win Probability: {match_data['model_prob_h']*100:.1f}%
    - Draw Probability: {match_data['model_prob_d']*100:.1f}%
    - Away Win Probability: {match_data['model_prob_a']*100:.1f}%
    - Both Teams To Score (BTTS) Probability: {advanced['BTTS']}%

    BOOKMAKER DATA:
    - Market Odds for Home Win: {match_data['market_odds_h']}
    - Market Odds for Draw: {match_data['market_odds_d']}
    - Market Odds for Away Win: {match_data['market_odds_a']}

    ANALYSIS TASK:
    1. Compare model probabilities to market odds to find value
       (edge > {EDGE_THRESHOLD*100:.1f}%).
    2. If value is found, recommend a bet.
    3. If no value is found, recommend skipping.
    4. Provide your **Final Decision** as a valid JSON object with keys
       'final_command' ('Bet' or 'Skip'), 'reasoning', and
       'recommended_side' ('H', 'D', 'A', or null).

    Example for value:
    {{"final_command": "Bet", "reasoning": "Model shows edge on Home.",
      "recommended_side": "H"}}
    Example for no value:
    {{"final_command": "Skip", "reasoning": "No significant value detected.",
      "recommended_side": null}}
    """

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        text = (response.text or "").strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
    except Exception as e:
        return json.dumps(
            {
                "final_command": "Error",
                "reasoning": f"AI Advice Error: {e}",
                "recommended_side": None,
            }
        )
