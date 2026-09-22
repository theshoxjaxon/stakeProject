"""Tier-gated prediction serialization — the ONLY place plan logic lives.

serialize_prediction(prediction, plan) is a pure function: every plan check
for what a caller sees lives here, never scattered through the routers.
Callers just call ``serialize_prediction(p, plan)`` and ship the result.

Tiers are additive — pro is free plus more, premium is pro plus more:

- free:    fixture, kickoff, 1X2 probabilities — but only 24h after kickoff
           ("delayed 24h": before that, probabilities are withheld entirely).
- pro:     + fair%, edge%, kelly stake% — "live": no delay, ever.
- premium: + model_version, feature breakdown, full settlement history.

Requires ``prediction.match`` to already be loaded (e.g. via
``selectinload(Prediction.match)`` in the caller's query) — this function
never touches a session or does its own lazy loading, by design.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

# Single source of truth for the model identifier — the legacy /predict
# endpoint imports this too, rather than hardcoding its own copy.
MODEL_VERSION = "dixon-coles-v1"

_FREE_TIER_DELAY = timedelta(hours=24)


def _as_utc(dt: datetime) -> datetime:
    """Normalize to aware UTC (naive DateTime columns are stored as UTC)."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _probabilities_visible(prediction: Any, plan: str) -> bool:
    """
    free: only once the fixture is >=24h in the past ("delayed 24h" — a
    historical/research feed, not a live one). pro/premium: always ("live").

    Keyed off kickoff, not created_at: predictions are updated in place
    (main.py upserts the same row), so created_at reflects when the fixture
    first entered the system, not when these probabilities were computed —
    it can't honestly answer "how fresh is this number." Kickoff can.
    """
    if plan != "free":
        return True
    if prediction.kickoff is None:
        return False
    return datetime.now(timezone.utc) - _as_utc(prediction.kickoff) >= _FREE_TIER_DELAY


def serialize_prediction(prediction: Any, plan: str) -> dict:
    """Serialize one prediction for a caller on `plan` ('free' | 'pro' | 'premium')."""
    probabilities_visible = _probabilities_visible(prediction, plan)

    out: dict = {
        "id": prediction.id,
        "match_id": prediction.match_id,
        "fixture": {
            "home_team": prediction.match.home_team,
            "away_team": prediction.match.away_team,
        },
        "kickoff": prediction.kickoff,
        "league": prediction.league,
        "probabilities": (
            {
                "home": prediction.home_prob,
                "draw": prediction.draw_prob,
                "away": prediction.away_prob,
            }
            if probabilities_visible
            else None
        ),
        "delayed": not probabilities_visible,
    }

    if plan == "free":
        return out

    # pro and premium both get this.
    out["fair_probabilities"] = {
        "home": 1.0 / prediction.fair_home if prediction.fair_home else None,
        "draw": 1.0 / prediction.fair_draw if prediction.fair_draw else None,
        "away": 1.0 / prediction.fair_away if prediction.fair_away else None,
    }
    out["edge_percent"] = {
        "home": (
            round(prediction.edge_home * 100, 4)
            if prediction.edge_home is not None
            else None
        ),
        "draw": (
            round(prediction.edge_draw * 100, 4)
            if prediction.edge_draw is not None
            else None
        ),
        "away": (
            round(prediction.edge_away * 100, 4)
            if prediction.edge_away is not None
            else None
        ),
    }
    out["recommended_selection"] = prediction.recommended_selection
    out["kelly_stake_percent"] = prediction.recommended_stake_percent

    if plan == "pro":
        return out

    # premium only.
    out["model_version"] = MODEL_VERSION
    # Per-prediction feature multipliers (form/H2H/rest/xG/injury) aren't
    # persisted anywhere yet — main.py computes them transiently and
    # discards them. Recomputing them here would need a DB session, which
    # this function deliberately never takes. Real values need a schema
    # change (store them at prediction time); until then this is honestly
    # null rather than fabricated.
    out["feature_breakdown"] = None
    out["settlement"] = {
        "result_settled": prediction.result_settled,
        "actual_outcome": prediction.actual_outcome,
        "was_win": prediction.was_win,
        "profit": prediction.profit,
        "settled_at": prediction.settled_at,
    }
    return out
