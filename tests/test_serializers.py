"""Tests for src/api/serializers.py — the single source of tier-gated fields.

serialize_prediction(prediction, plan) is the ONLY place plan logic lives:
free/pro/premium each add fields on top of the last, and free additionally
withholds 1X2 probabilities until 24h after kickoff ("delayed 24h").
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.api.serializers import MODEL_VERSION, serialize_prediction


def _prediction(**overrides):
    """A fake Prediction-shaped object — serialize_prediction must be a pure
    function of its fields plus a pre-loaded .match, never touching a session."""
    now = datetime.now(timezone.utc)
    defaults = dict(
        id=1,
        match_id="match-1",
        kickoff=now + timedelta(hours=2),  # future by default: not yet delayed-eligible
        league="soccer_epl",
        home_prob=0.45,
        draw_prob=0.30,
        away_prob=0.25,
        fair_home=2.2,
        fair_draw=3.4,
        fair_away=4.0,
        edge_home=0.07,
        edge_draw=-0.02,
        edge_away=0.01,
        recommended_selection="home",
        recommended_stake_percent=3.5,
        result_settled=False,
        actual_outcome=None,
        was_win=None,
        profit=0.0,
        settled_at=None,
        match=SimpleNamespace(home_team="Arsenal", away_team="Chelsea"),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# --- free ---------------------------------------------------------------------


def test_free_tier_has_only_base_fields() -> None:
    out = serialize_prediction(_prediction(), plan="free")

    assert out["id"] == 1
    assert out["match_id"] == "match-1"
    assert out["fixture"] == {"home_team": "Arsenal", "away_team": "Chelsea"}
    assert out["league"] == "soccer_epl"
    assert "kickoff" in out
    assert "fair_probabilities" not in out
    assert "edge_percent" not in out
    assert "kelly_stake_percent" not in out
    assert "model_version" not in out
    assert "feature_breakdown" not in out
    assert "settlement" not in out


def test_free_tier_withholds_probabilities_before_24h_post_kickoff() -> None:
    """Kickoff was recent (<24h ago): free tier must not see 1X2 probabilities."""
    recent_kickoff = datetime.now(timezone.utc) - timedelta(hours=1)
    out = serialize_prediction(_prediction(kickoff=recent_kickoff), plan="free")

    assert out["probabilities"] is None
    assert out["delayed"] is True


def test_free_tier_reveals_probabilities_24h_after_kickoff() -> None:
    old_kickoff = datetime.now(timezone.utc) - timedelta(hours=25)
    out = serialize_prediction(_prediction(kickoff=old_kickoff), plan="free")

    assert out["probabilities"] == {"home": 0.45, "draw": 0.30, "away": 0.25}
    assert out["delayed"] is False


def test_free_tier_withholds_probabilities_for_future_kickoff() -> None:
    future_kickoff = datetime.now(timezone.utc) + timedelta(hours=3)
    out = serialize_prediction(_prediction(kickoff=future_kickoff), plan="free")

    assert out["probabilities"] is None
    assert out["delayed"] is True


# --- pro ------------------------------------------------------------------------


def test_pro_tier_adds_fair_edge_and_stake_fields() -> None:
    out = serialize_prediction(_prediction(), plan="pro")

    assert out["fair_probabilities"] == {
        "home": 1.0 / 2.2,
        "draw": 1.0 / 3.4,
        "away": 1.0 / 4.0,
    }
    assert out["edge_percent"] == {"home": 7.0, "draw": -2.0, "away": 1.0}
    assert out["recommended_selection"] == "home"
    assert out["kelly_stake_percent"] == 3.5
    assert "model_version" not in out
    assert "feature_breakdown" not in out
    assert "settlement" not in out


def test_pro_tier_never_delays_probabilities() -> None:
    """'live' — no 24h delay applies above free."""
    future_kickoff = datetime.now(timezone.utc) + timedelta(hours=3)
    out = serialize_prediction(_prediction(kickoff=future_kickoff), plan="pro")

    assert out["probabilities"] == {"home": 0.45, "draw": 0.30, "away": 0.25}
    assert out["delayed"] is False


# --- premium ----------------------------------------------------------------------


def test_premium_tier_adds_model_version_features_and_settlement() -> None:
    settled_at = datetime.now(timezone.utc)
    out = serialize_prediction(
        _prediction(
            result_settled=True,
            actual_outcome="home",
            was_win=True,
            profit=245.0,
            settled_at=settled_at,
        ),
        plan="premium",
    )

    assert out["model_version"] == MODEL_VERSION
    assert out["feature_breakdown"] is None  # not yet persisted — see docstring
    assert out["settlement"] == {
        "result_settled": True,
        "actual_outcome": "home",
        "was_win": True,
        "profit": 245.0,
        "settled_at": settled_at,
    }
    # Premium is a superset of pro.
    assert "fair_probabilities" in out
    assert "kelly_stake_percent" in out


def test_premium_tier_never_delays_probabilities() -> None:
    future_kickoff = datetime.now(timezone.utc) + timedelta(hours=3)
    out = serialize_prediction(_prediction(kickoff=future_kickoff), plan="premium")

    assert out["delayed"] is False
