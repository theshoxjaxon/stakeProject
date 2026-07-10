"""Tests for the Dixon-Coles goal model."""

from __future__ import annotations

import numpy as np
import pytest

from src.poisson_model import GoalEngine, dixon_coles_tau


def _fitted_engine() -> GoalEngine:
    g = GoalEngine()
    g.teams = ["A", "B"]
    g.attack_params = {"A": 0.1, "B": -0.1}
    g.defence_params = {"A": -0.05, "B": 0.05}
    g.home_advantage = 0.2
    g.rho = -0.05
    return g


def test_dixon_coles_tau_neutral_for_high_scores() -> None:
    assert dixon_coles_tau(3, 2, 1.5, 1.1, -0.1) == 1.0


def test_dixon_coles_tau_adjusts_low_scores() -> None:
    rho = -0.1
    assert dixon_coles_tau(0, 0, 1.5, 1.1, rho) == pytest.approx(1 - 1.5 * 1.1 * rho)
    assert dixon_coles_tau(1, 1, 1.5, 1.1, rho) == pytest.approx(1 - rho)


def test_goal_engine_matrix_stochastic() -> None:
    g = _fitted_engine()
    m, ph, pd_, pa, p_over, p_btts = g.predict_match("A", "B", use_features=False)
    assert isinstance(m, np.ndarray)
    assert m.shape == (6, 6)
    assert float(m.sum()) == pytest.approx(1.0, abs=1e-6)
    assert ph + pd_ + pa == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= p_over <= 1.0
    assert 0.0 <= p_btts <= 1.0


def test_home_advantage_tilts_probabilities() -> None:
    g = _fitted_engine()
    # Same two equal teams: the home side should win more often than the away side.
    g.attack_params = {"A": 0.0, "B": 0.0}
    g.defence_params = {"A": 0.0, "B": 0.0}
    _, ph, _, pa, _, _ = g.predict_match("A", "B", use_features=False)
    assert ph > pa


def test_unknown_team_without_session_returns_uniform() -> None:
    g = _fitted_engine()
    m, ph, pd_, pa, _, _ = g.predict_match("A", "Unknown FC", use_features=False)
    assert m.shape == (6, 6)
    assert ph + pd_ + pa == pytest.approx(1.0, abs=1e-2)
