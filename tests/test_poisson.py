"""Tests for Poisson goal model helpers."""

from __future__ import annotations

import pytest

from src.poisson_model import GoalEngine, _poisson_p


def test_poisson_p_sum_near_one() -> None:
    lam = 1.4
    s = sum(_poisson_p(k, lam) for k in range(20))
    assert s == pytest.approx(1.0, abs=1e-3)


def test_goal_engine_matrix_stochastic() -> None:
    g = GoalEngine()
    g.attack_strength["A"] = 1.0
    g.attack_strength["B"] = 1.0
    g.defense_strength["A"] = 1.0
    g.defense_strength["B"] = 1.0
    g.attack_home["A"] = g.attack_away["A"] = 1.0
    g.attack_home["B"] = g.attack_away["B"] = 1.0
    g.defense_home["A"] = g.defense_away["A"] = 1.0
    g.defense_home["B"] = g.defense_away["B"] = 1.0
    m, ph, pd, pa, _, _ = g.predict_match("A", "B", use_features=False)
    assert m.shape == (6, 6)
    assert ph + pd + pa == pytest.approx(1.0, abs=1e-6)
    assert float(m.sum()) == pytest.approx(1.0, abs=1e-6)
