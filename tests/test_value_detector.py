"""Tests for margin removal and Kelly staking."""

from __future__ import annotations

import pytest

from src.value_detector import ValueQuant


def test_remove_margin_sum_of_inverse_fair_odds() -> None:
    q = ValueQuant(kelly_fraction=0.25)
    odds = [2.0, 3.5, 4.0]
    fair = q.remove_margin(odds)
    inv = sum(1.0 / f for f in fair if f and f > 0)
    assert inv == pytest.approx(1.0, abs=1e-2)


def test_calculate_stake_non_negative() -> None:
    q = ValueQuant(kelly_fraction=0.25)
    stake = q.calculate_stake(0.05, 2.0)
    assert 0.0 <= stake <= 1.0
