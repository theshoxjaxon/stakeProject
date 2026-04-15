"""Tests for Elo expected score and basic invariants."""

from __future__ import annotations

import pytest

from src.elo_model import expected_score


def test_expected_score_symmetry() -> None:
    r = 1500.0
    assert expected_score(r, r) == pytest.approx(0.5, abs=1e-9)


def test_expected_score_stronger_favorite() -> None:
    e = expected_score(1600.0, 1500.0)
    assert e > 0.5
    assert expected_score(1500.0, 1600.0) == pytest.approx(1.0 - e, abs=1e-9)
