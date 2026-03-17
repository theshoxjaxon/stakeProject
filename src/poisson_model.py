"""Poisson-based goal model (GoalEngine)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database import Match, ensure_team_ratings

MATRIX_SIZE = 6  # we model 0–5 goals
LEAGUE_AVG_GOALS_PER_MATCH = 2.7  # rough global prior


def _poisson_p(k: int, lam: float) -> float:
    """Return Poisson P(X = k) with mean lam."""
    if lam <= 0.0 or k < 0:
        return 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


@dataclass
class GoalEngine:
    """Goal probability engine using global attack/defense strengths."""

    attack_strength: Dict[str, float] = field(default_factory=dict)
    defense_strength: Dict[str, float] = field(default_factory=dict)

    def fit_from_matches(self, session: Session, min_matches: int = 5) -> None:
        """
        Estimate attack/defense strengths from completed matches in the DB.

        Teams with fewer than ``min_matches`` fall back to neutral strength (1.0).
        """
        stmt = select(Match).where(
            Match.status.in_(["completed", "finished"]),
            Match.home_score.isnot(None),
            Match.away_score.isnot(None),
        )
        matches = list(session.execute(stmt).scalars().all())
        if not matches:
            return

        team_stats: Dict[str, Dict[str, float]] = {}
        total_goals = 0.0
        total_matches = 0.0

        for m in matches:
            hs = float(m.home_score)
            as_ = float(m.away_score)
            total_goals += hs + as_
            total_matches += 1.0

            for team, gf, ga in ((m.home_team, hs, as_), (m.away_team, as_, hs)):
                entry = team_stats.setdefault(team, {"gf": 0.0, "ga": 0.0, "n": 0.0})
                entry["gf"] += gf
                entry["ga"] += ga
                entry["n"] += 1.0

        league_avg_goals_per_team = (total_goals / max(total_matches, 1.0)) / 2.0

        for team, st in team_stats.items():
            n = st["n"]
            if n < min_matches:
                self.attack_strength[team] = 1.0
                self.defense_strength[team] = 1.0
                continue

            avg_for = st["gf"] / n
            avg_against = st["ga"] / n
            self.attack_strength[team] = avg_for / league_avg_goals_per_team
            self.defense_strength[team] = avg_against / league_avg_goals_per_team

        ensure_team_ratings(session, team_stats.keys())

    def _lambda_for_team(self, team: str, opponent: str, is_home: bool) -> float:
        """Return expected goals (lambda) for a team vs an opponent."""
        att = self.attack_strength.get(team, 1.0)
        opp_def = self.defense_strength.get(opponent, 1.0)
        base = (LEAGUE_AVG_GOALS_PER_MATCH / 2.0) * att * (1.0 / max(opp_def, 1e-6))
        if is_home:
            base *= 1.05  # mild home boost
        return max(base, 1e-6)

    def _build_matrix(self, lam_home: float, lam_away: float) -> np.ndarray:
        """Build a 6×6 matrix of score probabilities."""
        m = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
        for i in range(MATRIX_SIZE):
            for j in range(MATRIX_SIZE):
                m[i, j] = _poisson_p(i, lam_home) * _poisson_p(j, lam_away)
        s = m.sum()
        if s > 0:
            m /= s
        return m

    def _hda_from_matrix(self, m: np.ndarray) -> Tuple[float, float, float]:
        """Return (P_home, P_draw, P_away) from score matrix."""
        p_draw = float(np.trace(m))
        p_home = float(np.sum(np.tril(m, -1)))
        p_away = float(np.sum(np.triu(m, 1)))
        total = p_home + p_draw + p_away
        if total > 0:
            p_home /= total
            p_draw /= total
            p_away /= total
        return p_home, p_draw, p_away

    def _over_under_25(self, m: np.ndarray) -> Tuple[float, float]:
        """Return (P_over_2_5, P_under_2_5) for total goals line 2.5."""
        p_over = 0.0
        p_under = 0.0
        for i in range(MATRIX_SIZE):
            for j in range(MATRIX_SIZE):
                if i + j >= 3:
                    p_over += m[i, j]
                else:
                    p_under += m[i, j]
        return p_over, p_under

    def _btts(self, m: np.ndarray) -> float:
        """Return probability that both teams score at least once."""
        p = 0.0
        for i in range(1, MATRIX_SIZE):
            for j in range(1, MATRIX_SIZE):
                p += m[i, j]
        return p

    def predict_match(
        self,
        home_team: str,
        away_team: str,
    ) -> Tuple[np.ndarray, float, float, float, float, float]:
        """
        Predict score distribution and summary probabilities for a match.

        Returns
        -------
        (matrix, p_home, p_draw, p_away, p_over_2_5, p_btts)
        """
        lam_home = self._lambda_for_team(home_team, away_team, is_home=True)
        lam_away = self._lambda_for_team(away_team, home_team, is_home=False)
        m = self._build_matrix(lam_home, lam_away)
        p_home, p_draw, p_away = self._hda_from_matrix(m)
        p_over, _ = self._over_under_25(m)
        p_btts = self._btts(m)
        return m, p_home, p_draw, p_away, p_over, p_btts

