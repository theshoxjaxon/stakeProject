"""Poisson-based goal model (GoalEngine) with optional home/away strength split and feature multipliers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, Iterable

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models import Match, TeamRating

MATRIX_SIZE = 6  # we model 0–5 goals
LEAGUE_AVG_GOALS_PER_MATCH = 2.7  # rough global prior


def ensure_team_ratings(session: Session, team_names: Iterable[str]):
    # This assumes TeamRating is defined in src.models
    cleaned = {t for t in team_names if t}
    if not cleaned:
        return
    existing = {name for (name,) in session.query(
        TeamRating.team_name).filter(TeamRating.team_name.in_(cleaned)).all()}
    for name in cleaned - existing:
        session.add(TeamRating(team_name=name))
    session.commit()



def _poisson_p(k: int, lam: float) -> float:
    """Return Poisson P(X = k) with mean lam."""
    if lam <= 0.0 or k < 0:
        return 0.0
    return (lam**k) * math.exp(-lam) / math.factorial(k)


@dataclass
class GoalEngine:
    """Goal probability engine using global attack/defense strengths."""

    attack_strength: Dict[str, float] = field(default_factory=dict)
    defense_strength: Dict[str, float] = field(default_factory=dict)
    attack_home: Dict[str, float] = field(default_factory=dict)
    attack_away: Dict[str, float] = field(default_factory=dict)
    defense_home: Dict[str, float] = field(default_factory=dict)
    defense_away: Dict[str, float] = field(default_factory=dict)

    def fit_from_matches(self, session: Session, min_matches: int = 5) -> None:
        """
        Estimate attack/defense strengths from completed matches in the DB.

        Teams with fewer than ``min_matches`` total appearances fall back to neutral (1.0).
        When enough home/away games exist per team, separate home vs away strengths are used.
        """
        stmt = select(Match).where(
            Match.status.in_(["completed", "finished"]),
            Match.home_score.isnot(None),
            Match.away_score.isnot(None),
        )
        matches = list(session.execute(stmt).scalars().all())
        if not matches:
            return

        # Aggregate goals for league averages
        total_goals = 0.0
        total_home_goals = 0.0
        total_away_goals = 0.0
        n_m = float(len(matches))

        # Per team: goals for/conceded when home vs away
        ph: Dict[str, Dict[str, float]] = {}
        pa: Dict[str, Dict[str, float]] = {}

        for m in matches:
            hs = float(m.home_score)
            as_ = float(m.away_score)
            total_goals += hs + as_
            total_home_goals += hs
            total_away_goals += as_

            h, a = m.home_team, m.away_team
            ph.setdefault(h, {"gf": 0.0, "ga": 0.0, "n": 0.0})
            ph[h]["gf"] += hs
            ph[h]["ga"] += as_
            ph[h]["n"] += 1.0
            pa.setdefault(a, {"gf": 0.0, "ga": 0.0, "n": 0.0})
            pa[a]["gf"] += as_
            pa[a]["ga"] += hs
            pa[a]["n"] += 1.0

        league_avg_goals_per_team = (total_goals / max(n_m, 1.0)) / 2.0
        avg_home = total_home_goals / max(n_m, 1.0)
        avg_away = total_away_goals / max(n_m, 1.0)

        all_teams = set(ph.keys()) | set(pa.keys())

        for team in all_teams:
            st_all = ph.get(team, {"gf": 0.0, "ga": 0.0, "n": 0.0})
            sa_all = pa.get(team, {"gf": 0.0, "ga": 0.0, "n": 0.0})
            n_total = st_all["n"] + sa_all["n"]
            if n_total < min_matches:
                self.attack_strength[team] = 1.0
                self.defense_strength[team] = 1.0
                self.attack_home[team] = 1.0
                self.attack_away[team] = 1.0
                self.defense_home[team] = 1.0
                self.defense_away[team] = 1.0
                continue

            # Blended (legacy) strengths
            gf_t = st_all["gf"] + sa_all["gf"]
            ga_t = st_all["ga"] + sa_all["ga"]
            self.attack_strength[team] = (
                gf_t / max(n_total, 1.0)
            ) / league_avg_goals_per_team
            self.defense_strength[team] = (
                ga_t / max(n_total, 1.0)
            ) / league_avg_goals_per_team

            # Home split
            sh = ph.get(team, {"gf": 0.0, "ga": 0.0, "n": 0.0})
            nh = sh["n"]
            if nh >= max(2, min_matches // 2):
                self.attack_home[team] = (sh["gf"] / nh) / max(avg_home, 1e-6)
                self.defense_home[team] = (sh["ga"] / nh) / max(avg_home, 1e-6)
            else:
                self.attack_home[team] = self.attack_strength[team]
                self.defense_home[team] = self.defense_strength[team]

            # Away split
            sa = pa.get(team, {"gf": 0.0, "ga": 0.0, "n": 0.0})
            na = sa["n"]
            if na >= max(2, min_matches // 2):
                self.attack_away[team] = (sa["gf"] / na) / max(avg_away, 1e-6)
                self.defense_away[team] = (sa["ga"] / na) / max(avg_away, 1e-6)
            else:
                self.attack_away[team] = self.attack_strength[team]
                self.defense_away[team] = self.defense_strength[team]

        ensure_team_ratings(session, all_teams)

    def _lambda_for_team(self, team: str, opponent: str, is_home: bool) -> float:
        """Return expected goals (lambda) for a team vs an opponent."""
        if is_home:
            att = self.attack_home.get(team, self.attack_strength.get(team, 1.0))
            opp_def = self.defense_away.get(
                opponent, self.defense_strength.get(opponent, 1.0)
            )
        else:
            att = self.attack_away.get(team, self.attack_strength.get(team, 1.0))
            opp_def = self.defense_home.get(
                opponent, self.defense_strength.get(opponent, 1.0)
            )

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
        session: Session | None = None,
        kickoff: object | None = None,
        use_features: bool = True,
    ) -> Tuple[np.ndarray, float, float, float, float, float]:
        """
        Predict score distribution and summary probabilities for a match.

        When ``session`` and ``kickoff`` are provided and ``FEATURES_ENABLED`` is true,
        applies form / H2H / rest multipliers from ``feature_engineering``.
        """
        lam_home = self._lambda_for_team(home_team, away_team, is_home=True)
        lam_away = self._lambda_for_team(away_team, home_team, is_home=False)

        if use_features and session is not None and kickoff is not None:
            from datetime import datetime

            from src.config import FEATURES_ENABLED, FORM_WINDOW

            if FEATURES_ENABLED:
                from src.feature_engineering import compute_lambda_multipliers

                if isinstance(kickoff, datetime):
                    mult = compute_lambda_multipliers(
                        session, home_team, away_team, kickoff, FORM_WINDOW
                    )
                    lam_home *= mult.home
                    lam_away *= mult.away

        m = self._build_matrix(lam_home, lam_away)
        p_home, p_draw, p_away = self._hda_from_matrix(m)
        p_over, _ = self._over_under_25(m)
        p_btts = self._btts(m)
        return m, p_home, p_draw, p_away, p_over, p_btts
