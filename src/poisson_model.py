"""
Production-grade goal prediction model based on the Dixon-Coles model,
which extends the independent Poisson model to account for the correlation
between home and away goals, particularly the low-scoring draw bias.

The model parameters are estimated via maximum likelihood estimation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError

from src.models import Match, TeamRating

# Use the project's logger
logger = logging.getLogger(__name__)

MATRIX_SIZE = 6  # Set to 6x6 to align with main.py pipeline expectations


def dixon_coles_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """The Dixon-Coles adjustment function for low-scoring draws."""
    if x == 0 and y == 0:
        return 1 - (lam * mu * rho)
    if x == 0 and y == 1:
        return 1 + (lam * rho)
    if x == 1 and y == 0:
        return 1 + (mu * rho)
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _dixon_coles_log_likelihood(
    params: np.ndarray, matches: pd.DataFrame, teams: list[str]
) -> float:
    """Negative log-likelihood function for the Dixon-Coles model."""
    num_teams = len(teams)
    attack_params = params[0:num_teams]
    defence_params = params[num_teams : 2 * num_teams]
    home_adv = params[2 * num_teams]
    rho = np.clip(params[2 * num_teams + 1], -0.19, 0.19)

    team_map = {team: i for i, team in enumerate(teams)}
    
    home_indices = matches["home_team"].map(team_map).values
    away_indices = matches["away_team"].map(team_map).values
    
    home_goals = matches["home_score"].values
    away_goals = matches["away_score"].values

    lam = np.exp(attack_params[home_indices] + defence_params[away_indices] + home_adv)
    mu = np.exp(attack_params[away_indices] + defence_params[home_indices])

    tau = np.ones_like(lam)
    tau[(home_goals == 0) & (away_goals == 0)] = 1 - (lam[(home_goals == 0) & (away_goals == 0)] * mu[(home_goals == 0) & (away_goals == 0)] * rho)
    tau[(home_goals == 0) & (away_goals == 1)] = 1 + (lam[(home_goals == 0) & (away_goals == 1)] * rho)
    tau[(home_goals == 1) & (away_goals == 0)] = 1 + (mu[(home_goals == 1) & (away_goals == 0)] * rho)
    tau[(home_goals == 1) & (away_goals == 1)] = 1 - rho
    
    tau = np.clip(tau, 1e-10, None)
    
    log_likelihood = np.sum(
        -lam + home_goals * np.log(lam) - gammaln(home_goals + 1) +
        -mu + away_goals * np.log(mu) - gammaln(away_goals + 1) +
        np.log(tau)
    )
    return -log_likelihood


@dataclass
class GoalEngine:
    """
    A goal prediction model based on the Dixon-Coles paper.
    """
    teams: list[str] = field(default_factory=list)
    attack_params: dict[str, float] = field(default_factory=dict)
    defence_params: dict[str, float] = field(default_factory=dict)
    home_advantage: float = 0.0
    rho: float = 0.0
    
    def fit_from_matches(self, session: Session):
        """
        Fits the model parameters by maximizing the log-likelihood of observed matches.
        """
        logger.info("Fitting Dixon-Coles model from database matches...")

        try:
            stmt = select(Match).where(Match.status.in_(["completed", "finished"]))
            matches_df = pd.read_sql(stmt, session.get_bind())
        except OperationalError as e:
            raise RuntimeError(
                f"Database error fetching historical matches for model fitting: {e}"
            ) from e

        if matches_df.empty:
            raise RuntimeError(
                "Cannot fit model: no completed matches in the database. "
                "Run a data backfill before calling fit_from_matches()."
            )

        self.teams = sorted(list(set(matches_df["home_team"]) | set(matches_df["away_team"])))
        num_teams = len(self.teams)

        initial_params = np.concatenate([
            np.full(num_teams, 0.0), np.full(num_teams, 0.0), [0.1], [0.0],
        ])

        constraints = [{'type': 'eq', 'fun': lambda x: sum(x[0:num_teams])}]

        logger.info(f"Optimizing parameters for {num_teams} teams over {len(matches_df)} matches...")
        
        res = minimize(
            _dixon_coles_log_likelihood,
            initial_params,
            args=(matches_df, self.teams),
            constraints=constraints,
            method='SLSQP',
        )

        if not res.success:
            logger.error(f"Model fitting failed: {res.message}")
            return

        logger.info(f"Model fitted successfully. Final log-likelihood: {-res.fun:.2f}")

        opt_params = res.x
        team_map = {team: i for i, team in enumerate(self.teams)}
        self.attack_params = {team: opt_params[team_map[team]] for team in self.teams}
        self.defence_params = {team: opt_params[num_teams + team_map[team]] for team in self.teams}
        self.home_advantage = opt_params[2 * num_teams]
        self.rho = np.clip(opt_params[2 * num_teams + 1], -0.19, 0.19)
        
        self._save_parameters(session)

    def _save_parameters(self, session: Session):
        """Saves the fitted attack and defense parameters to the TeamRating table."""
        logger.info("Saving fitted model parameters to the database...")
        for team_name in self.teams:
            rating = session.execute(
                select(TeamRating).where(TeamRating.team_name == team_name)
            ).scalar_one_or_none()
            if not rating:
                rating = TeamRating(team_name=team_name)
                session.add(rating)
            
            rating.attack_strength = self.attack_params.get(team_name)
            rating.defense_strength = self.defence_params.get(team_name)
        
        session.commit()
        logger.info("Model parameters saved successfully.")

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        session: Session | None = None,
        kickoff: datetime | None = None,
        use_features: bool = True,
    ) -> tuple[np.ndarray, float, float, float, float, float]:
        """
        Predicts the outcome of a match using the fitted Dixon-Coles model.
        """
        home_known = home_team in self.teams
        away_known = away_team in self.teams

        if not home_known or not away_known:
            if session is None:
                logger.warning(
                    f"Cold-start: {home_team!r} or {away_team!r} not in fitted model "
                    "and no session provided for Elo fallback — returning uniform probabilities."
                )
                return np.zeros((MATRIX_SIZE, MATRIX_SIZE)), 0.33, 0.34, 0.33, 0.0, 0.0
            from src.elo_model import elo_to_xg, get_elo_ratings
            r_home, r_away = get_elo_ratings(session, home_team, away_team)
            lam, mu = elo_to_xg(r_home, r_away)
            logger.info(
                f"Cold-start: unknown team(s) [{home_team!r} known={home_known}, "
                f"{away_team!r} known={away_known}] — "
                f"Elo fallback λ_home={lam:.3f}, λ_away={mu:.3f} "
                f"(r_home={r_home:.0f}, r_away={r_away:.0f})"
            )
        else:
            lam = math.exp(self.attack_params.get(home_team, 0.0) + self.defence_params.get(away_team, 0.0) + self.home_advantage)
            mu = math.exp(self.attack_params.get(away_team, 0.0) + self.defence_params.get(home_team, 0.0))

        if use_features and session and kickoff:
            try:
                from src.feature_engineering import compute_lambda_multipliers
                mults = compute_lambda_multipliers(session, home_team, away_team, kickoff, 5) # Window is hardcoded for now
                lam *= mults.home
                mu *= mults.away
            except Exception as e:
                logger.warning(f"Could not apply feature multipliers: {e}", exc_info=True)

        matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
        for i in range(MATRIX_SIZE):
            for j in range(MATRIX_SIZE):
                tau = dixon_coles_tau(i, j, lam, mu, self.rho)
                matrix[i, j] = tau * ((lam**i * math.exp(-lam)) / math.factorial(i)) * ((mu**j * math.exp(-mu)) / math.factorial(j))

        matrix /= matrix.sum()
        
        p_home = float(np.sum(np.tril(matrix, -1)))
        p_draw = float(np.trace(matrix))
        p_away = float(np.sum(np.triu(matrix, 1)))
        p_over_2_5 = float(np.sum(matrix[np.add.outer(range(MATRIX_SIZE), range(MATRIX_SIZE)) >= 3]))
        p_btts = float(np.sum(matrix[1:, 1:]))

        return matrix, p_home, p_draw, p_away, p_over_2_5, p_btts
