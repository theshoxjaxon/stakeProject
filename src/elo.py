"""Elo rating engine for football matches."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pow
from typing import Dict, Tuple


@dataclass
class EloEngine:
    """
    Standalone Elo rating engine.

    Attributes
    ----------
    default_rating:
        Initial rating assigned to any team with no prior history.
    home_field_advantage:
        Elo points added to the home team when computing win probabilities.
    k_factor:
        Sensitivity of rating updates to new results.
    ratings:
        In‑memory map of team name -> current Elo rating.
    """

    default_rating: float = 1500.0
    home_field_advantage: float = 50.0
    k_factor: float = 20.0
    ratings: Dict[str, float] = field(default_factory=dict)

    def get_rating(self, team: str) -> float:
        """Return the current rating for a team, initializing if unknown."""
        if team not in self.ratings:
            self.ratings[team] = self.default_rating
        return self.ratings[team]

    def get_win_probs(self, rating_diff: float) -> Tuple[float, float]:
        """
        Calculate Elo-based win probabilities from a rating difference.

        Parameters
        ----------
        rating_diff:
            Difference in ratings (home_rating + home_field_advantage - away_rating).

        Returns
        -------
        (p_home, p_away):
            Probability home team wins, probability away team wins.
            Draws are not handled here and should be modeled separately.
        """
        p_home = 1.0 / (1.0 + pow(10.0, -rating_diff / 400.0))
        p_away = 1.0 - p_home
        return p_home, p_away

    def update_ratings(
        self,
        home_team: str,
        away_team: str,
        score: tuple[int, int],
    ) -> tuple[float, float]:
        """
        Update ratings for a completed match.

        Parameters
        ----------
        home_team:
            Name of the home team.
        away_team:
            Name of the away team.
        score:
            Tuple of (home_goals, away_goals).

        Returns
        -------
        (new_home_rating, new_away_rating)
        """
        home_goals, away_goals = score

        r_home = self.get_rating(home_team)
        r_away = self.get_rating(away_team)

        # Apply home-field advantage only in the probability calculation
        r_home_adj = r_home + self.home_field_advantage
        expected_home = 1.0 / (1.0 + pow(10.0, (r_away - r_home_adj) / 400.0))
        expected_away = 1.0 - expected_home

        if home_goals > away_goals:
            s_home, s_away = 1.0, 0.0
        elif home_goals < away_goals:
            s_home, s_away = 0.0, 1.0
        else:
            s_home = s_away = 0.5

        r_home_new = r_home + self.k_factor * (s_home - expected_home)
        r_away_new = r_away + self.k_factor * (s_away - expected_away)

        self.ratings[home_team] = r_home_new
        self.ratings[away_team] = r_away_new

        return r_home_new, r_away_new

