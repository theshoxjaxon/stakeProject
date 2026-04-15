"""Value detection: Bivariate Poisson model with Elo xG, Kelly staking, and DNB flags."""

import math

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import DATABASE_PATH, EDGE_THRESHOLD
from src.database import get_engine, init_db
from src.models import Odds
from src.elo_model import LEAGUE_AVG_GOALS, elo_to_xg, get_elo_ratings
from src.logger import get_logger
from src.match_queries import matches_for_prediction

logger = get_logger(__name__)
HIGH_PROB_DRAW_THRESHOLD = 0.25  # 0-0 + 1-1 combined > 25%
MATRIX_SIZE = 6  # 0-5 goals per team


def poisson_probability(actual: int, lambda_val: float) -> float:
    """
    P(X = actual) for Poisson with mean lambda_val.
    Uses math.exp: (λ^k * e^(-λ)) / k!
    """
    if lambda_val <= 0 or actual < 0:
        return 0.0
    return (lambda_val**actual) * math.exp(-lambda_val) / math.factorial(actual)


def calculate_match_probs(
    home_expected_goals: float, away_expected_goals: float
) -> np.ndarray:
    """
    Build a 6x6 matrix of score probabilities (0-5 goals per team).
    P(i,j) = P(home=i) * P(away=j) via independent Poisson.
    """
    matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
    for i in range(MATRIX_SIZE):
        for j in range(MATRIX_SIZE):
            matrix[i, j] = poisson_probability(
                i, home_expected_goals
            ) * poisson_probability(j, away_expected_goals)
    return matrix


def build_goal_matrix(lambda_home: float, lambda_away: float) -> np.ndarray:
    """
    Build 6x6 matrix of score probabilities (alias for calculate_match_probs).
    """
    return calculate_match_probs(lambda_home, lambda_away)


def prob_over_under_25(matrix: np.ndarray) -> tuple[float, float]:
    """
    P(Over 2.5 goals) = sum of cells where home_goals + away_goals >= 3.
    P(Under 2.5 goals) = sum where home_goals + away_goals <= 2.
    """
    p_over = 0.0
    p_under = 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            total_goals = i + j
            if total_goals >= 3:
                p_over += matrix[i, j]
            else:
                p_under += matrix[i, j]
    return (p_over, p_under)


def prob_both_teams_to_score(matrix: np.ndarray) -> float:
    """
    P(BTTS) = sum of scorelines where both teams score at least 1 goal (i >= 1, j >= 1).
    """
    p_btts = 0.0
    for i in range(1, matrix.shape[0]):
        for j in range(1, matrix.shape[1]):
            p_btts += matrix[i, j]
    return p_btts


def probabilities_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    """
    Extract P(draw), P(home win), P(away win) from goal matrix.
    Draw: diagonal (0-0, 1-1, 2-2, ...)
    Home win: lower triangle (home_goals > away_goals)
    Away win: upper triangle (away_goals > home_goals)
    """
    p_draw = np.trace(matrix)
    p_home = np.sum(np.tril(matrix, -1))  # below diagonal
    p_away = np.sum(np.triu(matrix, 1))  # above diagonal
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total
    return (p_home, p_draw, p_away)


def is_high_probability_draw(matrix: np.ndarray) -> bool:
    """
    Flag when 0-0 and 1-1 combined exceed 25%.
    """
    p_00 = matrix[0, 0]
    p_11 = matrix[1, 1]
    return (p_00 + p_11) > HIGH_PROB_DRAW_THRESHOLD


def get_model_probabilities(
    session: Session, home_team: str, away_team: str
) -> tuple[float, float, float, np.ndarray | None]:
    """
    Return (P_home, P_draw, P_away) using Elo xG + Bivariate Poisson.
    Also returns the goal matrix for High-Probability Draw check.
    """
    r_home, r_away = get_elo_ratings(session, home_team, away_team)
    lambda_home, lambda_away = elo_to_xg(r_home, r_away, LEAGUE_AVG_GOALS)
    matrix = build_goal_matrix(lambda_home, lambda_away)
    p_h, p_d, p_a = probabilities_from_matrix(matrix)
    return (p_h, p_d, p_a, matrix)


def kelly_dnb_stake(p_win: float, p_draw: float, decimal_odds: float) -> float:
    """
    Kelly criterion for Draw No Bet (DNB) scenario.
    DNB: win if your side wins, stake returned if draw, lose if opponent wins.
    f* = (b*p - q) / b  where b = decimal_odds - 1, p = P(win), q = P(lose).
    Returns fraction of bankroll to stake (capped 0–1).
    """
    p = p_win
    q = 1.0 - p_win - p_draw  # P(lose)
    if decimal_odds <= 0:
        return 0.0
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    f = (b * p - q) / b
    return max(0.0, min(1.0, f))


def implied_probability(odds: float) -> float:
    """Implied probability from decimal odds: 1 / odds."""
    return 1.0 / odds if odds > 0 else 0.0


def get_latest_odds(
    session: Session, match_id: str
) -> tuple[float, float, float] | None:
    """Get the latest h_odds, d_odds, a_odds for a match (by max timestamp)."""
    stmt = (
        select(Odds.h_odds, Odds.d_odds, Odds.a_odds)
        .where(Odds.match_id == match_id)
        .order_by(Odds.timestamp.desc())
        .limit(1)
    )
    row = session.execute(stmt).first()
    return (row.h_odds, row.d_odds, row.a_odds) if row else None


def run_value_detection(edge_threshold: float | None = None) -> None:
    """
    Load matches whose kickoff is strictly in the future (UTC), compute implied vs model
    probability, print VALUE BETs, High-Probability Draws, and Kelly staking.
    """
    if edge_threshold is None:
        edge_threshold = EDGE_THRESHOLD
    init_db(DATABASE_PATH)
    engine = get_engine(DATABASE_PATH)

    with Session(engine) as session:
        # Kickoff in the future (UTC); optional horizon from PREDICTION_HORIZON_DAYS
        matches = list(session.execute(matches_for_prediction()).scalars().all())

        if not matches:
            logger.info("No upcoming matches found.")
            return

        for match in matches:
            odds_row = get_latest_odds(session, match.id)
            if odds_row is None:
                continue

            h_odds, d_odds, a_odds = odds_row
            impl_h = implied_probability(h_odds)
            impl_d = implied_probability(d_odds)
            impl_a = implied_probability(a_odds)

            model_h, model_d, model_a, matrix = get_model_probabilities(
                session, match.home_team, match.away_team
            )
            high_prob_draw = is_high_probability_draw(matrix)

            edge_h = model_h - impl_h
            edge_d = model_d - impl_d
            edge_a = model_a - impl_a

            has_value = (
                edge_h > edge_threshold
                or edge_d > edge_threshold
                or edge_a > edge_threshold
            )

            if has_value:
                flags = []
                if high_prob_draw:
                    flags.append("HIGH-PROB-DRAW")
                line = (
                    f"VALUE BET: {match.home_team} vs {match.away_team} "
                    f"(Home: {edge_h:.3f}, Draw: {edge_d:.3f}, Away: {edge_a:.3f})"
                )
                if flags:
                    line += f" [{', '.join(flags)}]"
                logger.info(line)

                # Kelly staking for DNB scenarios (Home DNB, Away DNB)
                kelly_h = kelly_dnb_stake(model_h, model_d, h_odds)
                kelly_a = kelly_dnb_stake(model_a, model_d, a_odds)
                if kelly_h > 0.01 or kelly_a > 0.01:
                    logger.info(
                        "  Kelly DNB: Home %.1f%%, Away %.1f%%",
                        kelly_h * 100,
                        kelly_a * 100,
                    )


if __name__ == "__main__":
    run_value_detection()
