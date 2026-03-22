"""Match report: projected score, BTTS, value indicator, and fractional Kelly staking."""

from __future__ import annotations

import numpy as np

from src.config import BTTS_VALUE_THRESHOLD, KELLY_FRACTION
from src.logger import get_logger

logger = get_logger(__name__)


def kelly_stake(
    edge: float, odds: float, kelly_fraction: float = KELLY_FRACTION
) -> float:
    """
    Recommend fraction of bankroll to bet using Fractional Kelly (0.25 multiplier).
    edge = our_prob - implied_prob; odds = decimal odds.
    Returns stake fraction in [0, 1].
    """
    if odds <= 0:
        return 0.0
    implied = 1.0 / odds
    p = implied + edge  # our estimated probability
    q = 1.0 - p
    b = odds - 1.0
    if b <= 0:
        return 0.0
    f_full = (b * p - q) / b
    f_frac = kelly_fraction * f_full
    return max(0.0, min(1.0, f_frac))


def projected_score(matrix: np.ndarray) -> tuple[int, int]:
    """The most likely score from the matrix (argmax over (i, j))."""
    flat_idx = np.argmax(matrix)
    i = flat_idx // matrix.shape[1]
    j = flat_idx % matrix.shape[1]
    return (int(i), int(j))


def btts_value_indicator(model_btts_pct: float, implied_btts_pct: float) -> bool:
    """True if our BTTS % is at least 5 percentage points above bookie implied (value)."""
    return model_btts_pct >= implied_btts_pct + (BTTS_VALUE_THRESHOLD * 100)


def format_match_report(
    home_team: str,
    away_team: str,
    matrix: np.ndarray,
    btts_odds: float | None = None,
) -> str:
    """
    Build a refined report line: Projected Score, BTTS %, Value indicator.
    btts_odds: optional decimal odds for BTTS; if provided, value shown when model is 5%+ above implied.
    """
    from src.predict import prob_both_teams_to_score

    proj_h, proj_a = projected_score(matrix)
    p_btts = prob_both_teams_to_score(matrix)
    btts_pct = p_btts * 100

    lines = [
        f"{home_team} vs {away_team}",
        f"  Projected Score: {proj_h}-{proj_a}",
        f"  BTTS Probability: {btts_pct:.1f}%",
    ]

    if btts_odds is not None and btts_odds > 0:
        implied_pct = (1.0 / btts_odds) * 100
        value = btts_value_indicator(btts_pct, implied_pct)
        lines.append(f"  BTTS Implied (book): {implied_pct:.1f}%")
        if value:
            edge = (btts_pct / 100.0) - (1.0 / btts_odds)
            stake_pct = kelly_stake(edge, btts_odds) * 100
            lines.append("  BTTS: VALUE (model ≥5% above book)")
            lines.append(f"  Recommended stake (¼ Kelly): {stake_pct:.1f}%")

    return "\n".join(lines)


def run_reports(
    btts_odds_by_match_id: dict[str, float] | None = None,
) -> None:
    """
    Print refined reports for matches with kickoff in the future (same filter as predictions).

    btts_odds_by_match_id: optional map of match_id -> BTTS decimal odds for value check.
    """
    from src.config import DATABASE_PATH
    from src.database import get_engine, init_db
    from src.predict import get_model_probabilities

    init_db(DATABASE_PATH)
    engine = get_engine(DATABASE_PATH)
    btts_odds_by_match_id = btts_odds_by_match_id or {}

    from sqlalchemy.orm import Session

    from src.match_queries import matches_for_prediction

    with Session(engine) as session:
        matches = list(session.execute(matches_for_prediction()).scalars().all())

        if not matches:
            logger.info("No upcoming matches.")
            return

        for match in matches:
            _, _, _, matrix = get_model_probabilities(
                session, match.home_team, match.away_team
            )
            if matrix is None:
                continue
            btts_odds = btts_odds_by_match_id.get(match.id)
            logger.info(
                "%s\n",
                format_match_report(
                    match.home_team, match.away_team, matrix, btts_odds
                ),
            )


if __name__ == "__main__":
    run_reports()
