"""Main CLI orchestrator for the modular football prediction engine."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.backfill import run_backfill
from src.config import DATABASE_PATH
from src.database import Match, Odds, get_engine, init_db
from src.fetch_data import run_update_cycle
from src.poisson_model import GoalEngine
from src.value_detector import ValueQuant


def _db_is_empty() -> bool:
    """Return True if the matches table has no rows."""
    init_db(DATABASE_PATH)
    engine = get_engine(DATABASE_PATH)
    with Session(engine) as session:
        row = session.execute(select(Match.id).limit(1)).first()
        return row is None


def _get_latest_odds(session: Session, match_id: str) -> tuple[float, float, float] | None:
    """Return latest (h_odds, d_odds, a_odds) for a match, or None."""
    stmt = (
        select(Odds.h_odds, Odds.d_odds, Odds.a_odds)
        .where(Odds.match_id == match_id)
        .order_by(Odds.timestamp.desc())
        .limit(1)
    )
    row = session.execute(stmt).first()
    if row is None:
        return None
    return float(row.h_odds), float(row.d_odds), float(row.a_odds)


def run_pipeline() -> None:
    """
    Run full flow:

    1. If DB is empty – run backfill to seed matches and Elo.
    2. Fit Poisson strengths from historical matches.
    3. Fetch current odds.
    4. Predict results and print value bets where edge > 5%.
    """
    init_db(DATABASE_PATH)
    engine = get_engine(DATABASE_PATH)

    if _db_is_empty():
        print("Database empty – running backfill to seed matches and Elo...")
        n = run_backfill()
        print(f"Backfill updated {n} matches.")

    with Session(engine) as session:
        goal_engine = GoalEngine()
        goal_engine.fit_from_matches(session)

        result = run_update_cycle()
        print(
            f"Fetched odds: {result['matches_processed']} matches, "
            f"{result['odds_added']} odds rows, {result['sports_processed']} sports."
        )

        matches = (
            session.execute(
                select(Match).where(Match.status.in_(["scheduled", "upcoming"]))
            )
            .scalars()
            .all()
        )
        if not matches:
            print("No upcoming matches found.")
            return

        quant = ValueQuant()

        print()
        print(f"{'Match':40} {'Side':6} {'Model%':7} {'Fair%':7} {'Edge%':7} {'Stake%':7}")
        print("-" * 80)

        for m in matches:
            odds_row = _get_latest_odds(session, m.id)
            if odds_row is None:
                continue
            h_odds, d_odds, a_odds = odds_row
            if h_odds <= 0 or d_odds <= 0 or a_odds <= 0:
                continue

            matrix, p_h, p_d, p_a, _, _ = goal_engine.predict_match(m.home_team, m.away_team)

            # Remove margin once per match for 1X2
            fair_odds = quant.remove_margin([h_odds, d_odds, a_odds])
            sides = [
                ("H", p_h, h_odds, fair_odds[0]),
                ("D", p_d, d_odds, fair_odds[1]),
                ("A", p_a, a_odds, fair_odds[2]),
            ]

            for side, model_p, mkt_odds, fair in sides:
                if fair <= 0 or mkt_odds <= 0:
                    continue
                implied_fair = 1.0 / fair
                edge = model_p - implied_fair
                if edge <= 0.05:  # 5% edge threshold
                    continue
                stake = quant.calculate_stake(edge, mkt_odds)
                print(
                    f"{m.home_team} vs {m.away_team:20} {side:6} "
                    f"{model_p*100:6.1f} {implied_fair*100:6.1f} {edge*100:6.1f} {stake*100:6.1f}"
                )


def main() -> None:
    """Entry point: run the prediction pipeline."""
    run_pipeline()


if __name__ == "__main__":
    main()

