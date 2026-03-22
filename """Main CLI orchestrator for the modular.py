"""Main CLI orchestrator for the modular football prediction engine."""

from __future__ import annotations
import datetime
import json
import numpy as np
from src.models import Prediction, Bet
from src.database import SessionLocal
from src.ai_advisor import get_ai_betting_advice, analyze_advanced_markets

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.backfill import run_backfill
from src.config import DATABASE_PATH, EDGE_THRESHOLD, SHOW_ONLY_VALUE_BETS
from src.database import get_engine, init_db
from src.models import Match, Odds
from src.fetch_data import run_update_cycle
from src.logger import get_logger
from src.match_queries import matches_for_prediction
from src.poisson_model import GoalEngine
from src.value_detector import ValueQuant
logger = get_logger(__name__)
# At the very top of main.py


def _fmt_kickoff(dt: datetime | None) -> str:
    """Format kickoff for logs (stored times treated as UTC if naive)."""
    if dt is None:
        return "?"
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return dt.strftime("%Y-%m-%d %H:%M") + " UTC"


def _db_is_empty() -> bool:
    """Return True if the matches table has no rows."""
    init_db(DATABASE_PATH)
    engine = get_engine(DATABASE_PATH)
    with Session(engine) as session:
        row = session.execute(select(Match.id).limit(1)).first()
        return row is None


def _get_latest_odds(
    session: Session, match_id: str
) -> tuple[float, float, float] | None:
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
    2. Fit Poisson strengths from historical matches (home/away split + optional features).
    3. Fetch current odds.
    4. Print predictions for every upcoming match with odds (sorted by kickoff), or only
       value-bet lines if ``SHOW_ONLY_VALUE_BETS`` is true.
    """
    init_db(DATABASE_PATH)
    engine = get_engine(DATABASE_PATH)

    if _db_is_empty():
        logger.info(
            "Database empty — running backfill to seed matches and Elo...")
        n = run_backfill()
        logger.info("Backfill updated %s matches.", n)

    with Session(engine) as session:
        goal_engine = GoalEngine()
        goal_engine.fit_from_matches(session)

        result = run_update_cycle()
        logger.info(
            "Fetched odds: %s matches, %s odds rows, %s sports.",
            result["matches_processed"],
            result["odds_added"],
            result["sports_processed"],
        )

        matches = list(session.execute(
            matches_for_prediction()).scalars().all())
        if not matches:
            logger.info("No upcoming matches found.")
            return

        quant = ValueQuant()
        skipped_no_odds = 0
        shown = 0

        logger.info("")
        logger.info(
            "Upcoming matches: %s (sorted by kickoff). Horizon/env: PREDICTION_HORIZON_DAYS if set.",
            len(matches),
        )

        if SHOW_ONLY_VALUE_BETS:
            logger.info(
                "SHOW_ONLY_VALUE_BETS=true — listing only sides with edge > %.1f%%",
                EDGE_THRESHOLD * 100,
            )
            logger.info(
                "%-36s %-6s %7s %7s %7s %7s",
                "Match",
                "Side",
                "Model%",
                "Fair%",
                "Edge%",
                "Stake%",
            )
            logger.info("-" * 88)

        for m in matches:
            odds_row = _get_latest_odds(session, m.id)
            if odds_row is None:
                skipped_no_odds += 1
                continue
            h_odds, d_odds, a_odds = odds_row
            if h_odds <= 0 or d_odds <= 0 or a_odds <= 0:
                skipped_no_odds += 1
                continue

            matrix, p_h, p_d, p_a, _, _ = goal_engine.predict_match(
                m.home_team,
                m.away_team,
                session=session,
                kickoff=m.date,
                use_features=True,
            )

            if not isinstance(matrix, np.ndarray) or matrix.shape != (6, 6):
                logger.warning(
                    f"Skipping match {m.id} due to invalid matrix shape: {matrix.shape if isinstance(matrix, np.ndarray) else 'Not an array'}"
                )
                continue

            # AI Advisor Integration
            match_data = {
                "home_team": m.home_team,
                "away_team": m.away_team,
                "model_prob_h": p_h,
                "model_prob_d": p_d,
                "model_prob_a": p_a,
                "market_odds_h": h_odds,
                "market_odds_d": d_odds,
                "market_odds_a": a_odds,
            }

            ai_advice_str = get_ai_betting_advice(match_data, matrix)
            # Print raw response
            logger.info(f"🤖 AI Advisor Raw Response: {ai_advice_str}")

            try:
                ai_advice = json.loads(ai_advice_str)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    f"🤖 AI Advisor did not return valid JSON: {ai_advice_str}")
                ai_advice = {"final_command": "Error",
                             "reasoning": "Invalid JSON from AI", "recommended_side": None}

            final_command = ai_advice.get('final_command', 'Unknown')
            ai_reasoning = ai_advice.get('reasoning', '')
            recommended_side = ai_advice.get('recommended_side')

            # Save the prediction
            with SessionLocal() as db_session:
                fair_odds = quant.remove_margin([h_odds, d_odds, a_odds])
                edge_h, edge_d, edge_a = None, None, None
                if fair_odds[0] > 0:
                    edge_h = p_h - (1 / fair_odds[0])
                if fair_odds[1] > 0:
                    edge_d = p_d - (1 / fair_odds[1])
                if fair_odds[2] > 0:
                    edge_a = p_a - (1 / fair_odds[2])

                prediction = Prediction(
                    match_id=m.id,
                    home_prob=p_h,
                    draw_prob=p_d,
                    away_prob=p_a,
                    market_home=h_odds,
                    market_draw=d_odds,
                    market_away=a_odds,
                    fair_home=1 / fair_odds[0] if fair_odds[0] > 0 else None,
                    fair_draw=1 / fair_odds[1] if fair_odds[1] > 0 else None,
                    fair_away=1 / fair_odds[2] if fair_odds[2] > 0 else None,
                    edge_home=edge_h,
                    edge_draw=edge_d,
                    edge_away=edge_a,
                    recommended_selection=recommended_side,
                )
                db_session.add(prediction)
                db_session.commit()

            # Display logic remains
            if SHOW_ONLY_VALUE_BETS:
                if recommended_side:  # If AI recommended a side
                    for side, model_p, mkt_odds, fair in [
                        ("H", p_h, h_odds, fair_odds[0]),
                        ("D", p_d, d_odds, fair_odds[1]),
                        ("A", p_a, a_odds, fair_odds[2]),
                    ]:
                        if side != recommended_side:
                            continue

                        if fair <= 0 or mkt_odds <= 0:
                            continue
                        implied_fair = 1.0 / fair
                        edge = model_p - implied_fair
                        if edge <= EDGE_THRESHOLD:
                            continue

                        stake_percent = quant.calculate_stake(edge, mkt_odds)
                        logger.info(
                            "%-36s %-6s %6.1f %6.1f %6.1f %6.1f",
                            f"{m.home_team} vs {m.away_team}"[:36],
                            side,
                            model_p * 100,
                            implied_fair * 100,
                            edge * 100,
                            stake_percent * 100,
                        )
                        shown += 1
            else:
                # Full report for every match
                logger.info("")
                logger.info("=" * 88)
                logger.info(
                    "Kickoff: %s | %s vs %s | id=%s",
                    _fmt_kickoff(m.date),
                    m.home_team,
                    m.away_team,
                    m.id,
                )
                logger.info(
                    "Model 1X2:  Home %5.1f%%  Draw %5.1f%%  Away %5.1f%%",
                    p_h * 100,
                    p_d * 100,
                    p_a * 100,
                )
                logger.info(
                    "%-4s %6s %7s %7s %7s %8s",
                    "Side",
                    "Mkt",
                    "Fair%",
                    "Edge%",
                    "Stake%",
                    "Note",
                )
                logger.info("-" * 88)

                for side, model_p, mkt_odds, fair in [
                    ("H", p_h, h_odds, fair_odds[0]),
                    ("D", p_d, d_odds, fair_odds[1]),
                    ("A", p_a, a_odds, fair_odds[2]),
                ]:
                    if fair <= 0 or mkt_odds <= 0:
                        logger.info(
                            "%-4s %6.2f %7s %7s %7s %8s",
                            side, mkt_odds, "—", "—", "—", "bad fair",
                        )
                        continue
                    implied_fair = 1.0 / fair
                    edge = model_p - implied_fair
                    stake = quant.calculate_stake(edge, mkt_odds)
                    note = ""
                    if recommended_side == side:
                        note = "AI Pick"
                    elif edge > EDGE_THRESHOLD:
                        note = "value"

                    stake_pct = stake * 100 if edge > EDGE_THRESHOLD else 0.0
                    logger.info(
                        "%-4s %6.2f %6.1f %+7.1f %6.1f %8s",
                        side, mkt_odds, implied_fair * 100, edge * 100, stake_pct, note,
                    )
                shown += 1

        if skipped_no_odds:
            logger.info(
                "Skipped %s upcoming match(es) with no usable 1X2 odds in DB.",
                skipped_no_odds,
            )
        logger.info("")
        logger.info(
            "Listed %s match(es) with odds. Edge threshold for 'value': %.1f%%.",
            shown,
            EDGE_THRESHOLD * 100,
        )


def main() -> None:
    """Entry point: run the prediction pipeline."""
    run_pipeline()


if __name__ == "__main__":
    main()
