"""Main CLI orchestrator for the modular football prediction engine."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.ai_advisor import get_ai_betting_advice
from src.backfill import run_backfill
from src.config import DATABASE_PATH, EDGE_THRESHOLD, SHOW_ONLY_VALUE_BETS
from src.database import SessionLocal, get_engine, init_db
from src.fetch_advanced import sync_league_xg
from src.fetch_data import run_update_cycle
from src.logger import get_logger
from src.match_queries import matches_for_prediction
from src.models import Bet, Match, Odds, Prediction
from src.poisson_model import GoalEngine
from src.value_detector import ValueQuant

logger = get_logger(__name__)

SPORT_LEAGUE_MAP = {
    "soccer_epl": "ENG-Premier League",
    "soccer_spain_la_liga": "ESP-La Liga",
    "soccer_germany_bundesliga": "GER-Bundesliga",
    "soccer_italy_serie_a": "ITA-Serie A",
    "soccer_france_ligue_one": "FRA-Ligue 1",
}


class ResilientAIClient:
    """AI wrapper that always returns a safe decision payload."""

    def get_betting_advice(self, match_data: dict, matrix: np.ndarray) -> dict:
        try:
            advice_raw = get_ai_betting_advice(match_data, matrix)
            advice = json.loads(advice_raw)
            if isinstance(advice, dict):
                return advice
            raise TypeError("AI response is not a JSON object.")
        except Exception as exc:
            logger.warning("AI advice fallback used: %s", exc)
            return {
                "final_command": "Error",
                "reasoning": f"AI advice error: {exc}",
                "recommended_side": None,
            }


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


def run_pipeline(ai_router: ResilientAIClient) -> None:
    """
    Run full flow:

    1. If DB is empty, run backfill to seed matches and Elo.
    2. Fit Poisson strengths from historical matches.
    3. Fetch current odds.
    4. Print predictions with value filtering.
    """
    init_db(DATABASE_PATH)
    engine = get_engine(DATABASE_PATH)

    from src.config import DEFAULT_SPORTS

    leagues_to_sync = [
        SPORT_LEAGUE_MAP[sport] for sport in DEFAULT_SPORTS if sport in SPORT_LEAGUE_MAP
    ]
    current_year = datetime.now().year
    seasons_to_sync = [current_year, current_year - 1]

    try:
        sync_league_xg(leagues=leagues_to_sync, seasons=seasons_to_sync)
    except Exception as exc:
        logger.error("Failed to sync xG data: %s", exc, exc_info=True)

    def infer_league_name(match_id: str) -> str:
        """Infer league label from match id text."""
        lower_id = match_id.lower()
        for sport_key, league_name in SPORT_LEAGUE_MAP.items():
            if sport_key in lower_id:
                return league_name
        return "Unknown"

    def normalize_stake_percent(stake_value: float) -> float:
        """Normalize stake values into a 0-100 percentage scale."""
        value = max(stake_value, 0.0)
        return value * 100.0 if value <= 1.0 else value

    def export_predictions() -> None:
        """Export prediction history once pipeline processing is complete."""
        try:
            with Session(engine) as history_session:
                rows = history_session.execute(
                    select(Prediction).order_by(Prediction.match_id, Prediction.created_at)
                ).scalars().all()

            grouped: dict[str, list[Prediction]] = {}
            for row in rows:
                grouped.setdefault(row.match_id, []).append(row)

            payload: list[dict] = []
            for match_id, history in grouped.items():
                for idx, row in enumerate(history):
                    # Keep last five probability deltas for sparkline-like rendering.
                    changes: list[dict[str, float]] = []
                    start_idx = max(1, idx - 4)
                    for i in range(start_idx, idx + 1):
                        prev = history[i - 1]
                        curr = history[i]
                        changes.append(
                            {
                                "home_delta": round(curr.home_prob - prev.home_prob, 6),
                                "draw_delta": round(curr.draw_prob - prev.draw_prob, 6),
                                "away_delta": round(curr.away_prob - prev.away_prob, 6),
                            }
                        )

                    payload.append(
                        {
                            "id": row.id,
                            "match_id": match_id,
                            "created_at": row.created_at.isoformat(),
                            "recommended_selection": row.recommended_selection,
                            "edge_home": row.edge_home,
                            "edge_draw": row.edge_draw,
                            "edge_away": row.edge_away,
                            "momentum_history": changes[-5:],
                        }
                    )

            out_dir = Path("history")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "predictions.json"
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.info("History export completed: %s rows -> %s", len(payload), out_path)
        except Exception as exc:
            logger.error("History export failed: %s", exc, exc_info=True)

    def export_league_exposure_heatmap() -> None:
        """Export bankroll exposure by league for currently open bets."""
        try:
            with Session(engine) as history_session:
                rows = history_session.execute(
                    select(Bet, Prediction, Match)
                    .join(Prediction, Bet.prediction_id == Prediction.id)
                    .join(Match, Prediction.match_id == Match.id)
                    .where(Bet.status == "pending")
                ).all()

            exposure_by_league: dict[str, float] = {}
            for bet, _, match in rows:
                league_name = infer_league_name(match.id)
                stake_pct = normalize_stake_percent(float(bet.stake_percent or 0.0))
                exposure_by_league[league_name] = (
                    exposure_by_league.get(league_name, 0.0) + stake_pct
                )

            heatmap_payload = [
                {"league": league, "bankroll_exposure_pct": round(pct, 2)}
                for league, pct in sorted(
                    exposure_by_league.items(), key=lambda item: item[1], reverse=True
                )
            ]

            out_dir = Path("history")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "exposure_heatmap.json"
            out_path.write_text(json.dumps(heatmap_payload, indent=2), encoding="utf-8")
            logger.info("Exposure heatmap export completed: %s", out_path)
        except Exception as exc:
            logger.error("Exposure heatmap export failed: %s", exc, exc_info=True)

    if _db_is_empty():
        logger.info("Database empty - running backfill to seed matches and Elo.")
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

        matches = list(session.execute(matches_for_prediction()).scalars().all())
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
                "SHOW_ONLY_VALUE_BETS=true - listing only sides with edge > %.1f%%",
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
                    "Skipping match %s due to invalid matrix shape: %s",
                    m.id,
                    matrix.shape if isinstance(matrix, np.ndarray) else "Not an array",
                )
                continue

            fair_odds = quant.remove_margin([h_odds, d_odds, a_odds])
            edge_h = p_h - (1 / fair_odds[0]) if fair_odds[0] > 0 else None
            edge_d = p_d - (1 / fair_odds[1]) if fair_odds[1] > 0 else None
            edge_a = p_a - (1 / fair_odds[2]) if fair_odds[2] > 0 else None
            edge_candidates = [edge for edge in [edge_h, edge_d, edge_a] if edge is not None]
            max_edge = max(edge_candidates) if edge_candidates else 0.0

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

            ai_advice = {
                "final_command": "Skip",
                "reasoning": f"No side exceeded edge threshold ({EDGE_THRESHOLD * 100:.1f}%).",
                "recommended_side": None,
            }
            if max_edge > EDGE_THRESHOLD:
                logger.info(
                    "Calling AI for %s vs %s (max edge %.1f%%).",
                    m.home_team,
                    m.away_team,
                    max_edge * 100,
                )
                ai_advice = ai_router.get_betting_advice(match_data, matrix)
            else:
                logger.info(
                    "Skipping AI for %s vs %s (max edge %.1f%% <= threshold %.1f%%).",
                    m.home_team,
                    m.away_team,
                    max_edge * 100,
                    EDGE_THRESHOLD * 100,
                )

            final_command = ai_advice.get("final_command", "Unknown")
            ai_reasoning = ai_advice.get("reasoning", "")
            recommended_side = ai_advice.get("recommended_side")
            logger.info(
                "AI decision for %s vs %s: %s | reason: %s",
                m.home_team,
                m.away_team,
                final_command,
                ai_reasoning,
            )

            with SessionLocal() as db_session:
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

            if SHOW_ONLY_VALUE_BETS:
                if recommended_side:
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
                    "Model 1X2: Home %5.1f%% Draw %5.1f%% Away %5.1f%%",
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
                            side,
                            mkt_odds,
                            "-",
                            "-",
                            "-",
                            "bad fair",
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
                        side,
                        mkt_odds,
                        implied_fair * 100,
                        edge * 100,
                        stake_pct,
                        note,
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

    threading.Thread(target=export_predictions, daemon=True).start()
    threading.Thread(target=export_league_exposure_heatmap, daemon=True).start()


def main() -> None:
    """Entry point: run the prediction pipeline."""
    ai_router = ResilientAIClient()
    run_pipeline(ai_router)


if __name__ == "__main__":
    main()
