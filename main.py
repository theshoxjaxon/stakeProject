"""Main CLI orchestrator for the modular football prediction engine.

Pipeline: (optional xG sync) → backfill if empty → fetch odds (leagues +
in-season tournaments) → fit Dixon-Coles → predict every upcoming match with
odds → save/update one Prediction row per match → print the value table.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.ai_advisor import advisor_enabled, get_ai_betting_advice
from src.backfill import run_backfill
from src.elo_model import BASE_RATING, get_elo_ratings
from src.config import (
    DATABASE_PATH,
    EDGE_THRESHOLD,
    SHOW_ONLY_VALUE_BETS,
    XG_SYNC_ENABLED,
)
from src.database import get_engine, init_db
from src.fetch_data import run_update_cycle
from src.logger import get_logger
from src.match_queries import matches_for_prediction
from src.models import Match, Odds, Prediction
from src.poisson_model import GoalEngine
from src.tournaments import sport_title
from src.value_detector import ValueQuant

logger = get_logger(__name__)

# Odds API sport key -> soccerdata/FBref league id (used only for xG sync).
SPORT_LEAGUE_MAP = {
    "soccer_epl": "ENG-Premier League",
    "soccer_spain_la_liga": "ESP-La Liga",
    "soccer_germany_bundesliga": "GER-Bundesliga",
    "soccer_italy_serie_a": "ITA-Serie A",
    "soccer_france_ligue_one": "FRA-Ligue 1",
}

_SIDES = ("H", "D", "A")


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


def _sync_xg_if_enabled() -> None:
    """Scrape FBref xG when XG_SYNC_ENABLED=true (slow; off by default)."""
    if not XG_SYNC_ENABLED:
        return
    from src.config import DEFAULT_SPORTS
    from src.fetch_advanced import sync_league_xg

    leagues = [SPORT_LEAGUE_MAP[s] for s in DEFAULT_SPORTS if s in SPORT_LEAGUE_MAP]
    year = datetime.now(timezone.utc).year
    try:
        sync_league_xg(leagues=leagues, seasons=[year, year - 1])
    except Exception as e:
        logger.error("Failed to sync xG data: %s", e, exc_info=True)


def _consult_ai_advisor(
    match: Match,
    probs: dict[str, float],
    odds: dict[str, float],
    matrix: np.ndarray,
) -> tuple[str, str | None, str]:
    """
    Ask Gemini for a verdict on a value candidate.

    Returns (verdict, side, reasoning) where verdict is one of:
    - "bet":   AI confirms; side is 'H' | 'D' | 'A'
    - "skip":  AI vetoes the bet
    - "error": AI unavailable / unusable response — caller should fail open
    """
    payload = {
        "home_team": match.home_team,
        "away_team": match.away_team,
        "model_prob_h": probs["H"],
        "model_prob_d": probs["D"],
        "model_prob_a": probs["A"],
        "market_odds_h": odds["H"],
        "market_odds_d": odds["D"],
        "market_odds_a": odds["A"],
    }
    raw = get_ai_betting_advice(payload, matrix)
    try:
        advice = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("AI advisor returned invalid JSON, ignoring: %.200s", raw)
        return "error", None, "invalid AI response"

    command = advice.get("final_command")
    side = advice.get("recommended_side")
    reasoning = " ".join(str(advice.get("reasoning", "")).split())[:120]
    if command == "Bet" and side in _SIDES:
        return "bet", side, reasoning
    if command == "Skip":
        return "skip", None, reasoning
    return "error", None, reasoning


def _upsert_prediction(
    session: Session,
    match: Match,
    probs: dict[str, float],
    market: dict[str, float],
    fair_probs: dict[str, float | None],
    edges: dict[str, float | None],
    recommended_side: str | None,
    stake_percent: float,
) -> None:
    """One live Prediction row per match: update the unsettled one or insert."""
    existing = session.execute(
        select(Prediction)
        .where(Prediction.match_id == match.id, Prediction.result_settled.is_(False))
        .order_by(Prediction.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()

    values = dict(
        home_prob=probs["H"],
        draw_prob=probs["D"],
        away_prob=probs["A"],
        market_home=market["H"],
        market_draw=market["D"],
        market_away=market["A"],
        fair_home=fair_probs["H"],
        fair_draw=fair_probs["D"],
        fair_away=fair_probs["A"],
        edge_home=edges["H"],
        edge_draw=edges["D"],
        edge_away=edges["A"],
        recommended_selection=recommended_side,
        recommended_stake_percent=stake_percent,
        edge_used=edges.get(recommended_side) if recommended_side else None,
    )

    if existing is not None:
        for key, val in values.items():
            setattr(existing, key, val)
    else:
        session.add(Prediction(match_id=match.id, **values))


def run_pipeline() -> None:
    """
    Run the full flow:

    1. Optional xG sync (XG_SYNC_ENABLED).
    2. If DB is empty — backfill recent results to seed matches and Elo.
    3. Fetch current odds for leagues + in-season tournaments.
    4. Fit the Dixon-Coles engine from completed matches.
    5. Predict every upcoming match with odds; save predictions; print table.
    """
    init_db(DATABASE_PATH)
    engine = get_engine(DATABASE_PATH)

    _sync_xg_if_enabled()

    if _db_is_empty():
        logger.info("Database empty — running backfill to seed matches and Elo...")
        n = run_backfill()
        logger.info("Backfill updated %s matches.", n)

    # Fetch odds before opening the long-lived read session below — SQLite
    # write transactions and an open read snapshot don't mix well.
    result = run_update_cycle()
    logger.info(
        "Odds sync: %s matches, %s odds inserted, %s updated, %s sport(s) fresh from cache, %s sport(s) total.",
        result["matches_processed"],
        result["odds_inserted"],
        result["odds_updated"],
        result["sports_skipped_cache"],
        result["sports_processed"],
    )

    use_ai = advisor_enabled()
    if not use_ai:
        logger.info("AI advisor disabled — running pure quant recommendations.")

    with Session(engine) as session:
        goal_engine = GoalEngine()
        try:
            goal_engine.fit_from_matches(session)
        except RuntimeError as exc:
            logger.warning("%s — continuing with Elo fallback only.", exc)

        matches = list(session.execute(matches_for_prediction()).scalars().all())
        if not matches:
            logger.info("No upcoming matches found.")
            return

        quant = ValueQuant()
        skipped_no_odds = 0
        shown = 0
        value_bets = 0
        ai_consecutive_errors = 0
        _AI_ERROR_LIMIT = 2  # circuit breaker: stop calling a failing AI API

        logger.info("")
        logger.info(
            "Upcoming matches: %s (sorted by kickoff). Edge threshold: %.1f%%.",
            len(matches),
            EDGE_THRESHOLD * 100,
        )

        if SHOW_ONLY_VALUE_BETS:
            logger.info(
                "SHOW_ONLY_VALUE_BETS=true — listing only sides with edge > %.1f%%",
                EDGE_THRESHOLD * 100,
            )
            logger.info(
                "%-34s %-22s %-4s %7s %7s %7s %7s",
                "Match", "Competition", "Side", "Model%", "Fair%", "Edge%", "Stake%",
            )
            logger.info("-" * 96)

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
                logger.warning("Skipping match %s — invalid score matrix.", m.id)
                continue

            probs = {"H": p_h, "D": p_d, "A": p_a}
            market = {"H": h_odds, "D": d_odds, "A": a_odds}
            fair_odds = quant.remove_margin([h_odds, d_odds, a_odds])
            fair_probs: dict[str, float | None] = {}
            edges: dict[str, float | None] = {}
            for side, fair in zip(_SIDES, fair_odds):
                if fair and fair > 0:
                    fair_probs[side] = 1.0 / fair
                    edges[side] = probs[side] - fair_probs[side]
                else:
                    fair_probs[side] = None
                    edges[side] = None

            # Cold start with no rating history at all (typical for national
            # teams the first time a tournament appears): the probabilities
            # are close to uninformative priors, so any "edge" against the
            # market is noise. Show the numbers, but never recommend a bet.
            home_known = m.home_team in goal_engine.teams
            away_known = m.away_team in goal_engine.teams
            no_information = False
            if not home_known or not away_known:
                r_h, r_a = get_elo_ratings(session, m.home_team, m.away_team)
                no_information = r_h == BASE_RATING and r_a == BASE_RATING

            # Best quant side above the edge threshold (if any).
            quant_side = None
            best_edge = EDGE_THRESHOLD
            if not no_information:
                for side in _SIDES:
                    if edges[side] is not None and edges[side] > best_edge:
                        quant_side = side
                        best_edge = edges[side]

            # AI advisor is consulted only for value candidates — it can veto
            # the quant pick but never invent a bet the quant model didn't find.
            # If the AI API itself fails, fail open to the quant pick.
            recommended_side = quant_side
            ai_note = ""
            if quant_side and use_ai and ai_consecutive_errors < _AI_ERROR_LIMIT:
                verdict, ai_side, reasoning = _consult_ai_advisor(
                    m, probs, market, matrix
                )
                if verdict == "bet" and edges.get(ai_side) is not None:
                    recommended_side = ai_side
                    ai_note = f"AI: {reasoning}" if reasoning else "AI confirmed"
                    ai_consecutive_errors = 0
                elif verdict == "skip":
                    recommended_side = None
                    ai_note = f"AI veto: {reasoning}" if reasoning else "AI veto"
                    ai_consecutive_errors = 0
                else:
                    ai_note = "AI unavailable — quant only"
                    ai_consecutive_errors += 1
                    if ai_consecutive_errors == _AI_ERROR_LIMIT:
                        logger.warning(
                            "AI advisor failed %d times in a row — skipping it for the rest of this run.",
                            _AI_ERROR_LIMIT,
                        )

            stake = 0.0
            if recommended_side:
                stake = quant.stake_from_prob(
                    probs[recommended_side], market[recommended_side]
                )
                value_bets += 1

            _upsert_prediction(
                session, m, probs, market, fair_probs, edges,
                recommended_side, stake * 100,
            )

            competition = sport_title(m.sport_key)

            if SHOW_ONLY_VALUE_BETS:
                if recommended_side:
                    logger.info(
                        "%-34s %-22s %-4s %6.1f %6.1f %+6.1f %6.1f",
                        f"{m.home_team} vs {m.away_team}"[:34],
                        competition[:22],
                        recommended_side,
                        probs[recommended_side] * 100,
                        (fair_probs[recommended_side] or 0) * 100,
                        (edges[recommended_side] or 0) * 100,
                        stake * 100,
                    )
                    shown += 1
                continue

            # Full per-match report
            logger.info("")
            logger.info("=" * 96)
            logger.info(
                "%s | %s | %s vs %s",
                _fmt_kickoff(m.date), competition, m.home_team, m.away_team,
            )
            logger.info(
                "Model 1X2:  Home %5.1f%%  Draw %5.1f%%  Away %5.1f%%",
                p_h * 100, p_d * 100, p_a * 100,
            )
            logger.info(
                "%-4s %7s %7s %7s %7s  %s",
                "Side", "Mkt", "Fair%", "Edge%", "Stake%", "Note",
            )
            logger.info("-" * 96)
            for side in _SIDES:
                if fair_probs[side] is None:
                    logger.info(
                        "%-4s %7.2f %7s %7s %7s  %s",
                        side, market[side], "—", "—", "—", "bad fair odds",
                    )
                    continue
                note = ""
                if side == recommended_side:
                    note = "BET" + (f" — {ai_note}" if ai_note else "")
                elif side == quant_side and recommended_side is None:
                    note = ai_note or "value (vetoed)"
                elif edges[side] is not None and edges[side] > EDGE_THRESHOLD:
                    note = (
                        "edge ignored — no team history (cold start)"
                        if no_information
                        else "value"
                    )
                logger.info(
                    "%-4s %7.2f %7.1f %+7.1f %7.1f  %s",
                    side,
                    market[side],
                    fair_probs[side] * 100,
                    edges[side] * 100,
                    (stake * 100 if side == recommended_side else 0.0),
                    note,
                )
            shown += 1

        session.commit()

        logger.info("")
        logger.info("=" * 96)
        if skipped_no_odds:
            logger.info(
                "Skipped %s upcoming match(es) with no usable 1X2 odds in DB.",
                skipped_no_odds,
            )
        logger.info(
            "Listed %s match(es); %s bet recommendation(s) above the %.1f%% edge threshold.",
            shown, value_bets, EDGE_THRESHOLD * 100,
        )


def main() -> None:
    """Entry point: run the prediction pipeline."""
    run_pipeline()


if __name__ == "__main__":
    main()
