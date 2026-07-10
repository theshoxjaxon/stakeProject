"""Settle outstanding predictions by comparing them to completed match results."""

from datetime import datetime
from sqlalchemy import select
from sqlalchemy.orm import Session
from src.database import get_engine
from src.models import Prediction, Match


def settle_bets() -> None:
    """
    Matches saved predictions with actual results and updates the database.
    This function processes unsettled predictions, determines their outcomes based on
    final match scores, and updates their status, win/loss record, and profit.
    """
    engine = get_engine()
    with Session(engine) as session:
        # 1. Grab all predictions that haven't been settled yet
        stmt = select(Prediction).where(Prediction.result_settled.is_(False))
        pending_preds = list(session.execute(stmt).scalars().all())

        if not pending_preds:
            print("✅ No pending bets to settle.")
            return

        print(f"🧐 Processing {len(pending_preds)} pending bets...")

        for pred in pending_preds:
            # 2. Look for the corresponding match result
            match = session.get(Match, pred.match_id)

            if (
                not match
                or match.status not in ("completed", "finished")
                or match.home_score is None
                or match.away_score is None
            ):
                continue

            h_score = match.home_score
            a_score = match.away_score

            # 3. Determine actual outcome
            actual_outcome = "D"
            if h_score > a_score:
                actual_outcome = "H"
            elif a_score > h_score:
                actual_outcome = "A"

            # Predictions with no recommended side ("Skip") carry no stake:
            # record the outcome but keep them out of win/loss accounting.
            if pred.recommended_selection is None:
                pred.actual_outcome = actual_outcome
                pred.result_settled = True
                pred.settled_at = datetime.utcnow()
                continue

            # 4. Logic: Did the prediction win?
            won = (pred.recommended_selection == actual_outcome)

            market_odds = {
                "H": pred.market_home,
                "D": pred.market_draw,
                "A": pred.market_away,
            }.get(pred.recommended_selection) or 0.0

            # 5. Calculate profit
            profit = -pred.recommended_stake_amount
            if won and market_odds > 0:
                profit = pred.recommended_stake_amount * (market_odds - 1)

            # 6. Update the Prediction record
            pred.actual_outcome = actual_outcome
            pred.was_win = won
            pred.profit = profit
            pred.result_settled = True
            pred.settled_at = datetime.utcnow()

            status = "WIN ✅" if won else "LOSS ❌"
            print(
                f"Match: {match.home_team} vs {match.away_team} | Result: {h_score}-{a_score} | "
                f"Prediction: {pred.recommended_selection} @ {market_odds:.2f} | "
                f"Stake: {pred.recommended_stake_amount:.0f} | "
                f"Profit: {profit:.0f} | {status}"
            )

        session.commit()
        print("💾 Database updated. You can now run the Accountant Report.")


if __name__ == "__main__":
    settle_bets()
