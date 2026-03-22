"""Calculate and report on betting performance."""

from sqlalchemy import select
from sqlalchemy.orm import Session
from src.database import get_engine
from src.models import Prediction

def calculate_efficiency() -> None:
    """
    Calculates and prints a report of betting performance, including total bets,
    win rate, and total profit/loss.
    """
    engine = get_engine()
    with Session(engine) as session:
        # 1. Grab all settled predictions
        stmt = select(Prediction).where(Prediction.result_settled == True)
        completed_preds = list(session.execute(stmt).scalars().all())

        if not completed_preds:
            print("No settled bets yet. Run the settlement script after matches have finished.")
            return

        # 2. Calculate metrics
        total_bets = len(completed_preds)
        wins = sum(1 for p in completed_preds if p.was_win)
        win_rate = (wins / total_bets) * 100 if total_bets > 0 else 0
        total_profit = sum(p.profit for p in completed_preds)

        # 3. Print report
        print("--- 📊 ACCOUNTANT'S REPORT ---")
        print(f"Total Settled Bets: {total_bets}")
        print(f"Win Rate:           {win_rate:.1f}%")
        print(f"Total Profit/Loss:  {total_profit:,.0f} UZS")
        print("-------------------------------")

if __name__ == "__main__":
    calculate_efficiency()
