"""Sports betting value detection system - main entry point."""

from src.backfill_elo import run_backfill_elo
from src.fetch_data import fetch_historical_scores, run_update_cycle


def main() -> None:
    """Run full pipeline: 1. Fetch Upcoming Odds, 2. Fetch Historical Scores, 3. Backfill Elo."""
    result = run_update_cycle()
    print(
        f"Upcoming odds: {result['matches_processed']} matches, "
        f"{result['odds_added']} odds, {result['sports_processed']} sports"
    )
    scores = fetch_historical_scores()
    print(
        f"Historical scores: {scores['matches_updated']} matches updated, "
        f"{scores['sports_processed']} sports"
    )
    backfilled = run_backfill_elo()
    print(f"Elo backfill: {backfilled} matches processed.")


if __name__ == "__main__":
    main()
