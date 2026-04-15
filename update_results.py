#!/usr/bin/env python3
"""Update predictions with actual match results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.database import get_session
from src.models import Match, Prediction, Bet
from src.logger import get_logger
from datetime import datetime, timedelta

logger = get_logger(__name__)


def update_prediction_result(session, match_id, home_score, away_score):
    """Update prediction with actual result."""
    # Determine actual outcome
    if home_score > away_score:
        actual_outcome = 'home'
    elif away_score > home_score:
        actual_outcome = 'away'
    else:
        actual_outcome = 'draw'
    
    # Find the latest prediction for this match
    prediction = session.query(Prediction).filter(
        Prediction.match_id == match_id
    ).order_by(Prediction.created_at.desc()).first()
    
    if not prediction:
        return False
    
    prediction.actual_outcome = actual_outcome
    prediction.result_settled = True
    prediction.settled_at = datetime.utcnow()
    
    # Update associated bet if exists
    if prediction.bet:
        if prediction.bet.selection == actual_outcome:
            prediction.bet.status = 'won'
            prediction.bet.profit = prediction.bet.stake_amount * (prediction.bet.odds - 1)
            logger.info(f"✅ BET WON! {prediction.bet.selection.upper()} - Profit: +{prediction.bet.profit:,.0f} UZS")
        else:
            prediction.bet.status = 'lost'
            prediction.bet.profit = -prediction.bet.stake_amount
            logger.info(f"❌ BET LOST: {prediction.bet.selection.upper()} - Loss: {prediction.bet.stake_amount:,.0f} UZS")
        
        prediction.bet.settled_at = datetime.utcnow()
    
    session.commit()
    return True


def get_performance_summary(session, days: int = 7):
    """Get performance summary."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    bets = session.query(Bet).filter(
        Bet.status.in_(['won', 'lost']),
        Bet.placed_at >= cutoff
    ).all()
    
    if not bets:
        return {'total_bets': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'total_profit': 0, 'roi': 0}
    
    wins = sum(1 for b in bets if b.status == 'won')
    losses = sum(1 for b in bets if b.status == 'lost')
    total_profit = sum(b.profit for b in bets)
    total_staked = sum(b.stake_amount for b in bets)
    
    return {
        'total_bets': len(bets),
        'wins': wins,
        'losses': losses,
        'win_rate': (wins / len(bets) * 100) if bets else 0,
        'total_profit': total_profit,
        'roi': (total_profit / total_staked * 100) if total_staked > 0 else 0
    }


def main():
    """Update results for completed matches."""
    logger.info("🔄 Starting result update...")
    
    with get_session() as session:
        # Get matches that finished in the last 7 days and have scores
        completed = session.query(Match).filter(
            Match.status.in_(['completed', 'finished']),
            Match.home_score.isnot(None),
            Match.away_score.isnot(None),
            Match.date >= datetime.utcnow() - timedelta(days=7)
        ).all()
        
        updated_count = 0
        for match in completed:
            if update_prediction_result(session, match.id, match.home_score, match.away_score):
                updated_count += 1
                logger.info(f"✅ Updated: {match.home_team} {match.home_score}-{match.away_score} {match.away_team}")
        
        logger.info(f"📊 Updated {updated_count} predictions")
        
        # Show performance
        summary = get_performance_summary(session, days=7)
        if summary['total_bets'] > 0:
            print("\n" + "="*60)
            print("📈 PERFORMANCE (Last 7 Days)")
            print("="*60)
            print(f"Total bets: {summary['total_bets']}")
            print(f"Wins: {summary['wins']}")
            print(f"Losses: {summary['losses']}")
            print(f"Win rate: {summary['win_rate']:.1f}%")
            print(f"Total profit: {summary['total_profit']:+,.0f} UZS")
            print(f"ROI: {summary['roi']:+.1f}%")
            print("="*60)


if __name__ == "__main__":
    main()