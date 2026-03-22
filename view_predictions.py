#!/usr/bin/env python3
"""View saved predictions and performance."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.database import get_session
from src.models import Prediction, Bet
from datetime import datetime


def get_performance_summary(session, days: int = 7):
    """Get performance summary for recent bets."""
    from datetime import timedelta
    
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    # Get settled bets
    bets = session.query(Bet).filter(
        Bet.status.in_(['won', 'lost']),
        Bet.placed_at >= cutoff
    ).all()
    
    if not bets:
        return {
            'total_bets': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'total_profit': 0,
            'total_staked': 0,
            'roi': 0
        }
    
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
        'total_staked': total_staked,
        'roi': (total_profit / total_staked * 100) if total_staked > 0 else 0
    }


def main():
    """Display saved predictions."""
    with get_session() as session:
        print("\n" + "="*80)
        print("📊 PREDICTION TRACKING DASHBOARD")
        print("="*80)
        
        # Check if predictions table exists and has data
        try:
            # Count total predictions
            total_predictions = session.query(Prediction).count()
            
            if total_predictions == 0:
                print("\n⚠️  No predictions found in database.")
                print("Run 'python main.py' first to generate predictions.\n")
                return
            
            print(f"\n📈 Total predictions in database: {total_predictions}")
            
            # Get recent predictions
            predictions = session.query(Prediction).order_by(
                Prediction.created_at.desc()
            ).limit(20).all()
            
            print(f"\n📋 RECENT PREDICTIONS (last {len(predictions)}):")
            print("-"*80)
            
            for p in predictions:
                match = p.match
                if not match:
                    continue
                    
                status = "✅" if p.result_settled else "⏳"
                outcome = f"→ {p.actual_outcome.upper()}" if p.actual_outcome else ""
                
                # Format date
                match_date = match.date.strftime('%Y-%m-%d %H:%M') if match.date else "Unknown"
                
                print(f"\n{status} {match_date} | {match.home_team} vs {match.away_team}")
                print(f"   Model: H:{p.home_prob:.1%} D:{p.draw_prob:.1%} A:{p.away_prob:.1%}")
                
                if p.bet:
                    bet_status = p.bet.status
                    if bet_status == 'won':
                        bet_icon = "💰 WON"
                    elif bet_status == 'lost':
                        bet_icon = "💸 LOST"
                    else:
                        bet_icon = "⏳ PENDING"
                    
                    print(f"   Bet: {p.bet.selection.upper()} @ {p.bet.odds:.2f} | Stake: {p.bet.stake_amount:,.0f} UZS | {bet_icon}")
                    
                    if p.bet.profit != 0:
                        profit_icon = "➕" if p.bet.profit > 0 else "➖"
                        print(f"   Profit: {profit_icon} {abs(p.bet.profit):,.0f} UZS")
                else:
                    print(f"   No bet recommended (edge below threshold)")
            
            # Performance summary
            print("\n" + "="*80)
            print("📈 PERFORMANCE SUMMARY")
            print("-"*80)
            
            for days in [1, 7, 30]:
                summary = get_performance_summary(session, days=days)
                if summary['total_bets'] > 0:
                    print(f"\nLast {days} days:")
                    print(f"   Bets: {summary['total_bets']} | Wins: {summary['wins']} | Losses: {summary['losses']}")
                    print(f"   Win rate: {summary['win_rate']:.1f}%")
                    print(f"   Profit: {summary['total_profit']:+,.0f} UZS")
                    print(f"   ROI: {summary['roi']:+.1f}%")
                else:
                    print(f"\nLast {days} days: No settled bets yet")
            
        except Exception as e:
            print(f"\n❌ Error accessing database: {e}")
            print("Make sure you've run migrations: 'alembic upgrade head'")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    main()