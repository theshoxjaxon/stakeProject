"""Save predictions to database for tracking and analysis."""

from datetime import datetime
from sqlalchemy.orm import Session
from src.models import Match, Prediction, Bet
from src.logger import get_logger

logger = get_logger(__name__)


def save_prediction(
    session: Session,
    match: Match,
    probs: dict,  # {'home': float, 'draw': float, 'away': float}
    odds: dict,   # {'home': float, 'draw': float, 'away': float}
    fair_odds: dict,  # {'home': float, 'draw': float, 'away': float}
    edges: dict,  # {'home': float, 'draw': float, 'away': float}
    stake_percent: float = 0.0,
    selected_side: str = None,
    edge_threshold: float = 0.05,
    bankroll: float = 500000  # Default in UZS
) -> Prediction:
    """
    Save a prediction to the database.
    
    Args:
        session: SQLAlchemy session
        match: Match object
        probs: Dict with 'home', 'draw', 'away' probabilities (0-1)
        odds: Dict with 'home', 'draw', 'away' market odds
        fair_odds: Dict with 'home', 'draw', 'away' fair odds
        edges: Dict with 'home', 'draw', 'away' edges (%)
        stake_percent: Recommended stake percentage (0-100)
        selected_side: 'home', 'draw', 'away', or None
        edge_threshold: Minimum edge to recommend bet (e.g., 0.05 = 5%)
        bankroll: Current bankroll in currency units
    
    Returns:
        Created Prediction object
    """
    
    # Calculate stake amount
    stake_amount = (stake_percent / 100) * bankroll if stake_percent > 0 else 0.0
    
    # Determine recommended bet
    recommended_selection = None
    recommended_stake = 0.0
    edge_used = 0.0
    
    if selected_side and edges.get(selected_side, 0) >= edge_threshold:
        recommended_selection = selected_side
        recommended_stake = stake_percent
        edge_used = edges[selected_side]
    
    # Create prediction record
    prediction = Prediction(
        match_id=match.id,
        created_at=datetime.utcnow(),
        home_prob=probs['home'],
        draw_prob=probs['draw'],
        away_prob=probs['away'],
        market_home=odds.get('home'),
        market_draw=odds.get('draw'),
        market_away=odds.get('away'),
        fair_home=fair_odds.get('home'),
        fair_draw=fair_odds.get('draw'),
        fair_away=fair_odds.get('away'),
        edge_home=edges.get('home'),
        edge_draw=edges.get('draw'),
        edge_away=edges.get('away'),
        recommended_selection=recommended_selection,
        recommended_stake_percent=recommended_stake,
        recommended_stake_amount=stake_amount,
        edge_used=edge_used,
        result_settled=False
    )
    
    session.add(prediction)
    session.flush()  # Get ID without committing
    
    # If there's a recommended bet, create a bet record (pending placement)
    if recommended_selection and recommended_stake > 0:
        bet = Bet(
            prediction_id=prediction.id,
            selection=recommended_selection,
            odds=odds.get(recommended_selection),
            stake_amount=stake_amount,
            stake_percent=recommended_stake,
            status='pending'
        )
        session.add(bet)
        logger.info(f"📊 BET RECOMMENDED: {match.home_team} vs {match.away_team} -> {recommended_selection.upper()} @ {odds.get(recommended_selection):.2f} | Stake: {recommended_stake:.1f}% ({stake_amount:,.0f} UZS) | Edge: +{edge_used:.1f}%")
    
    session.commit()
    logger.debug(f"Prediction saved for match {match.id}: {match.home_team} vs {match.away_team}")
    
    return prediction


def update_prediction_result(
    session: Session,
    match_id: str,
    home_score: int,
    away_score: int
) -> bool:
    """
    Update prediction with actual match result.
    
    Args:
        session: SQLAlchemy session
        match_id: Match ID (string from Odds API)
        home_score: Actual home goals
        away_score: Actual away goals
    
    Returns:
        True if updated, False if not found
    """
    
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
        logger.warning(f"No prediction found for match {match_id}")
        return False
    
    prediction.actual_outcome = actual_outcome
    prediction.result_settled = True
    prediction.settled_at = datetime.utcnow()
    
    # Update associated bet if exists
    if prediction.bet:
        if prediction.bet.selection == actual_outcome:
            prediction.bet.status = 'won'
            # Calculate profit: stake * (odds - 1)
            prediction.bet.profit = prediction.bet.stake_amount * (prediction.bet.odds - 1)
            logger.info(f"✅ BET WON! {prediction.bet.selection.upper()} - Profit: +{prediction.bet.profit:,.0f} UZS")
        else:
            prediction.bet.status = 'lost'
            prediction.bet.profit = -prediction.bet.stake_amount
            logger.info(f"❌ BET LOST: {prediction.bet.selection.upper()} - Loss: {prediction.bet.stake_amount:,.0f} UZS")
        
        prediction.bet.settled_at = datetime.utcnow()
    
    session.commit()
    logger.info(f"Updated prediction for {prediction.match.home_team} vs {prediction.match.away_team}: {actual_outcome.upper()}")
    
    return True


def get_performance_summary(session: Session, days: int = 7) -> dict:
    """
    Get performance summary for recent predictions.
    
    Args:
        session: SQLAlchemy session
        days: Number of days to look back
    
    Returns:
        Dict with performance metrics
    """
    
    from datetime import timedelta
    
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    # Get settled predictions with bets
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


def get_todays_predictions(session: Session) -> list:
    """Get all predictions made today."""
    today = datetime.utcnow().date()
    tomorrow = today.replace(day=today.day + 1)
    
    return session.query(Prediction).filter(
        Prediction.created_at >= today,
        Prediction.created_at < tomorrow
    ).all()