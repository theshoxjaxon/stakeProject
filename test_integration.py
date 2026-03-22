"""Full system integration test."""

import numpy as np
from sqlalchemy.orm import Session
from src.database import get_engine, init_db
from src.models import Match, Prediction
from src.ai_advisor import get_ai_betting_advice
from datetime import datetime
import uuid
import json

def test_full_system():
    """
    Tests the full pipeline from prediction to persistence.
    1. Initializes a test database.
    2. Creates a mock match.
    3. Simulates a value bet scenario.
    4. Calls the AI advisor.
    5. Saves the prediction to the database.
    6. Verifies the prediction is saved correctly.
    """
    print("🚀 Starting Integration Test...")
    
    # 1. Initialize DB
    engine = get_engine("sqlite:///:memory:")
    init_db("sqlite:///:memory:")
    
    with Session(engine) as session:
        # 2. Create a mock match
        match_id = str(uuid.uuid4())
        mock_match = Match(
            id=match_id,
            date=datetime.utcnow(),
            home_team="Test United",
            away_team="Logic City",
            status="scheduled"
        )
        session.add(mock_match)
        session.commit()

        # 3. Simulate a value bet
        fake_matrix = np.random.dirichlet(np.ones(36), size=1).reshape(6, 6)
        match_data = {
            "home_team": mock_match.home_team,
            "away_team": mock_match.away_team,
            "model_prob_h": 0.45,
            "model_prob_d": 0.30,
            "model_prob_a": 0.25,
            "market_odds_h": 2.10,
            "market_odds_d": 3.40,
            "market_odds_a": 3.80,
            "value_side": "H",
            "value_edge": 0.05,
        }

        # 4. Test AI Advisor
        print("🤖 Testing Gemini Connection...")
        try:
            advice_str = get_ai_betting_advice(match_data, fake_matrix)
            advice = json.loads(advice_str)
            print(f"Gemini says: {advice.get('final_command', 'N/A')}") 
        except Exception as e:
            print(f"❌ AI Advisor Failed: {e}")

        # 5. Test Database Persistence
        print("📂 Testing Database Save...")
        stake_percent = 0.01
        stake_uzs = stake_percent * 500_000
        
        new_pred = Prediction(
            match_id=mock_match.id,
            home_prob=0.45,
            draw_prob=0.30,
            away_prob=0.25,
            market_home=2.10,
            market_draw=3.40,
            market_away=3.80,
            recommended_selection="H",
            recommended_stake_percent=stake_percent,
            recommended_stake_amount=stake_uzs,
            edge_used=0.05
        )
        session.add(new_pred)
        session.commit()
        
        # 6. Verify it exists
        saved = session.query(Prediction).filter_by(match_id=mock_match.id).first()
        if saved:
            print(f"✅ Success: Bet saved in vault (ID: {saved.id})")
            assert saved.recommended_selection == "H"
            assert saved.recommended_stake_amount == 5000
        else:
            print("❌ Error: Bet not found in DB.")
            assert False, "Prediction was not saved to the database"

if __name__ == "__main__":
    test_full_system()
