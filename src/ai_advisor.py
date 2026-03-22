"""AI Advisor for complex betting markets and risk management."""

import os
import numpy as np
from google import genai
from dotenv import load_dotenv

from src.config import EDGE_THRESHOLD

# 1. Load Environment Variables
load_dotenv()

# 2. Configure Gemini Safely
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # This prevents the script from running if the key is missing in .env
    raise ValueError("❌ Error: GEMINI_API_KEY not found in .env file!")

# Initialize the client
client = genai.Client(api_key=api_key)


def analyze_advanced_markets(matrix):
    """
    Slices the 6x6 Poisson matrix to extract complex probabilities.
    Logic:
    - BTTS: Any score where both teams score >= 1.
    - Away Win: Any score where Away goals > Home goals.
    - BTTS + Away Win: Slices the matrix to find scores like 1-2, 1-3, 2-3, etc.
    """
    # Ensure we are working with a NumPy array for mathematical operations
    matrix = np.array(matrix)

    # Add validation check for the matrix
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        return {
            "BTTS": 0.0,
            "Away_Win": 0.0,
            "BTTS_Away_Win": 0.0
        }

    # 1. Probability of Both Teams to Score (Exclude row 0 and column 0)
    prob_btts = np.sum(matrix[1:, 1:])

    # 2. Probability of Away Win (Upper triangle of the matrix)
    prob_away_win = np.sum(np.triu(matrix, k=1))

    # 3. Probability of Away Win AND BTTS
    # Slice the matrix to remove the 'Home 0 goals' row, then find Away Wins
    prob_btts_away = np.sum(np.triu(matrix[1:, :], k=1))

    return {
        "BTTS": round(float(prob_btts) * 100, 2),
        "Away_Win": round(float(prob_away_win) * 100, 2),
        "BTTS_Away_Win": round(float(prob_btts_away) * 100, 2)
    }


def get_ai_betting_advice(match_data: dict, matrix: np.ndarray) -> str:
    """
    Uses Gemini 1.5 Pro to provide a final risk assessment for the bet.
    """
    advanced = analyze_advanced_markets(matrix)

    prompt = f"""
    ROLE: Professional Sports Betting Risk Manager.

    You are analyzing a football match using a statistical model.
    The model has generated a **Poisson Matrix** of score probabilities.
    Based on this matrix and market odds, provide a **Final Decision**.
    
    MATCH: {match_data['home_team']} vs {match_data['away_team']}
    
    MODEL PREDICTIONS (from Poisson Matrix):
    - Home Win Probability: {match_data['model_prob_h']*100:.1f}%
    - Draw Probability: {match_data['model_prob_d']*100:.1f}%
    - Away Win Probability: {match_data['model_prob_a']*100:.1f}%
    - Both Teams To Score (BTTS) Probability: {advanced['BTTS']}%
    
    BOOKMAKER DATA:
    - Market Odds for Home Win: {match_data['market_odds_h']}
    - Market Odds for Draw: {match_data['market_odds_d']}
    - Market Odds for Away Win: {match_data['market_odds_a']}
    
    ANALYSIS TASK: 
    1. Compare model probabilities to market odds to find value (edge > {EDGE_THRESHOLD*100:.1f}%).
    2. If value is found, recommend a bet.
    3. If no value is found, recommend skipping.
    4. Provide your **Final Decision** as a valid JSON object with keys 'final_command' ('Bet' or 'Skip'), 'reasoning', and 'recommended_side' ('H', 'D', 'A', or null).
    
    Example for value: {{"final_command": "Bet", "reasoning": "Model shows edge on Home.", "recommended_side": "H"}}
    Example for no value: {{"final_command": "Skip", "reasoning": "No significant value detected.", "recommended_side": null}}
    """

    try:
        response = client.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents=prompt
        )
        # It's good practice to strip the response and remove backticks if the LLM wraps it in a code block
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        return text
    except Exception as e:
        return f'{{"final_command": "Error", "reasoning": "AI Advice Error: {str(e)}", "recommended_side": null}}'
