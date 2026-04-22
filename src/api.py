"""FastAPI web server for the EdgeAI sports betting trading terminal."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from src.config import DATABASE_PATH
from src.database import get_engine, init_db
from src.match_queries import (
    BetSortField,
    SortDir,
    get_bet_history,
    matches_for_prediction,
)
from src.models import Match
from src.poisson_model import GoalEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CORS — configurable via CORS_ORIGINS env var (comma-separated list of origins)
# Defaults to every common local frontend dev-server port so the React app
# on localhost works out of the box without touching .env.
# ---------------------------------------------------------------------------

_DEFAULT_CORS_ORIGINS: list[str] = [
    "http://localhost:3000",   # Create React App
    "http://localhost:5173",   # Vite (React / Vue / Svelte)
    "http://localhost:5174",   # Vite alt port
    "http://localhost:8080",   # Vue CLI / Webpack dev server
    "http://localhost:4200",   # Angular CLI
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]


def _resolve_cors_origins() -> list[str]:
    """Return CORS origins from env var or fall back to local-dev defaults."""
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return _DEFAULT_CORS_ORIGINS


# ---------------------------------------------------------------------------
# Lifespan — fit GoalEngine exactly once at startup, reuse across all requests.
# If the DB is empty, the engine starts unfitted and /predict returns 503 until
# a backfill + /model/refit is called.
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise database schema and fit the Dixon-Coles GoalEngine."""
    init_db(DATABASE_PATH)
    db_engine = get_engine(DATABASE_PATH)
    goal_engine = GoalEngine()

    with Session(db_engine) as session:
        try:
            goal_engine.fit_from_matches(session)
            logger.info(
                "GoalEngine fitted on startup: %d teams loaded.",
                len(goal_engine.teams),
            )
        except RuntimeError as exc:
            logger.warning(
                "GoalEngine could not be fitted at startup: %s "
                "— /predict will return 503 until data is loaded and /model/refit is called.",
                exc,
            )

    app.state.goal_engine = goal_engine
    yield
    # SQLite needs no explicit teardown.


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EdgeAI Trading Terminal API",
    version="1.0.0",
    description=(
        "Backend API for the EdgeAI sports betting trading terminal. "
        "Exposes prediction probabilities, bet history, and system controls."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_resolve_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# DB dependency — one session per request, closed automatically on teardown.
# Using a module-level engine so SQLAlchemy's connection pool is shared across
# requests rather than re-created on every call.
# ---------------------------------------------------------------------------

_db_engine = get_engine(DATABASE_PATH)


def get_db():
    """Request-scoped SQLAlchemy session (sync). Safe for lazy-loaded relationships."""
    with Session(_db_engine) as session:
        yield session


# ---------------------------------------------------------------------------
# Pydantic response / request schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    teams_fitted: int
    database: str


class UpcomingMatchOut(BaseModel):
    id: str
    date: datetime
    home_team: str
    away_team: str
    status: str
    sport_key: str | None

    model_config = ConfigDict(from_attributes=True)


class HistoryItem(BaseModel):
    """One settled prediction row, including denormalised match context."""

    id: int
    created_at: datetime
    # Match context (fetched via lazy-loaded relationship while session is open)
    match_date: datetime
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None
    sport_key: str | None
    # Prediction fields
    home_prob: float
    draw_prob: float
    away_prob: float
    recommended_selection: str | None
    recommended_stake_percent: float
    profit: float
    was_win: bool | None
    edge_used: float | None
    settled_at: datetime | None


class PredictRequest(BaseModel):
    home_team: str = Field(..., min_length=1, examples=["Arsenal"])
    away_team: str = Field(..., min_length=1, examples=["Chelsea"])
    kickoff: datetime | None = Field(
        default=None,
        description=(
            "ISO-8601 kickoff time in UTC. "
            "When provided, enables form / H2H / xG / injury feature adjustments."
        ),
        examples=["2026-04-26T15:00:00Z"],
    )


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    # 1X2 probabilities
    p_home: float
    p_draw: float
    p_away: float
    # Derived markets
    p_over_2_5: float
    p_btts: float
    # De-margined fair odds (1 / probability, capped at 999 for near-zero probs)
    fair_odds_home: float
    fair_odds_draw: float
    fair_odds_away: float
    # Full 6×6 score matrix for custom market construction in the UI
    score_matrix: list[list[float]]
    model_version: str
    features_applied: bool


class RefitResponse(BaseModel):
    teams_fitted: int
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Liveness check",
)
def health(request: Request) -> HealthResponse:
    """Returns server status, number of teams the model knows, and DB path."""
    ge: GoalEngine = request.app.state.goal_engine
    return HealthResponse(
        status="ok",
        teams_fitted=len(ge.teams),
        database=str(DATABASE_PATH),
    )


@app.get(
    "/upcoming",
    response_model=list[UpcomingMatchOut],
    tags=["Matches"],
    summary="List upcoming fixtures",
)
def upcoming_matches(
    horizon_days: int | None = None,
    db: Session = Depends(get_db),
) -> list[UpcomingMatchOut]:
    """
    All future, non-terminal matches ordered by kick-off time ascending.

    - **horizon_days**: optional upper bound (e.g. `7` = next 7 days only)
    """
    stmt = matches_for_prediction(horizon_days=horizon_days)
    rows: list[Match] = db.execute(stmt).scalars().all()
    return [
        UpcomingMatchOut(
            id=m.id,
            date=m.date,
            home_team=m.home_team,
            away_team=m.away_team,
            status=m.status,
            sport_key=m.sport_key,
        )
        for m in rows
    ]


@app.get(
    "/history",
    response_model=list[HistoryItem],
    tags=["History"],
    summary="Settled bet history (sorted)",
)
def bet_history(
    sort_by: BetSortField = BetSortField.DATE,
    sort_dir: SortDir = SortDir.DESC,
    settled_only: bool = True,
    db: Session = Depends(get_db),
) -> list[HistoryItem]:
    """
    Prediction history for the Trading Terminal history view.

    Sorting is fully delegated to SQLite — no in-memory sort, safe for large tables.

    - **sort_by**: `date` | `pnl` | `result`
    - **sort_dir**: `asc` | `desc`
    - **settled_only**: `false` to include pending/unsettled bets (live positions view)
    """
    stmt = get_bet_history(sort_by=sort_by, sort_dir=sort_dir, settled_only=settled_only)
    predictions = db.execute(stmt).scalars().all()

    # Build response while the session is still open so lazy-loaded .match works.
    items: list[HistoryItem] = []
    for p in predictions:
        m: Match = p.match
        items.append(
            HistoryItem(
                id=p.id,
                created_at=p.created_at,
                match_date=m.date,
                home_team=m.home_team,
                away_team=m.away_team,
                home_score=m.home_score,
                away_score=m.away_score,
                sport_key=m.sport_key,
                home_prob=p.home_prob,
                draw_prob=p.draw_prob,
                away_prob=p.away_prob,
                recommended_selection=p.recommended_selection,
                recommended_stake_percent=p.recommended_stake_percent,
                profit=p.profit,
                was_win=p.was_win,
                edge_used=p.edge_used,
                settled_at=p.settled_at,
            )
        )
    return items


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Predictions"],
    summary="Run Dixon-Coles prediction for a fixture",
)
def predict(
    body: PredictRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> PredictResponse:
    """
    Run the Dixon-Coles GoalEngine for a single fixture and return probabilities.

    **Model behaviour:**
    - Teams in the fitted model use MLE attack/defence parameters.
    - Teams *not* in the model fall back to Elo-derived λ (cold-start bridge).
    - Passing `kickoff` activates form, H2H, rest, xG regression, and injury
      penalty adjustments on top of the base λ.

    **Response includes:**
    - 1X2 probabilities and fair (de-margined) odds
    - Over 2.5 goals and Both Teams to Score probabilities
    - Full 6×6 score probability matrix for custom market construction
    """
    ge: GoalEngine = request.app.state.goal_engine
    if not ge.teams:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "GoalEngine has no fitted parameters — the database may be empty. "
                "Run a data backfill then call POST /model/refit."
            ),
        )

    features_applied = body.kickoff is not None
    matrix, p_home, p_draw, p_away, p_over_2_5, p_btts = ge.predict_match(
        home_team=body.home_team,
        away_team=body.away_team,
        session=db,
        kickoff=body.kickoff,
        use_features=features_applied,
    )

    def _fair_odds(prob: float) -> float:
        """1 / p, capped at 999.0 for near-zero probabilities."""
        return round(1.0 / prob, 4) if prob > 1e-6 else 999.0

    return PredictResponse(
        home_team=body.home_team,
        away_team=body.away_team,
        p_home=round(p_home, 6),
        p_draw=round(p_draw, 6),
        p_away=round(p_away, 6),
        p_over_2_5=round(p_over_2_5, 6),
        p_btts=round(p_btts, 6),
        fair_odds_home=_fair_odds(p_home),
        fair_odds_draw=_fair_odds(p_draw),
        fair_odds_away=_fair_odds(p_away),
        score_matrix=matrix.tolist(),
        model_version="dixon-coles-v1",
        features_applied=features_applied,
    )


@app.post(
    "/model/refit",
    response_model=RefitResponse,
    tags=["System"],
    summary="Re-fit GoalEngine from current DB data",
)
def refit_model(
    request: Request,
    db: Session = Depends(get_db),
) -> RefitResponse:
    """
    Re-fit the Dixon-Coles GoalEngine in-place without restarting the server.

    Call this after a data backfill or score settlement cycle to refresh
    model parameters. The updated engine is immediately live for subsequent
    `/predict` requests.
    """
    ge: GoalEngine = request.app.state.goal_engine
    try:
        ge.fit_from_matches(db)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return RefitResponse(
        teams_fitted=len(ge.teams),
        message=f"Model re-fitted on {len(ge.teams)} teams.",
    )
