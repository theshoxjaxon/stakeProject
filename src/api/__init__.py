"""FastAPI web server for the EdgeAI sports betting trading terminal."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.deps import db_engine, get_db, get_redis
from src.api.routers import auth as auth_router
from src.api.routers import org as org_router
from src.api.routers import predictions as predictions_router
from src.api.serializers import MODEL_VERSION
from src.match_queries import (
    BetSortField,
    SortDir,
    get_bet_history,
    matches_for_prediction,
)
from src.models import Match
from src.poisson_model import GoalEngine
from src.tournaments import get_active_tournaments

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CORS — exactly one dashboard origin, credentials allowed. Never a wildcard
# or a list here: allow_credentials=True with allow_origins=["*"] leaks
# cross-origin credentialed requests to any site (browsers reject the literal
# "*" combination, but there's no reason to rely on that as the only guard).
# ---------------------------------------------------------------------------

DASHBOARD_ORIGIN = os.getenv("DASHBOARD_ORIGIN", "http://localhost:5173").strip()
if DASHBOARD_ORIGIN == "*" or "," in DASHBOARD_ORIGIN:
    raise RuntimeError(
        "DASHBOARD_ORIGIN must be a single, exact origin — never '*' or a "
        "comma-separated list — because CORS is configured with "
        "allow_credentials=True."
    )


# ---------------------------------------------------------------------------
# Lifespan — fit GoalEngine exactly once at startup, reuse across all requests.
# If the DB is empty, the engine starts unfitted and /predict returns 503 until
# a backfill + /model/refit is called.
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Fit the Dixon-Coles GoalEngine at startup.

    Does NOT create/migrate schema — that's alembic's job, run as an
    explicit deploy step (`alembic upgrade head`) before the app starts.
    Doing it here raced with multiple uvicorn workers, and Base.metadata's
    create_all() doesn't know to CREATE EXTENSION citext first, so it fails
    outright on a fresh DB that hasn't been migrated yet.
    """
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
    allow_origins=[DASHBOARD_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router, prefix="/v1/auth", tags=["Auth"])
app.include_router(predictions_router.router, prefix="/v1/predictions", tags=["Predictions"])
app.include_router(org_router.router, prefix="/v1/org", tags=["Organization"])

# DB dependency: get_db / db_engine live in src.api.deps (imported above) so
# that module stays the single owner of the pooled engine — deps never
# imports from this package's __init__, only the other way around.

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


class TournamentOut(BaseModel):
    key: str
    title: str


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
    """Returns server status, number of teams the model knows, and DB target (no credentials)."""
    ge: GoalEngine = request.app.state.goal_engine
    url = db_engine.url
    return HealthResponse(
        status="ok",
        teams_fitted=len(ge.teams),
        database=f"{url.get_backend_name()}://{url.host or 'localhost'}:{url.port or 5432}/{url.database}",
    )


@app.get("/healthz", tags=["System"], summary="Liveness probe")
def healthz() -> dict:
    """
    Process-is-alive check. Deliberately checks nothing else: a liveness
    probe that depends on the DB/Redis being reachable causes an orchestrator
    to kill and restart a perfectly healthy process during a downstream
    outage. That's what /readyz is for.
    """
    return {"status": "ok"}


@app.get("/readyz", tags=["System"], summary="Readiness probe (DB + Redis)")
def readyz(
    db: Session = Depends(get_db),
    redis_client=Depends(get_redis),
) -> JSONResponse:
    """
    Checks the DB and Redis are actually reachable. 503 if either is down.

    Goes through the same Depends(get_db) / Depends(get_redis) everything
    else uses — not a raw db_engine.connect() — so it's exercising the
    actual pooled clients the app serves requests with, not a side channel.
    """
    checks: dict[str, str] = {}

    try:
        db.execute(text("SELECT 1"))
        checks["db"] = "ok"
    except Exception as exc:  # noqa: BLE001 — a readiness check must never itself 500
        checks["db"] = f"error: {exc}"

    try:
        redis_client.ping()
        checks["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["redis"] = f"error: {exc}"

    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(status_code=200 if all_ok else 503, content=checks)


@app.get(
    "/tournaments/active",
    response_model=list[TournamentOut],
    tags=["Matches"],
    summary="Tournaments currently in season",
)
def active_tournaments() -> list[TournamentOut]:
    """
    Candidate tournaments (World Cup, Euros, UCL, Europa League, Copa América)
    that The Odds API currently marks as in season. The frontend can use this
    to render competition tabs; results are cached server-side for
    SPORTS_CACHE_TTL_HOURS.
    """
    return [TournamentOut(**t) for t in get_active_tournaments()]


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
        model_version=MODEL_VERSION,
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
