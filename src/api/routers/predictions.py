"""Predictions router: today's fixtures, single-prediction lookup.

Mounted at /v1/predictions by src/api/__init__.py.

Tier gating is entirely inside serialize_prediction (src/api/serializers.py)
— these routes never branch on plan themselves, only fetch rows and hand
them to the serializer. Every authenticated caller, on any plan, can call
these endpoints; what differs is which fields come back.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from src.api.deps import Principal, enforce_daily_rate_limit, get_current_plan, get_db
from src.api.serializers import serialize_prediction
from src.models import Prediction

router = APIRouter()


@router.get("/today")
def list_todays_predictions(
    principal: Principal = Depends(enforce_daily_rate_limit),
    plan: str = Depends(get_current_plan),
    db: Session = Depends(get_db),
) -> list[dict]:
    """
    Predictions for fixtures kicking off in the next 24h.

    "Today" is a rolling 24h window from now, not a calendar-day boundary —
    that matches what the prediction pipeline actually produces on each run
    (see main.py), rather than an arbitrary UTC midnight cutoff.
    """
    now = datetime.now(timezone.utc)
    stmt = (
        select(Prediction)
        .options(selectinload(Prediction.match))
        .where(
            Prediction.kickoff >= now, Prediction.kickoff < now + timedelta(hours=24)
        )
        .order_by(Prediction.kickoff.asc())
    )
    predictions = db.execute(stmt).scalars().all()
    return [serialize_prediction(p, plan) for p in predictions]


@router.get("/{prediction_id}")
def get_prediction(
    prediction_id: int,
    principal: Principal = Depends(enforce_daily_rate_limit),
    plan: str = Depends(get_current_plan),
    db: Session = Depends(get_db),
) -> dict:
    prediction = db.execute(
        select(Prediction)
        .options(selectinload(Prediction.match))
        .where(Prediction.id == prediction_id)
    ).scalar_one_or_none()
    if prediction is None:
        raise HTTPException(status_code=404, detail={"error": "not_found"})
    return serialize_prediction(prediction, plan)
