from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import Base, Prediction
from app.database.session import engine, get_db


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "bike_demand_model.pkl"

FEATURE_COLUMNS = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]
MODEL_NAME = "bike_demand_hist_gradient_boosting_v1"

model: Any | None = None


class BikeDemandFeatures(BaseModel):
    season: int = Field(..., ge=1, le=4)
    yr: int = Field(..., ge=0, le=1)
    mnth: int = Field(..., ge=1, le=12)
    hr: int = Field(..., ge=0, le=23)
    holiday: int = Field(..., ge=0, le=1)
    weekday: int = Field(..., ge=0, le=6)
    workingday: int = Field(..., ge=0, le=1)
    weathersit: int = Field(..., ge=1, le=4)
    temp: float = Field(..., ge=0, le=1)
    atemp: float = Field(..., ge=0, le=1)
    hum: float = Field(..., ge=0, le=1)
    windspeed: float = Field(..., ge=0, le=1)


class PredictionRequest(BaseModel):
    features: list[BikeDemandFeatures] = Field(..., min_length=1)
    source: str = Field(default="webapp", pattern="^(webapp|scheduled)$")


class PredictionItem(BaseModel):
    prediction: float
    features: BikeDemandFeatures


class PredictionResponse(BaseModel):
    model_name: str
    predictions: list[PredictionItem]


class PastPredictionItem(BaseModel):
    id: int
    prediction: float
    source: str
    model_name: str
    features: dict[str, Any]
    created_at: datetime

    model_config = {"from_attributes": True}


class PastPredictionsResponse(BaseModel):
    predictions: list[PastPredictionItem]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Bike Sharing Demand API",
    description=(
        "Predict hourly bike rental demand from weather and calendar data."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )

    rows = [item.model_dump() for item in request.features]
    data = pd.DataFrame(rows, columns=FEATURE_COLUMNS)

    try:
        raw_predictions = model.predict(data)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {exc}",
        ) from exc

    predictions = []
    for prediction, features in zip(raw_predictions, request.features):
        prediction_value = float(prediction)
        predictions.append(
            PredictionItem(prediction=prediction_value, features=features)
        )
        db.add(
            Prediction(
                prediction=prediction_value,
                source=request.source,
                model_name=MODEL_NAME,
                features=features.model_dump(),
            )
        )

    db.commit()

    return PredictionResponse(
        model_name=MODEL_NAME,
        predictions=predictions,
    )


@app.get("/past-predictions", response_model=PastPredictionsResponse)
def past_predictions(
    source: str = Query(default="all", pattern="^(webapp|scheduled|all)$"),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> PastPredictionsResponse:
    query = select(Prediction)

    if source != "all":
        query = query.where(Prediction.source == source)
    if start_date is not None:
        query = query.where(Prediction.created_at >= start_date)
    if end_date is not None:
        query = query.where(Prediction.created_at <= end_date)

    query = query.order_by(Prediction.created_at.desc()).limit(limit)
    rows = db.execute(query).scalars().all()

    return PastPredictionsResponse(
        predictions=[PastPredictionItem.model_validate(row) for row in rows]
    )
