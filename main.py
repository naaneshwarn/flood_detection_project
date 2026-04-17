from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🌊 Flood Risk Forecasting API",
    description="India Flood Risk Prediction API with City and Date support.",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model artifacts ───────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent

try:
    model      = joblib.load(BASE / "flood_model.pkl")
    le_cause   = joblib.load(BASE / "le_cause.pkl")
    le_season  = joblib.load(BASE / "le_season.pkl")
    le_state   = joblib.load(BASE / "le_state.pkl")
    le_durbin  = joblib.load(BASE / "le_durbin.pkl")
    le_target  = joblib.load(BASE / "le_target.pkl")

    with open(BASE / "model_meta.json") as f:
        META = json.load(f)
except Exception as e:
    print(f"❌ Error loading artifacts: {e}. Ensure train_and_save.py was run!")

# ── Helpers ───────────────────────────────────────────────────────────────────
SEASON_MAP = {
    1: "winter", 2: "winter", 3: "pre_monsoon", 4: "pre_monsoon", 5: "pre_monsoon",
    6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "post_monsoon", 11: "post_monsoon", 12: "winter",
}

RISK_ADVICE = {
    "Low": {"color": "#22c55e", "emoji": "🟢", "advice": "Normal conditions. Routine monitoring recommended.", 
            "actions": ["Monitor local weather forecasts", "Keep emergency contacts ready"]},
    "Medium": {"color": "#f59e0b", "emoji": "🟡", "advice": "Moderate flood risk. Precautionary measures advised.", 
               "actions": ["Prepare emergency kits", "Identify nearest high-ground areas"]},
    "High": {"color": "#ef4444", "emoji": "🔴", "advice": "High flood risk. Immediate preparedness required.", 
             "actions": ["Pack essentials and move to safety", "Follow local evacuation orders"]},
    "Extreme": {"color": "#7c3aed", "emoji": "🟣", "advice": "Extreme flood risk. Emergency protocols must be activated.", 
                "actions": ["IMMEDIATE EVACUATION REQUIRED", "Contact emergency rescue teams"]},
}

def duration_to_bin(days: int) -> str:
    if days <= 1:   return "single_day"
    if days <= 3:   return "2_3days"
    if days <= 7:   return "week"
    if days <= 14:  return "2weeks"
    return "prolonged"

def safe_encode(le, value, default_idx=0):
    try:
        # We convert to string to ensure the encoder can process it
        return int(le.transform([str(value)])[0])
    except ValueError:
        return default_idx

def build_features(month, year, duration, cause, state):
    season = SEASON_MAP.get(month, "monsoon")
    dur_bin = duration_to_bin(duration)
    
    # These indices MUST match the order used in train_and_save.py
    return np.array([[
        month, year, duration, 
        safe_encode(le_cause, cause), 
        safe_encode(le_season, season), 
        safe_encode(le_state, state), 
        safe_encode(le_durbin, dur_bin)
    ]])

# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Date in YYYY-MM-DD")
    city: str = Field(..., description="Name of the City")
    state: str = Field(..., description="Name of the State")
    duration_days: int = Field(..., ge=0, le=365)
    cause: str = Field(..., description="Cause of flood (e.g. Heavy Rains)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "date": "2026-08-15",
                "city": "Bengaluru",
                "state": "Karnataka",
                "duration_days": 3,
                "cause": "Heavy Rains"
            }
        }
    )

class PredictResponse(BaseModel):
    city: str
    risk_level: str
    confidence: float
    probabilities: Dict[str, float]
    color: str
    emoji: str
    advice: str
    recommended_actions: List[str]

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "API is online", "docs": "/docs"}

@app.post("/predict", response_model=PredictResponse)
def predict_flood_risk(req: PredictRequest):
    # 1. Parse Date to get Month and Year for the model
    try:
        date_obj = datetime.strptime(req.date, "%Y-%m-%d")
        month = date_obj.month
        year = date_obj.year
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    # 2. Build features for the model
    features = build_features(month, year, req.duration_days, req.cause, req.state)
    
    # 3. Model Inference
    proba = model.predict_proba(features)[0]
    pred_idx = int(np.argmax(proba))
    
    risk_label = le_target.classes_[pred_idx]
    advice_info = RISK_ADVICE.get(risk_label, RISK_ADVICE["Low"])
    
    return PredictResponse(
        city=req.city,
        risk_level=risk_label,
        confidence=round(float(proba[pred_idx]) * 100, 2),
        probabilities={cls: round(float(p) * 100, 2) for cls, p in zip(le_target.classes_, proba)},
        color=advice_info["color"],
        emoji=advice_info["emoji"],
        advice=advice_info["advice"],
        recommended_actions=advice_info["actions"]
    )

@app.get("/states")
def get_supported_states():
    return {"states": sorted(list(le_state.classes_))}