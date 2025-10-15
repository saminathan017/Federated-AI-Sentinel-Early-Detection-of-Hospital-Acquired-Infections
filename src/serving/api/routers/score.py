"""
Infection risk scoring endpoints.

Provides real-time risk predictions with calibrated probabilities and uncertainty.
"""

import logging
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.serving.api.main import prediction_counter, prediction_latency

router = APIRouter()
logger = logging.getLogger(__name__)


class PatientFeatures(BaseModel):
    """Input features for a patient at a specific time window."""

    patient_id: str = Field(..., description="Unique patient identifier")
    encounter_id: str = Field(..., description="Current encounter/admission ID")
    
    # Vital signs
    heart_rate_mean: float | None = None
    heart_rate_max: float | None = None
    heart_rate_min: float | None = None
    heart_rate_std: float | None = None
    
    respiratory_rate_mean: float | None = None
    respiratory_rate_max: float | None = None
    respiratory_rate_min: float | None = None
    respiratory_rate_std: float | None = None
    
    temperature_mean: float | None = None
    temperature_max: float | None = None
    temperature_min: float | None = None
    temperature_std: float | None = None
    
    oxygen_saturation_mean: float | None = None
    oxygen_saturation_max: float | None = None
    oxygen_saturation_min: float | None = None
    oxygen_saturation_std: float | None = None
    
    # Lab results
    wbc_count_latest: float | None = None
    wbc_count_delta: float | None = None
    c_reactive_protein_latest: float | None = None
    c_reactive_protein_delta: float | None = None
    procalcitonin_latest: float | None = None
    procalcitonin_delta: float | None = None
    lactate_latest: float | None = None
    lactate_delta: float | None = None
    
    # Metadata
    hours_since_admission: float | None = None
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "P000123",
                "encounter_id": "E001",
                "heart_rate_mean": 95.5,
                "heart_rate_max": 110.0,
                "heart_rate_min": 82.0,
                "heart_rate_std": 8.2,
                "temperature_mean": 38.2,
                "temperature_max": 38.7,
                "temperature_min": 37.8,
                "wbc_count_latest": 14.5,
                "c_reactive_protein_latest": 85.3,
                "procalcitonin_latest": 1.8,
                "lactate_latest": 2.1,
                "hours_since_admission": 72.0,
            }
        }


class RiskPrediction(BaseModel):
    """Risk prediction response."""

    patient_id: str
    encounter_id: str
    infection_risk_score: float = Field(..., ge=0.0, le=1.0, description="Probability of infection in next 24h")
    risk_level: str = Field(..., description="HIGH, MODERATE, or LOW")
    uncertainty: float | None = Field(None, description="Prediction uncertainty (std dev)")
    confidence_interval_lower: float | None = None
    confidence_interval_upper: float | None = None
    recommended_action: str = Field(..., description="Clinical recommendation")
    timestamp: str


@router.post("/score", response_model=RiskPrediction)
async def predict_infection_risk(
    features: PatientFeatures,
    request: Request,
):
    """
    Predict infection risk for a patient.
    
    Returns calibrated probability with uncertainty estimates.
    """
    start_time = time.time()
    
    try:
        # Get predictor from app state
        predictor = request.app.state.get_predictor()
        
        # Convert features to array
        feature_dict = features.dict(exclude={"patient_id", "encounter_id"})
        feature_array = np.array([v if v is not None else 0.0 for v in feature_dict.values()])
        
        # Get prediction
        result = predictor.predict_with_uncertainty(feature_array)
        
        risk_score = result["prediction"]
        uncertainty = result.get("uncertainty", None)
        
        # Determine risk level
        if risk_score >= 0.5:
            risk_level = "HIGH"
            action = "Consider infection workup. Review vitals and labs. Consult infectious disease if indicated."
        elif risk_score >= 0.2:
            risk_level = "MODERATE"
            action = "Monitor closely. Reassess in 4-6 hours. Maintain infection control precautions."
        else:
            risk_level = "LOW"
            action = "Continue routine monitoring. Standard infection prevention measures."
        
        # Log prediction
        prediction_counter.labels(outcome=risk_level).inc()
        
        from datetime import datetime
        
        response = RiskPrediction(
            patient_id=features.patient_id,
            encounter_id=features.encounter_id,
            infection_risk_score=float(risk_score),
            risk_level=risk_level,
            uncertainty=float(uncertainty) if uncertainty is not None else None,
            confidence_interval_lower=result.get("lower_95"),
            confidence_interval_upper=result.get("upper_95"),
            recommended_action=action,
            timestamp=datetime.now().isoformat(),
        )
        
        # Record latency
        duration = time.time() - start_time
        prediction_latency.observe(duration)
        
        logger.info(
            f"Prediction for {features.patient_id}: risk={risk_score:.3f}, "
            f"level={risk_level}, latency={duration:.3f}s"
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/batch_score")
async def batch_predict(
    patients: list[PatientFeatures],
    request: Request,
):
    """
    Batch prediction for multiple patients.
    
    More efficient for scoring many patients at once.
    """
    try:
        predictor = request.app.state.get_predictor()
        
        results = []
        
        for patient in patients:
            feature_dict = patient.dict(exclude={"patient_id", "encounter_id"})
            feature_array = np.array([v if v is not None else 0.0 for v in feature_dict.values()])
            
            result = predictor.predict_with_uncertainty(feature_array)
            
            risk_score = result["prediction"]
            
            if risk_score >= 0.5:
                risk_level = "HIGH"
            elif risk_score >= 0.2:
                risk_level = "MODERATE"
            else:
                risk_level = "LOW"
            
            results.append({
                "patient_id": patient.patient_id,
                "encounter_id": patient.encounter_id,
                "infection_risk_score": float(risk_score),
                "risk_level": risk_level,
            })
        
        return {"predictions": results, "count": len(results)}
    
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

