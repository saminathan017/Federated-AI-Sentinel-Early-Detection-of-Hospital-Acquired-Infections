"""
Explainability endpoints.

Provides SHAP explanations and counterfactual what-if scenarios.
"""

import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.serving.api.routers.score import PatientFeatures

router = APIRouter()
logger = logging.getLogger(__name__)


class FeatureContribution(BaseModel):
    """Feature contribution to the prediction."""

    feature_name: str
    feature_value: float
    contribution: float
    impact: str  # "increases risk" or "decreases risk"


class ExplanationResponse(BaseModel):
    """SHAP explanation response."""

    patient_id: str
    encounter_id: str
    prediction: float
    top_contributors: list[FeatureContribution]
    explanation_text: str


class CounterfactualChange(BaseModel):
    """Suggested change to reduce risk."""

    feature_name: str
    current_value: float
    target_value: float
    change: float
    percent_change: float


class CounterfactualResponse(BaseModel):
    """Counterfactual what-if response."""

    patient_id: str
    encounter_id: str
    current_risk: float
    target_risk: float
    achievable_risk: float
    success: bool
    suggested_changes: list[CounterfactualChange]
    action_plan: str


@router.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(
    features: PatientFeatures,
    request: Request,
):
    """
    Generate SHAP explanation for a prediction.
    
    Shows which clinical features drive the infection risk score.
    """
    try:
        predictor = request.app.state.get_predictor()
        
        # Convert features to array
        feature_dict = features.dict(exclude={"patient_id", "encounter_id"})
        feature_names = list(feature_dict.keys())
        feature_array = np.array([v if v is not None else 0.0 for v in feature_dict.values()])
        
        # Get explanation
        explanation = predictor.explain_prediction(feature_array, feature_names)
        
        # Format top contributors
        top_contributors = []
        
        for driver in explanation.get("top_drivers", [])[:5]:
            contrib = FeatureContribution(
                feature_name=driver["feature"].replace("_", " ").title(),
                feature_value=driver["value"],
                contribution=abs(driver["shap_value"]),
                impact=driver["impact"],
            )
            top_contributors.append(contrib)
        
        # Generate explanation text
        text = f"Infection risk: {explanation['prediction']:.1%}\n\n"
        text += "Key clinical drivers:\n"
        
        for i, contrib in enumerate(top_contributors, 1):
            text += (
                f"{i}. {contrib.feature_name} = {contrib.feature_value:.2f} "
                f"→ {contrib.impact}\n"
            )
        
        return ExplanationResponse(
            patient_id=features.patient_id,
            encounter_id=features.encounter_id,
            prediction=explanation["prediction"],
            top_contributors=top_contributors,
            explanation_text=text,
        )
    
    except Exception as e:
        logger.error(f"Explanation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.post("/counterfactual", response_model=CounterfactualResponse)
async def generate_counterfactual(
    features: PatientFeatures,
    target_risk: float = 0.3,
    request: Request = None,
):
    """
    Generate counterfactual what-if scenario.
    
    Shows what changes would be needed to reduce infection risk.
    """
    try:
        predictor = request.app.state.get_predictor()
        
        # Convert features
        feature_dict = features.dict(exclude={"patient_id", "encounter_id"})
        feature_names = list(feature_dict.keys())
        feature_array = np.array([v if v is not None else 0.0 for v in feature_dict.values()])
        
        # Generate counterfactual
        cf = predictor.generate_counterfactual(
            feature_array,
            feature_names,
            target_prob=target_risk,
        )
        
        # Format changes
        suggested_changes = []
        
        for change in cf.get("changes", [])[:5]:
            suggested_changes.append(
                CounterfactualChange(
                    feature_name=change["feature"].replace("_", " ").title(),
                    current_value=change["original_value"],
                    target_value=change["target_value"],
                    change=change["change"],
                    percent_change=change["percent_change"],
                )
            )
        
        # Generate action plan
        action_plan = f"Current risk: {cf['original_prediction']:.1%}\n"
        action_plan += f"Target risk: {target_risk:.1%}\n"
        action_plan += f"Achievable risk: {cf['counterfactual_prediction']:.1%}\n\n"
        
        if cf["success"]:
            action_plan += "Recommended interventions:\n"
        else:
            action_plan += "Suggested interventions (target not fully achievable):\n"
        
        for i, change in enumerate(suggested_changes, 1):
            direction = "increase" if change.change > 0 else "decrease"
            action_plan += (
                f"{i}. {direction.capitalize()} {change.feature_name} "
                f"from {change.current_value:.2f} to {change.target_value:.2f}\n"
            )
        
        return CounterfactualResponse(
            patient_id=features.patient_id,
            encounter_id=features.encounter_id,
            current_risk=cf["original_prediction"],
            target_risk=target_risk,
            achievable_risk=cf["counterfactual_prediction"],
            success=cf["success"],
            suggested_changes=suggested_changes,
            action_plan=action_plan,
        )
    
    except Exception as e:
        logger.error(f"Counterfactual generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Counterfactual generation failed: {str(e)}",
        )

