"""
Generate counterfactual explanations.

Show what changes to patient vitals or labs would reduce infection risk.
Respects clinical plausibility constraints.
"""

from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize


class CounterfactualGenerator:
    """Generate clinically plausible counterfactuals for risk reduction."""

    def __init__(
        self,
        model: nn.Module,
        feature_names: list[str],
        feature_ranges: dict[str, tuple[float, float]],
        device: str = "cpu",
    ):
        """
        Initialize counterfactual generator.
        
        Args:
            model: Trained prediction model
            feature_names: Names of input features
            feature_ranges: Valid ranges for each feature (min, max)
            device: Device to run on
        """
        self.model = model
        self.feature_names = feature_names
        self.feature_ranges = feature_ranges
        self.device = device

    def predict(self, x: np.ndarray) -> float:
        """Get model prediction for input."""
        self.model.eval()
        
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).unsqueeze(0).unsqueeze(1)  # (1, 1, features)
            x_tensor = x_tensor.transpose(0, 1)  # (1, 1, features)
            
            logits = self.model(x_tensor)
            prob = torch.sigmoid(logits).item()
        
        return prob

    def generate_counterfactual(
        self,
        original_instance: np.ndarray,
        target_prob: float = 0.3,
        max_iterations: int = 100,
        change_penalty: float = 1.0,
    ) -> dict[str, Any]:
        """
        Find a counterfactual that reduces risk to target probability.
        
        Uses constrained optimization to find minimal changes that achieve
        the target risk level while respecting clinical bounds.
        
        Args:
            original_instance: Original patient features
            target_prob: Target risk probability
            max_iterations: Maximum optimization iterations
            change_penalty: Weight for penalizing large changes
        
        Returns:
            Dictionary with counterfactual and recommended changes
        """
        original_pred = self.predict(original_instance)
        
        if original_pred <= target_prob:
            # Already below target
            return {
                "original_prediction": original_pred,
                "counterfactual_prediction": original_pred,
                "changes": [],
                "message": "Patient already below target risk threshold",
            }
        
        # Objective: minimize distance + prediction mismatch
        def objective(x: np.ndarray) -> float:
            # Prediction loss
            pred = self.predict(x)
            pred_loss = (pred - target_prob) ** 2
            
            # Distance from original (L2)
            distance = np.sum((x - original_instance) ** 2)
            
            return pred_loss + change_penalty * distance
        
        # Bounds: respect feature ranges
        bounds = []
        
        for i, feature in enumerate(self.feature_names):
            if feature in self.feature_ranges:
                bounds.append(self.feature_ranges[feature])
            else:
                # Default: allow ±20% change
                val = original_instance[i]
                bounds.append((val * 0.8, val * 1.2))
        
        # Optimize
        result = minimize(
            objective,
            x0=original_instance,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iterations},
        )
        
        counterfactual = result.x
        counterfactual_pred = self.predict(counterfactual)
        
        # Identify changes
        changes = []
        
        for i, feature in enumerate(self.feature_names):
            original_val = original_instance[i]
            new_val = counterfactual[i]
            
            if abs(new_val - original_val) > 1e-3:  # Significant change
                change = {
                    "feature": feature,
                    "original_value": float(original_val),
                    "target_value": float(new_val),
                    "change": float(new_val - original_val),
                    "percent_change": float((new_val - original_val) / original_val * 100)
                    if original_val != 0
                    else 0.0,
                }
                changes.append(change)
        
        # Sort by magnitude of change
        changes = sorted(changes, key=lambda x: abs(x["change"]), reverse=True)
        
        return {
            "original_prediction": float(original_pred),
            "counterfactual_prediction": float(counterfactual_pred),
            "target_prediction": target_prob,
            "success": counterfactual_pred <= target_prob,
            "changes": changes[:5],  # Top 5 changes
        }


def format_counterfactual_for_clinician(cf: dict[str, Any]) -> str:
    """
    Format counterfactual explanation for clinical use.
    
    Args:
        cf: Counterfactual dictionary
    
    Returns:
        Human-readable action plan
    """
    if not cf["changes"]:
        return cf.get("message", "No changes recommended")
    
    text = f"Current Risk: {cf['original_prediction']:.1%}\n"
    text += f"Target Risk: {cf['target_prediction']:.1%}\n"
    text += f"Achievable Risk: {cf['counterfactual_prediction']:.1%}\n\n"
    
    if cf["success"]:
        text += "Recommended Interventions:\n"
    else:
        text += "Suggested Interventions (target not fully achievable):\n"
    
    for i, change in enumerate(cf["changes"], 1):
        feature = change["feature"].replace("_", " ").title()
        orig = change["original_value"]
        target = change["target_value"]
        
        direction = "increase" if change["change"] > 0 else "decrease"
        
        text += f"{i}. {direction.capitalize()} {feature} from {orig:.2f} to {target:.2f}\n"
    
    return text


def main() -> None:
    """Example: generate counterfactual explanations."""
    from pathlib import Path
    
    from src.modeling.temporal.train import load_data
    from src.modeling.temporal.transformer_model import create_model
    
    print("Generating counterfactual explanations...")
    
    # Load data
    data_dir = Path("data/labeled")
    X, y = load_data(data_dir)
    
    # Load model
    model_path = Path("models/temporal/best_model.pt")
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run 'make train_temporal' first")
        return
    
    checkpoint = torch.load(model_path, map_location="cpu")
    
    model = create_model(input_dim=checkpoint["input_dim"], device="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Get feature names
    import pandas as pd
    
    sample_file = data_dir / "site_a_labeled.parquet"
    df = pd.read_parquet(sample_file)
    feature_names = [
        col
        for col in df.columns
        if col not in ["patient_id", "encounter_id", "window_end_time", "infection_label", "hours_to_infection"]
    ]
    
    # Define clinically plausible ranges
    feature_ranges = {
        "heart_rate_mean": (40, 160),
        "heart_rate_max": (50, 180),
        "temperature_mean": (35.0, 39.5),
        "temperature_max": (35.5, 40.0),
        "wbc_count_latest": (2.0, 25.0),
        "c_reactive_protein_latest": (0, 200),
        "lactate_latest": (0.5, 8.0),
    }
    
    # Create generator
    generator = CounterfactualGenerator(
        model=model,
        feature_names=feature_names,
        feature_ranges=feature_ranges,
        device="cpu",
    )
    
    # Find high-risk cases
    predictions = [generator.predict(X[i]) for i in range(min(100, len(X)))]
    high_risk_indices = [i for i, p in enumerate(predictions) if p > 0.5]
    
    if not high_risk_indices:
        print("No high-risk cases found in sample")
        return
    
    print("\n" + "=" * 70)
    print("COUNTERFACTUAL EXPLANATIONS FOR HIGH-RISK CASES")
    print("=" * 70)
    
    # Generate counterfactuals for a few high-risk cases
    for idx in high_risk_indices[:3]:
        cf = generator.generate_counterfactual(
            original_instance=X[idx],
            target_prob=0.3,
            max_iterations=50,
        )
        
        print(f"\nCase {idx+1}:")
        print(format_counterfactual_for_clinician(cf))
    
    print("\n✓ Counterfactual explanations generated")


if __name__ == "__main__":
    main()

