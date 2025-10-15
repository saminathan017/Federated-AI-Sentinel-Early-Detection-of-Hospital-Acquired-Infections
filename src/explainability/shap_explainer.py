"""
SHAP-based explanations for infection predictions.

Provides both global feature importance and local per-case explanations.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn


class SHAPExplainer:
    """Generate SHAP explanations for temporal transformer models."""

    def __init__(
        self,
        model: nn.Module,
        background_data: np.ndarray,
        feature_names: list[str],
        device: str = "cpu",
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model
            background_data: Background dataset for SHAP (subset of training data)
            feature_names: Names of input features
            device: Device to run on
        """
        self.model = model
        self.feature_names = feature_names
        self.device = device
        
        # Wrap model for SHAP
        def model_predict(x: np.ndarray) -> np.ndarray:
            """Wrapper for model prediction."""
            self.model.eval()
            
            with torch.no_grad():
                # Add sequence dimension if needed
                if len(x.shape) == 2:
                    x_tensor = torch.FloatTensor(x).unsqueeze(1)  # (batch, 1, features)
                else:
                    x_tensor = torch.FloatTensor(x)
                
                x_tensor = x_tensor.transpose(0, 1)  # (seq, batch, features)
                
                logits = self.model(x_tensor)
                probs = torch.sigmoid(logits)
                
                return probs.cpu().numpy()
        
        self.model_predict = model_predict
        
        # Create SHAP explainer
        # Use KernelExplainer for model-agnostic explanations
        self.explainer = shap.KernelExplainer(
            model=self.model_predict,
            data=background_data[:100],  # Use subset for efficiency
        )

    def explain_instance(
        self,
        instance: np.ndarray,
        num_samples: int = 100,
    ) -> dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            instance: Input features for one patient
            num_samples: Number of samples for SHAP estimation
        
        Returns:
            Dictionary with SHAP values and top drivers
        """
        # Compute SHAP values
        shap_values = self.explainer.shap_values(
            instance.reshape(1, -1),
            nsamples=num_samples,
        )
        
        # Get base value (expected value)
        base_value = self.explainer.expected_value
        
        # Get prediction
        pred = self.model_predict(instance.reshape(1, -1))[0]
        
        # Create explanation dictionary
        explanation = {
            "prediction": float(pred),
            "base_value": float(base_value),
            "shap_values": shap_values[0],
            "feature_values": instance,
            "feature_names": self.feature_names,
        }
        
        # Get top positive and negative drivers
        shap_abs = np.abs(shap_values[0])
        top_indices = np.argsort(shap_abs)[::-1][:5]
        
        top_drivers = []
        
        for idx in top_indices:
            driver = {
                "feature": self.feature_names[idx],
                "value": float(instance[idx]),
                "shap_value": float(shap_values[0][idx]),
                "impact": "increases risk" if shap_values[0][idx] > 0 else "decreases risk",
            }
            top_drivers.append(driver)
        
        explanation["top_drivers"] = top_drivers
        
        return explanation

    def global_importance(self, X: np.ndarray, num_samples: int = 100) -> pd.DataFrame:
        """
        Compute global feature importance.
        
        Args:
            X: Sample of input data
            num_samples: Number of samples for SHAP
        
        Returns:
            DataFrame with feature importance scores
        """
        shap_values = self.explainer.shap_values(X, nsamples=num_samples)
        
        # Compute mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": mean_abs_shap,
        })
        
        importance_df = importance_df.sort_values("importance", ascending=False)
        
        return importance_df


def format_explanation_for_clinician(explanation: dict[str, Any]) -> str:
    """
    Format SHAP explanation in human-readable text.
    
    Args:
        explanation: SHAP explanation dictionary
    
    Returns:
        Human-readable explanation string
    """
    pred = explanation["prediction"]
    risk_level = "HIGH" if pred > 0.5 else "MODERATE" if pred > 0.2 else "LOW"
    
    text = f"Infection Risk: {risk_level} ({pred:.1%})\n\n"
    text += "Top Clinical Drivers:\n"
    
    for i, driver in enumerate(explanation["top_drivers"], 1):
        feature = driver["feature"].replace("_", " ").title()
        value = driver["value"]
        impact = driver["impact"]
        
        text += f"{i}. {feature} = {value:.2f} → {impact}\n"
    
    return text


def main() -> None:
    """Example: generate SHAP explanations."""
    from src.modeling.temporal.train import load_data
    from src.modeling.temporal.transformer_model import create_model
    
    print("Generating SHAP explanations...")
    
    # Load data
    data_dir = Path("data/labeled")
    X, y = load_data(data_dir)
    
    print(f"Loaded {len(X)} samples")
    
    # Load trained model
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
    sample_file = data_dir / "site_a_labeled.parquet"
    df = pd.read_parquet(sample_file)
    feature_names = [
        col
        for col in df.columns
        if col not in ["patient_id", "encounter_id", "window_end_time", "infection_label", "hours_to_infection"]
    ]
    
    # Create SHAP explainer
    explainer = SHAPExplainer(
        model=model,
        background_data=X[:200],  # Use subset as background
        feature_names=feature_names,
        device="cpu",
    )
    
    # Explain a few instances
    print("\n" + "=" * 70)
    print("SHAP EXPLANATIONS FOR SAMPLE CASES")
    print("=" * 70)
    
    for i in [0, 10, 20]:
        explanation = explainer.explain_instance(X[i], num_samples=50)
        
        print(f"\nCase {i+1}:")
        print(format_explanation_for_clinician(explanation))
    
    # Global importance
    print("\n" + "=" * 70)
    print("GLOBAL FEATURE IMPORTANCE")
    print("=" * 70)
    
    importance_df = explainer.global_importance(X[:100], num_samples=50)
    print(importance_df.head(10).to_string(index=False))
    
    print("\n✓ SHAP explanations generated")


if __name__ == "__main__":
    main()

