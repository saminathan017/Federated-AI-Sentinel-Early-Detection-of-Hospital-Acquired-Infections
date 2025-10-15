"""
Production predictor with calibration and uncertainty.

Loads trained model, calibrator, and provides unified interface for inference.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.explainability.counterfactuals import CounterfactualGenerator
from src.explainability.shap_explainer import SHAPExplainer
from src.modeling.temporal.calibrate import ModelCalibrator
from src.modeling.temporal.transformer_model import create_model
from src.modeling.temporal.uncertainty import UncertaintyEstimator


class InfectionPredictor:
    """Production-ready predictor with all inference capabilities."""

    def __init__(
        self,
        model_path: Path,
        calibrator_path: Path | None = None,
        device: str = "cpu",
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            calibrator_path: Path to calibrator (optional)
            device: Device to run on
        """
        self.device = device
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        
        self.model = create_model(input_dim=checkpoint["input_dim"], device=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        self.input_dim = checkpoint["input_dim"]
        
        # Load calibrator if available
        self.calibrator = None
        if calibrator_path and calibrator_path.exists():
            self.calibrator = ModelCalibrator.load(calibrator_path)
        
        # Initialize uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(method="mcdropout", n_samples=20)
        
        # Initialize explainers (lazy loading)
        self._shap_explainer = None
        self._counterfactual_generator = None

    def predict(self, features: np.ndarray) -> float:
        """
        Get infection risk prediction.
        
        Args:
            features: Patient features array
        
        Returns:
            Risk probability
        """
        self.model.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).unsqueeze(1)  # (1, 1, features)
            x = x.transpose(0, 1)  # (1, 1, features)
            
            logits = self.model(x)
            prob = torch.sigmoid(logits).item()
        
        # Apply calibration if available
        if self.calibrator:
            prob = self.calibrator.transform(np.array([prob]))[0]
        
        return prob

    def predict_with_uncertainty(self, features: np.ndarray) -> dict[str, Any]:
        """
        Predict with uncertainty estimates.
        
        Args:
            features: Patient features array
        
        Returns:
            Dictionary with prediction and uncertainty metrics
        """
        x = torch.FloatTensor(features).unsqueeze(0).unsqueeze(1)
        x = x.transpose(0, 1)
        
        result = self.uncertainty_estimator.predict_with_uncertainty(self.model, x)
        
        # Apply calibration if available
        if self.calibrator:
            result["prediction"] = self.calibrator.transform(
                np.array([result["prediction"]])
            )[0]
            
            if result.get("lower_95") is not None:
                result["lower_95"] = self.calibrator.transform(
                    np.array([result["lower_95"]])
                )[0]
            
            if result.get("upper_95") is not None:
                result["upper_95"] = self.calibrator.transform(
                    np.array([result["upper_95"]])
                )[0]
        
        return result

    def explain_prediction(
        self,
        features: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, Any]:
        """
        Generate SHAP explanation.
        
        Args:
            features: Patient features
            feature_names: Names of features
        
        Returns:
            Explanation dictionary
        """
        # Lazy load SHAP explainer
        if self._shap_explainer is None:
            # Use a small background dataset (would be loaded from disk in production)
            background = np.random.randn(50, self.input_dim)
            
            self._shap_explainer = SHAPExplainer(
                model=self.model,
                background_data=background,
                feature_names=feature_names,
                device=self.device,
            )
        
        explanation = self._shap_explainer.explain_instance(features, num_samples=50)
        
        return explanation

    def generate_counterfactual(
        self,
        features: np.ndarray,
        feature_names: list[str],
        target_prob: float = 0.3,
    ) -> dict[str, Any]:
        """
        Generate counterfactual explanation.
        
        Args:
            features: Patient features
            feature_names: Names of features
            target_prob: Target risk probability
        
        Returns:
            Counterfactual dictionary
        """
        # Lazy load counterfactual generator
        if self._counterfactual_generator is None:
            # Define plausible ranges (simplified)
            feature_ranges = {name: (0.0, 200.0) for name in feature_names}
            
            self._counterfactual_generator = CounterfactualGenerator(
                model=self.model,
                feature_names=feature_names,
                feature_ranges=feature_ranges,
                device=self.device,
            )
        
        cf = self._counterfactual_generator.generate_counterfactual(
            original_instance=features,
            target_prob=target_prob,
            max_iterations=50,
        )
        
        return cf


def main() -> None:
    """Test predictor."""
    print("Testing production predictor...")
    
    model_path = Path("models/temporal/best_model.pt")
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run 'make train_temporal' first")
        return
    
    # Initialize predictor
    predictor = InfectionPredictor(model_path=model_path)
    
    # Generate random test input
    test_features = np.random.randn(predictor.input_dim)
    
    # Test prediction
    risk = predictor.predict(test_features)
    print(f"\nRisk prediction: {risk:.3f}")
    
    # Test with uncertainty
    result = predictor.predict_with_uncertainty(test_features)
    print(f"Prediction with uncertainty: {result['prediction']:.3f} ± {result['uncertainty']:.3f}")
    print(f"95% CI: [{result['lower_95']:.3f}, {result['upper_95']:.3f}]")
    
    print("\n✓ Predictor test passed")


if __name__ == "__main__":
    main()

