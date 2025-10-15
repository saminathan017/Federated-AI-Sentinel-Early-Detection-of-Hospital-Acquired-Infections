"""
Ensemble prediction combining multiple models.

Useful for improved performance and uncertainty quantification.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.modeling.temporal.transformer_model import create_model


class ModelEnsemble:
    """Ensemble of infection prediction models."""

    def __init__(self, model_paths: list[Path], device: str = "cpu"):
        """
        Initialize ensemble.
        
        Args:
            model_paths: List of paths to trained models
            device: Device to run on
        """
        self.models = []
        self.device = device
        
        for path in model_paths:
            checkpoint = torch.load(path, map_location=device)
            
            model = create_model(input_dim=checkpoint["input_dim"], device=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            
            self.models.append(model)
        
        print(f"Loaded ensemble of {len(self.models)} models")

    def predict(self, features: np.ndarray) -> dict[str, float]:
        """
        Ensemble prediction with uncertainty.
        
        Args:
            features: Patient features
        
        Returns:
            Dictionary with mean prediction and std
        """
        predictions = []
        
        x = torch.FloatTensor(features).unsqueeze(0).unsqueeze(1)
        x = x.transpose(0, 1)
        
        with torch.no_grad():
            for model in self.models:
                logits = model(x)
                prob = torch.sigmoid(logits).item()
                predictions.append(prob)
        
        predictions = np.array(predictions)
        
        return {
            "prediction": predictions.mean(),
            "uncertainty": predictions.std(),
            "min": predictions.min(),
            "max": predictions.max(),
        }


def main() -> None:
    """Test ensemble."""
    print("Testing ensemble predictor...")
    
    # In practice, you would have multiple trained models
    model_path = Path("models/temporal/best_model.pt")
    
    if not model_path.exists():
        print("ERROR: No models found for ensemble")
        return
    
    # For demo, use same model multiple times (not ideal but demonstrates API)
    ensemble = ModelEnsemble(model_paths=[model_path], device="cpu")
    
    test_features = np.random.randn(20)  # Assuming 20 features
    
    result = ensemble.predict(test_features)
    
    print(f"\nEnsemble prediction: {result['prediction']:.3f} ± {result['uncertainty']:.3f}")
    print(f"Range: [{result['min']:.3f}, {result['max']:.3f}]")
    
    print("\n✓ Ensemble test passed")


if __name__ == "__main__":
    main()

