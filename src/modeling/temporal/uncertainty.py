"""
Uncertainty quantification for predictions.

Provides Monte Carlo dropout and deep ensemble methods.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Callable


def enable_dropout(model: nn.Module) -> None:
    """Enable dropout layers during inference for MC dropout."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo dropout for uncertainty estimation.
    
    Args:
        model: Trained model with dropout layers
        x: Input tensor
        n_samples: Number of stochastic forward passes
    
    Returns:
        Tuple of (mean_prediction, uncertainty_std)
    """
    model.eval()
    enable_dropout(model)
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(x)
            probs = torch.sigmoid(logits)
            predictions.append(probs.cpu().numpy())
    
    predictions = np.array(predictions)  # Shape: (n_samples, batch_size)
    
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)
    
    return mean_pred, std_pred


def ensemble_predict(
    models: list[nn.Module],
    x: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deep ensemble prediction.
    
    Args:
        models: List of independently trained models
        x: Input tensor
    
    Returns:
        Tuple of (mean_prediction, uncertainty_std)
    """
    predictions = []
    
    with torch.no_grad():
        for model in models:
            model.eval()
            logits = model(x)
            probs = torch.sigmoid(logits)
            predictions.append(probs.cpu().numpy())
    
    predictions = np.array(predictions)  # Shape: (n_models, batch_size)
    
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)
    
    return mean_pred, std_pred


def entropy(probs: np.ndarray) -> np.ndarray:
    """
    Compute predictive entropy.
    
    Higher entropy indicates higher uncertainty.
    
    Args:
        probs: Predicted probabilities
    
    Returns:
        Entropy values
    """
    eps = 1e-10
    return -(probs * np.log(probs + eps) + (1 - probs) * np.log(1 - probs + eps))


def confidence_interval(
    mean: np.ndarray,
    std: np.ndarray,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute confidence interval.
    
    Args:
        mean: Mean predictions
        std: Standard deviation
        confidence: Confidence level (0-1)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy.stats import norm
    
    z_score = norm.ppf((1 + confidence) / 2)
    
    lower = mean - z_score * std
    upper = mean + z_score * std
    
    # Clip to [0, 1] for probabilities
    lower = np.clip(lower, 0, 1)
    upper = np.clip(upper, 0, 1)
    
    return lower, upper


class UncertaintyEstimator:
    """Wrapper for uncertainty estimation methods."""

    def __init__(
        self,
        method: str = "mcdropout",
        n_samples: int = 20,
    ):
        """
        Initialize uncertainty estimator.
        
        Args:
            method: 'mcdropout' or 'ensemble'
            n_samples: Number of samples for MC dropout
        """
        self.method = method
        self.n_samples = n_samples

    def predict_with_uncertainty(
        self,
        model: nn.Module | list[nn.Module],
        x: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        """
        Predict with uncertainty estimates.
        
        Returns dictionary with:
            - prediction: Mean prediction
            - uncertainty: Standard deviation
            - lower_95: Lower 95% confidence bound
            - upper_95: Upper 95% confidence bound
            - entropy: Predictive entropy
        """
        if self.method == "mcdropout":
            if isinstance(model, list):
                raise ValueError("MC dropout expects a single model")
            mean_pred, std_pred = mc_dropout_predict(model, x, self.n_samples)
        
        elif self.method == "ensemble":
            if not isinstance(model, list):
                raise ValueError("Ensemble expects a list of models")
            mean_pred, std_pred = ensemble_predict(model, x)
        
        else:
            raise ValueError(f"Unknown uncertainty method: {self.method}")
        
        lower_95, upper_95 = confidence_interval(mean_pred, std_pred, confidence=0.95)
        pred_entropy = entropy(mean_pred)
        
        return {
            "prediction": mean_pred,
            "uncertainty": std_pred,
            "lower_95": lower_95,
            "upper_95": upper_95,
            "entropy": pred_entropy,
        }


def main() -> None:
    """Test uncertainty estimation."""
    from src.modeling.temporal.transformer_model import create_model
    
    print("Testing uncertainty estimation...")
    
    # Create a dummy model
    model = create_model(input_dim=20, device="cpu")
    
    # Dummy input
    x = torch.randn(10, 4, 20)  # (seq_len, batch, features)
    
    # MC dropout
    estimator = UncertaintyEstimator(method="mcdropout", n_samples=20)
    results = estimator.predict_with_uncertainty(model, x)
    
    print(f"\nPredictions: {results['prediction']}")
    print(f"Uncertainty (std): {results['uncertainty']}")
    print(f"95% CI lower: {results['lower_95']}")
    print(f"95% CI upper: {results['upper_95']}")
    print(f"Entropy: {results['entropy']}")
    
    print("\n✓ Uncertainty estimation test passed")


if __name__ == "__main__":
    main()

