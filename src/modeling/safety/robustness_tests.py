"""
Robustness tests for model stability under stress conditions.

Tests model behavior with missing data, extreme values, and adversarial perturbations.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Callable


def test_missing_data_robustness(
    model: nn.Module,
    x: torch.Tensor,
    missing_rates: list[float] = [0.1, 0.2, 0.5],
) -> dict[float, np.ndarray]:
    """
    Test how model handles increasing rates of missing data.
    
    Args:
        model: Trained model
        x: Input tensor
        missing_rates: List of missing data rates to test
    
    Returns:
        Dictionary mapping missing_rate to predictions
    """
    model.eval()
    
    results = {}
    
    with torch.no_grad():
        # Baseline: no missing data
        baseline_pred = model.predict_proba(x).cpu().numpy()
        results[0.0] = baseline_pred
        
        # Test with missing data
        for rate in missing_rates:
            x_missing = x.clone()
            
            # Randomly mask features
            mask = torch.rand_like(x_missing) < rate
            x_missing[mask] = 0.0  # Simple zero imputation
            
            pred = model.predict_proba(x_missing).cpu().numpy()
            results[rate] = pred
    
    return results


def test_extreme_values(
    model: nn.Module,
    x: torch.Tensor,
    feature_indices: list[int],
) -> dict[str, np.ndarray]:
    """
    Test model behavior with extreme feature values.
    
    Args:
        model: Trained model
        x: Input tensor
        feature_indices: Indices of features to perturb
    
    Returns:
        Dictionary with baseline, min, and max predictions
    """
    model.eval()
    
    results = {}
    
    with torch.no_grad():
        # Baseline
        baseline_pred = model.predict_proba(x).cpu().numpy()
        results["baseline"] = baseline_pred
        
        # Test with very small values
        x_min = x.clone()
        for idx in feature_indices:
            x_min[:, :, idx] = x[:, :, idx].min() * 2  # Double the minimum
        
        pred_min = model.predict_proba(x_min).cpu().numpy()
        results["extreme_low"] = pred_min
        
        # Test with very large values
        x_max = x.clone()
        for idx in feature_indices:
            x_max[:, :, idx] = x[:, :, idx].max() * 2  # Double the maximum
        
        pred_max = model.predict_proba(x_max).cpu().numpy()
        results["extreme_high"] = pred_max
    
    return results


def test_input_noise_sensitivity(
    model: nn.Module,
    x: torch.Tensor,
    noise_levels: list[float] = [0.01, 0.05, 0.1],
) -> dict[float, dict[str, np.ndarray]]:
    """
    Test model sensitivity to input noise.
    
    Args:
        model: Trained model
        x: Input tensor
        noise_levels: Standard deviations of Gaussian noise to add
    
    Returns:
        Dictionary mapping noise_level to {mean_pred, std_pred}
    """
    model.eval()
    
    results = {}
    n_trials = 10
    
    with torch.no_grad():
        for noise_std in noise_levels:
            preds = []
            
            for _ in range(n_trials):
                # Add Gaussian noise
                noise = torch.randn_like(x) * noise_std
                x_noisy = x + noise
                
                pred = model.predict_proba(x_noisy).cpu().numpy()
                preds.append(pred)
            
            preds = np.array(preds)
            
            results[noise_std] = {
                "mean": preds.mean(axis=0),
                "std": preds.std(axis=0),
            }
    
    return results


def adversarial_perturbation_fgsm(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.1,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Fast Gradient Sign Method (FGSM) adversarial attack.
    
    Args:
        model: Trained model
        x: Input tensor (requires_grad=True)
        y: Ground truth labels
        epsilon: Perturbation magnitude
    
    Returns:
        Tuple of (perturbed_x, original_preds, adversarial_preds)
    """
    model.eval()
    
    x.requires_grad = True
    
    # Forward pass
    logits = model(x)
    
    # Compute loss
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, y)
    
    # Backward pass to get gradient
    model.zero_grad()
    loss.backward()
    
    # Create adversarial example
    x_grad_sign = x.grad.sign()
    x_adv = x + epsilon * x_grad_sign
    x_adv = x_adv.detach()
    
    # Get predictions
    with torch.no_grad():
        orig_preds = torch.sigmoid(model(x)).cpu().numpy()
        adv_preds = torch.sigmoid(model(x_adv)).cpu().numpy()
    
    return x_adv, orig_preds, adv_preds


def robustness_report(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor | None = None,
) -> None:
    """
    Generate a comprehensive robustness report.
    
    Args:
        model: Trained model
        x_test: Test input tensor
        y_test: Optional test labels for adversarial testing
    """
    print("\n" + "=" * 60)
    print("ROBUSTNESS EVALUATION REPORT")
    print("=" * 60)
    
    # Test 1: Missing data
    print("\n1. Missing Data Robustness:")
    missing_results = test_missing_data_robustness(model, x_test)
    
    baseline = missing_results[0.0]
    for rate, preds in missing_results.items():
        if rate > 0:
            delta = np.abs(preds - baseline).mean()
            print(f"   {int(rate*100):2d}% missing: Mean prediction change = {delta:.4f}")
    
    # Test 2: Extreme values
    print("\n2. Extreme Value Robustness:")
    feature_indices = [0, 1, 2]  # Test first few features
    extreme_results = test_extreme_values(model, x_test, feature_indices)
    
    baseline = extreme_results["baseline"]
    delta_low = np.abs(extreme_results["extreme_low"] - baseline).mean()
    delta_high = np.abs(extreme_results["extreme_high"] - baseline).mean()
    
    print(f"   Extreme low values: Mean prediction change = {delta_low:.4f}")
    print(f"   Extreme high values: Mean prediction change = {delta_high:.4f}")
    
    # Test 3: Input noise
    print("\n3. Input Noise Sensitivity:")
    noise_results = test_input_noise_sensitivity(model, x_test)
    
    for noise_std, result in noise_results.items():
        mean_uncertainty = result["std"].mean()
        print(f"   Noise σ={noise_std:.3f}: Mean uncertainty = {mean_uncertainty:.4f}")
    
    # Test 4: Adversarial robustness (if labels provided)
    if y_test is not None:
        print("\n4. Adversarial Robustness (FGSM):")
        
        # Take a small sample for adversarial testing
        sample_size = min(100, len(y_test))
        x_sample = x_test[:, :sample_size, :].clone()
        y_sample = y_test[:sample_size].clone()
        
        x_sample.requires_grad = True
        
        _, orig_preds, adv_preds = adversarial_perturbation_fgsm(
            model, x_sample, y_sample, epsilon=0.1
        )
        
        delta_adv = np.abs(adv_preds - orig_preds).mean()
        print(f"   ε=0.1: Mean prediction change = {delta_adv:.4f}")
    
    print("\n" + "=" * 60)


def main() -> None:
    """Test robustness evaluation."""
    from src.modeling.temporal.transformer_model import create_model
    
    print("Testing robustness evaluation...")
    
    # Create a dummy model
    model = create_model(input_dim=20, device="cpu")
    
    # Dummy test data
    batch_size = 10
    seq_len = 5
    input_dim = 20
    
    x_test = torch.randn(seq_len, batch_size, input_dim)
    y_test = torch.randint(0, 2, (batch_size,)).float()
    
    # Run robustness report
    robustness_report(model, x_test, y_test)
    
    print("\n✓ Robustness testing complete")


if __name__ == "__main__":
    main()

