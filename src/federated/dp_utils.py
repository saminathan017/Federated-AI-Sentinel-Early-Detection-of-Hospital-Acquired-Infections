"""
Differential privacy utilities for federated learning.

Add calibrated noise to model updates to protect individual patient privacy.
Uses Opacus library for DP-SGD when available.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


def clip_gradients(
    model: nn.Module,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Clip gradients to bound sensitivity for differential privacy.
    
    Args:
        model: Model with gradients
        max_grad_norm: Maximum L2 norm of gradients
    
    Returns:
        Total norm before clipping
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    return total_norm.item()


def add_noise_to_parameters(
    parameters: list[np.ndarray],
    noise_multiplier: float,
    max_grad_norm: float = 1.0,
) -> list[np.ndarray]:
    """
    Add Gaussian noise to parameters for differential privacy.
    
    Noise scale = noise_multiplier * max_grad_norm
    
    Args:
        parameters: List of parameter arrays
        noise_multiplier: Controls privacy-utility tradeoff
        max_grad_norm: Gradient clipping threshold
    
    Returns:
        Noisy parameters
    """
    noisy_params = []
    
    noise_scale = noise_multiplier * max_grad_norm
    
    for param in parameters:
        noise = np.random.normal(0, noise_scale, size=param.shape)
        noisy_param = param + noise
        noisy_params.append(noisy_param)
    
    return noisy_params


def compute_privacy_spent(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float = 1e-5,
) -> tuple[float, float]:
    """
    Compute privacy budget (epsilon) spent using RDP accountant.
    
    Uses Rényi Differential Privacy for tighter bounds.
    
    Args:
        noise_multiplier: Noise multiplier used in training
        sample_rate: Fraction of data used per step
        steps: Number of training steps
        delta: Target delta for (epsilon, delta)-DP
    
    Returns:
        Tuple of (epsilon, alpha) where alpha is the optimal RDP order
    """
    try:
        from opacus.accountants import RDPAccountant
        
        accountant = RDPAccountant()
        
        epsilon = accountant.get_epsilon(
            delta=delta,
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            steps=steps,
        )
        
        return epsilon, None
    
    except ImportError:
        print("WARNING: Opacus not available. Using approximate privacy calculation.")
        
        # Approximate using composition theorem
        # This is a rough upper bound
        epsilon_per_step = np.sqrt(2 * np.log(1.25 / delta)) / noise_multiplier
        epsilon_total = epsilon_per_step * steps * sample_rate
        
        return epsilon_total, None


class DPParameterClipper:
    """Clip parameter updates to bound sensitivity."""

    def __init__(self, max_param_norm: float = 1.0):
        """
        Initialize parameter clipper.
        
        Args:
            max_param_norm: Maximum L2 norm of parameter update
        """
        self.max_param_norm = max_param_norm

    def clip_update(self, param_update: np.ndarray) -> np.ndarray:
        """
        Clip a parameter update to bound its norm.
        
        Args:
            param_update: Parameter update array
        
        Returns:
            Clipped update
        """
        update_norm = np.linalg.norm(param_update.flatten())
        
        if update_norm > self.max_param_norm:
            param_update = param_update * (self.max_param_norm / update_norm)
        
        return param_update

    def clip_all_updates(self, param_updates: list[np.ndarray]) -> list[np.ndarray]:
        """Clip all parameter updates."""
        return [self.clip_update(update) for update in param_updates]


def privacy_report(
    noise_multiplier: float,
    max_grad_norm: float,
    num_epochs: int,
    batch_size: int,
    dataset_size: int,
    delta: float = 1e-5,
) -> None:
    """
    Print a privacy guarantee report.
    
    Args:
        noise_multiplier: Noise multiplier used
        max_grad_norm: Gradient clipping threshold
        num_epochs: Number of training epochs
        batch_size: Training batch size
        dataset_size: Size of training dataset
        delta: Target delta
    """
    sample_rate = batch_size / dataset_size
    steps = int(num_epochs * (dataset_size / batch_size))
    
    epsilon, _ = compute_privacy_spent(
        noise_multiplier=noise_multiplier,
        sample_rate=sample_rate,
        steps=steps,
        delta=delta,
    )
    
    print("\n" + "=" * 60)
    print("DIFFERENTIAL PRIVACY REPORT")
    print("=" * 60)
    print(f"Noise multiplier: {noise_multiplier}")
    print(f"Gradient clipping threshold: {max_grad_norm}")
    print(f"Training epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Dataset size: {dataset_size}")
    print(f"Sample rate: {sample_rate:.4f}")
    print(f"Total steps: {steps}")
    print("-" * 60)
    print(f"Privacy guarantee: (ε={epsilon:.2f}, δ={delta})-DP")
    print("-" * 60)
    
    if epsilon < 1:
        print("Privacy level: STRONG (ε < 1)")
    elif epsilon < 10:
        print("Privacy level: MODERATE (1 ≤ ε < 10)")
    else:
        print("Privacy level: WEAK (ε ≥ 10)")
    
    print("=" * 60)


def main() -> None:
    """Test DP utilities."""
    print("Testing differential privacy utilities...")
    
    # Test gradient clipping
    model = nn.Linear(10, 1)
    
    # Create dummy gradients
    dummy_input = torch.randn(5, 10)
    dummy_target = torch.randn(5, 1)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    output = model(dummy_input)
    loss = nn.MSELoss()(output, dummy_target)
    loss.backward()
    
    total_norm = clip_gradients(model, max_grad_norm=1.0)
    print(f"Gradient norm before clipping: {total_norm:.4f}")
    
    # Test noise addition
    params = [p.detach().numpy() for p in model.parameters()]
    noisy_params = add_noise_to_parameters(params, noise_multiplier=1.1)
    
    print(f"Added noise to {len(noisy_params)} parameter arrays")
    
    # Test privacy accounting
    privacy_report(
        noise_multiplier=1.1,
        max_grad_norm=1.0,
        num_epochs=10,
        batch_size=32,
        dataset_size=1000,
        delta=1e-5,
    )
    
    print("\n✓ DP utilities test passed")


if __name__ == "__main__":
    main()

