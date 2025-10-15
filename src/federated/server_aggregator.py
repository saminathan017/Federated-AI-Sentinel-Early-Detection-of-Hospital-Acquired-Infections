"""
Federated learning server aggregator.

Coordinates training across multiple hospital sites using Flower.
Aggregates model updates using FedAvg or custom strategies.
"""

from pathlib import Path
from typing import Any, Callable, Optional

import flwr as fl
import numpy as np
from flwr.common import Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg


class FederatedAggregationStrategy(FedAvg):
    """Custom aggregation strategy with additional logging and privacy controls."""

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        apply_differential_privacy: bool = False,
        dp_noise_multiplier: float = 1.0,
    ):
        """
        Initialize the aggregation strategy.
        
        Args:
            fraction_fit: Fraction of clients to sample for training
            fraction_evaluate: Fraction of clients to sample for evaluation
            min_fit_clients: Minimum number of clients for training
            min_evaluate_clients: Minimum number of clients for evaluation
            min_available_clients: Minimum total clients required
            on_fit_config_fn: Function to generate training config per round
            on_evaluate_config_fn: Function to generate evaluation config per round
            apply_differential_privacy: Whether to add DP noise to aggregated weights
            dp_noise_multiplier: Noise multiplier for differential privacy
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
        )
        
        self.apply_differential_privacy = apply_differential_privacy
        self.dp_noise_multiplier = dp_noise_multiplier

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[Any, Any]],
        failures: list[tuple[Any, Exception] | BaseException],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model updates from clients."""
        # Call parent aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None and self.apply_differential_privacy:
            # Add differential privacy noise
            print(f"[Round {server_round}] Applying differential privacy...")
            aggregated_parameters = self._add_dp_noise(aggregated_parameters)
        
        # Log aggregation metrics
        if aggregated_metrics:
            print(
                f"[Round {server_round}] Aggregated metrics: "
                f"loss={aggregated_metrics.get('loss', 'N/A')}"
            )
        
        return aggregated_parameters, aggregated_metrics

    def _add_dp_noise(self, parameters: Parameters) -> Parameters:
        """Add Gaussian noise to parameters for differential privacy."""
        params_list = fl.common.parameters_to_ndarrays(parameters)
        
        noisy_params = []
        
        for param_array in params_list:
            # Compute noise scale based on parameter sensitivity
            sensitivity = np.linalg.norm(param_array.flatten())
            noise_scale = sensitivity * self.dp_noise_multiplier
            
            # Add Gaussian noise
            noise = np.random.normal(0, noise_scale, size=param_array.shape)
            noisy_param = param_array + noise
            
            noisy_params.append(noisy_param)
        
        return fl.common.ndarrays_to_parameters(noisy_params)


def fit_config(server_round: int) -> dict[str, Scalar]:
    """Generate training configuration for each round."""
    config = {
        "server_round": server_round,
        "epochs": 1,  # Local epochs per round
        "learning_rate": 0.001,
    }
    
    return config


def evaluate_config(server_round: int) -> dict[str, Scalar]:
    """Generate evaluation configuration for each round."""
    return {"server_round": server_round}


def run_server(
    num_rounds: int = 10,
    min_clients: int = 2,
    apply_dp: bool = False,
    dp_noise_multiplier: float = 1.0,
) -> None:
    """
    Start the federated learning server.
    
    Args:
        num_rounds: Number of federated training rounds
        min_clients: Minimum number of clients required
        apply_dp: Whether to apply differential privacy
        dp_noise_multiplier: DP noise multiplier
    """
    strategy = FederatedAggregationStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        apply_differential_privacy=apply_dp,
        dp_noise_multiplier=dp_noise_multiplier,
    )
    
    print(f"Starting Federated Learning Server")
    print(f"  Rounds: {num_rounds}")
    print(f"  Min clients: {min_clients}")
    print(f"  Differential Privacy: {apply_dp}")
    if apply_dp:
        print(f"  DP noise multiplier: {dp_noise_multiplier}")
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    print("\n✓ Federated learning complete")


if __name__ == "__main__":
    import os
    
    num_rounds = int(os.getenv("FL_NUM_ROUNDS", "10"))
    min_clients = int(os.getenv("FL_MIN_CLIENTS", "2"))
    apply_dp = os.getenv("DP_ENABLED", "false").lower() == "true"
    dp_noise = float(os.getenv("DP_NOISE_MULTIPLIER", "1.0"))
    
    run_server(
        num_rounds=num_rounds,
        min_clients=min_clients,
        apply_dp=apply_dp,
        dp_noise_multiplier=dp_noise,
    )

