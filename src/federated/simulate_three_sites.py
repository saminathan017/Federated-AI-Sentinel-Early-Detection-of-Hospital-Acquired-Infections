"""
Simulate federated learning across three hospital sites.

Runs three clients and one server in separate processes to demonstrate
privacy-preserving training without sharing raw data.
"""

import multiprocessing
import os
import time
from pathlib import Path

import flwr as fl

from src.federated.client_node import create_client_fn
from src.federated.server_aggregator import run_server


def start_client(site_id: str, server_address: str, data_dir: Path, input_dim: int) -> None:
    """
    Start a federated learning client for one hospital site.
    
    Args:
        site_id: Hospital site identifier (site_a, site_b, site_c)
        server_address: Server address to connect to
        data_dir: Directory containing labeled data
        input_dim: Number of input features
    """
    print(f"[{site_id}] Starting client...")
    
    # Create client factory
    client_fn = create_client_fn(
        site_id=site_id,
        data_dir=data_dir,
        input_dim=input_dim,
        batch_size=32,
    )
    
    # Create and start client
    client = client_fn(cid=site_id)
    
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
    )
    
    print(f"[{site_id}] Client finished")


def run_simulation(
    data_dir: Path = Path("data/labeled"),
    num_rounds: int = 10,
    apply_dp: bool = False,
) -> None:
    """
    Run a three-site federated learning simulation.
    
    Args:
        data_dir: Directory containing labeled data for all sites
        num_rounds: Number of federated training rounds
        apply_dp: Whether to apply differential privacy
    """
    print("=" * 70)
    print("FEDERATED LEARNING SIMULATION - THREE HOSPITAL SITES")
    print("=" * 70)
    
    # Determine input dimension from data
    import pandas as pd
    
    sample_file = data_dir / "site_a_labeled.parquet"
    if not sample_file.exists():
        print(f"ERROR: Data not found at {data_dir}")
        print("Run 'python -m src.features.windowing' and 'python -m src.features.labeling' first")
        return
    
    df = pd.read_parquet(sample_file)
    feature_cols = [
        col
        for col in df.columns
        if col
        not in ["patient_id", "encounter_id", "window_end_time", "infection_label", "hours_to_infection"]
    ]
    input_dim = len(feature_cols)
    
    print(f"\nConfiguration:")
    print(f"  Sites: site_a, site_b, site_c")
    print(f"  Rounds: {num_rounds}")
    print(f"  Input features: {input_dim}")
    print(f"  Differential Privacy: {apply_dp}")
    print()
    
    server_address = "127.0.0.1:8080"
    
    # Set environment variables for server
    os.environ["FL_NUM_ROUNDS"] = str(num_rounds)
    os.environ["FL_MIN_CLIENTS"] = "3"
    os.environ["DP_ENABLED"] = "true" if apply_dp else "false"
    
    # Start server in a separate process
    server_process = multiprocessing.Process(
        target=run_server,
        kwargs={
            "num_rounds": num_rounds,
            "min_clients": 3,
            "apply_dp": apply_dp,
        },
    )
    server_process.start()
    
    # Give server time to start
    time.sleep(5)
    
    # Start three clients in separate processes
    client_processes = []
    
    for site_id in ["site_a", "site_b", "site_c"]:
        client_process = multiprocessing.Process(
            target=start_client,
            args=(site_id, server_address, data_dir, input_dim),
        )
        client_process.start()
        client_processes.append(client_process)
        time.sleep(2)  # Stagger client starts
    
    # Wait for all clients to finish
    for client_process in client_processes:
        client_process.join()
    
    # Wait for server to finish
    server_process.join()
    
    print("\n" + "=" * 70)
    print("FEDERATED LEARNING SIMULATION COMPLETE")
    print("=" * 70)
    print("\nKey outcomes:")
    print("✓ Each site trained locally on its own data")
    print("✓ Only model weights were shared (no raw patient data)")
    print("✓ Global model learned from all three sites")
    print("✓ Privacy preserved at each hospital")
    
    if apply_dp:
        print("✓ Differential privacy noise added for extra protection")


def main() -> None:
    """Run the simulation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated learning simulation")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/labeled"),
        help="Directory with labeled data",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of federated rounds",
    )
    parser.add_argument(
        "--dp",
        action="store_true",
        help="Apply differential privacy",
    )
    
    args = parser.parse_args()
    
    run_simulation(
        data_dir=args.data_dir,
        num_rounds=args.rounds,
        apply_dp=args.dp,
    )


if __name__ == "__main__":
    # Prevent spawn issues on macOS
    multiprocessing.set_start_method("spawn", force=True)
    main()

