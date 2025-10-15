"""
Federated learning client node.

Each hospital runs a client that trains locally and sends only model updates.
No raw patient data leaves the local site.
"""

from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.modeling.temporal.train import InfectionDataset, load_data
from src.modeling.temporal.transformer_model import create_model


class InfectionClient(fl.client.NumPyClient):
    """Flower client for federated infection prediction."""

    def __init__(
        self,
        site_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "cpu",
    ):
        """
        Initialize the federated client.
        
        Args:
            site_id: Unique hospital site identifier
            model: Local model instance
            train_loader: Training data loader
            test_loader: Test data loader
            device: Device to train on
        """
        self.site_id = site_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        """Return model parameters as numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Update model parameters from numpy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        """
        Train the model on local data.
        
        Args:
            parameters: Global model parameters
            config: Training configuration
        
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        # Update local model with global parameters
        self.set_parameters(parameters)
        
        # Get training config
        epochs = config.get("epochs", 1)
        learning_rate = config.get("learning_rate", 0.001)
        
        # Set up optimizer and criterion
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        # Train for specified epochs
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for features, labels in self.train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Transpose for transformer
                features = features.transpose(0, 1)
                
                optimizer.zero_grad()
                
                logits = self.model(features)
                loss = criterion(logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.train_loader)
            print(f"[{self.site_id}] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Return updated model parameters
        parameters_prime = self.get_parameters(config={})
        num_examples = len(self.train_loader.dataset)
        metrics = {"loss": avg_loss}
        
        return parameters_prime, num_examples, metrics

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, float]]:
        """
        Evaluate the model on local test data.
        
        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration
        
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        # Update model with parameters
        self.set_parameters(parameters)
        
        self.model.eval()
        criterion = nn.BCEWithLogitsLoss()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in self.test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                features = features.transpose(0, 1)
                
                logits = self.model(features)
                loss = criterion(logits, labels)
                
                probs = torch.sigmoid(logits)
                
                total_loss += loss.item()
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.test_loader)
        
        # Compute metrics
        from sklearn.metrics import average_precision_score, roc_auc_score
        
        y_pred = np.array(all_preds)
        y_true = np.array(all_labels)
        
        auroc = roc_auc_score(y_true, y_pred)
        auprc = average_precision_score(y_true, y_pred)
        
        num_examples = len(self.test_loader.dataset)
        
        metrics = {
            "auroc": auroc,
            "auprc": auprc,
        }
        
        print(f"[{self.site_id}] Evaluation - Loss: {avg_loss:.4f}, AUROC: {auroc:.4f}")
        
        return avg_loss, num_examples, metrics


def create_client_fn(
    site_id: str,
    data_dir: Path,
    input_dim: int,
    batch_size: int = 32,
) -> Callable[[str], InfectionClient]:
    """
    Create a client factory function for Flower.
    
    Args:
        site_id: Hospital site identifier
        data_dir: Directory containing labeled data
        input_dim: Number of input features
        batch_size: Batch size for training
    
    Returns:
        Client factory function
    """
    
    def client_fn(cid: str) -> InfectionClient:
        # Load site-specific data
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        labeled_file = data_dir / f"{site_id}_labeled.parquet"
        df = pd.read_parquet(labeled_file)
        
        # Extract features and labels
        feature_cols = [
            col
            for col in df.columns
            if col
            not in [
                "patient_id",
                "encounter_id",
                "window_end_time",
                "infection_label",
                "hours_to_infection",
            ]
        ]
        
        X = df[feature_cols].fillna(df[feature_cols].median()).values
        y = df["infection_label"].values
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Create datasets
        train_dataset = InfectionDataset(X_train, y_train)
        test_dataset = InfectionDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = create_model(input_dim=input_dim, device="cpu")
        
        return InfectionClient(
            site_id=site_id,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device="cpu",
        )
    
    return client_fn


if __name__ == "__main__":
    # Test single client
    print("Testing federated client...")
    
    from src.modeling.temporal.train import load_data
    from sklearn.model_selection import train_test_split
    
    # Load data
    data_dir = Path("data/labeled")
    X, y = load_data(data_dir)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    train_dataset = InfectionDataset(X_train[:100], y_train[:100])  # Small sample
    test_dataset = InfectionDataset(X_test[:50], y_test[:50])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = create_model(input_dim=X.shape[1], device="cpu")
    
    client = InfectionClient(
        site_id="test_site",
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device="cpu",
    )
    
    # Test fit
    params = client.get_parameters(config={})
    params_prime, num_examples, metrics = client.fit(params, config={"epochs": 1})
    
    print(f"Trained on {num_examples} examples")
    print(f"Training metrics: {metrics}")
    
    # Test evaluate
    loss, num_examples, eval_metrics = client.evaluate(params_prime, config={})
    
    print(f"Evaluated on {num_examples} examples")
    print(f"Evaluation metrics: {eval_metrics}")
    
    print("\n✓ Client test passed")

