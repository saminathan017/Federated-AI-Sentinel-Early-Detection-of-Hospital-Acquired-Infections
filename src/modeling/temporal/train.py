"""
Train the temporal transformer model.

Includes MLflow tracking, early stopping, and model checkpointing.
"""

import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.modeling.temporal.transformer_model import create_model


class InfectionDataset(Dataset):
    """PyTorch dataset for infection prediction."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            features: Array of shape (n_samples, n_features)
            labels: Array of shape (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Add sequence dimension (treat each sample as a sequence of length 1)
        # In practice, you would group samples by patient and create true sequences
        return self.features[idx].unsqueeze(0), self.labels[idx]


def load_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load labeled data from all sites."""
    all_data = []
    
    for site in ["site_a", "site_b", "site_c"]:
        labeled_file = data_dir / f"{site}_labeled.parquet"
        if labeled_file.exists():
            df = pd.read_parquet(labeled_file)
            all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Separate features and labels
    feature_cols = [
        col
        for col in combined.columns
        if col
        not in [
            "patient_id",
            "encounter_id",
            "window_end_time",
            "infection_label",
            "hours_to_infection",
        ]
    ]
    
    X = combined[feature_cols].fillna(combined[feature_cols].median()).values
    y = combined["infection_label"].values
    
    return X, y


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for features, labels in dataloader:
        features = features.to(device)  # (batch, seq_len, features)
        labels = labels.to(device)
        
        # Transpose for transformer: (seq_len, batch, features)
        features = features.transpose(0, 1)
        
        optimizer.zero_grad()
        
        logits = model(features)
        loss = criterion(logits, labels)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            
            features = features.transpose(0, 1)
            
            logits = model(features)
            loss = criterion(logits, labels)
            
            probs = torch.sigmoid(logits)
            
            total_loss += loss.item()
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, np.array(all_preds), np.array(all_labels)


def main() -> None:
    """Train the temporal transformer model."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("temporal_transformer")
    
    with mlflow.start_run():
        # Hyperparameters
        batch_size = 32
        learning_rate = 0.001
        num_epochs = 50
        early_stop_patience = 10
        
        # Log parameters
        mlflow.log_param("model_type", "temporal_transformer")
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_epochs", num_epochs)
        
        # Load data
        data_dir = Path("data/labeled")
        X, y = load_data(data_dir)
        
        print(f"Loaded {len(X)} samples with {X.shape[1]} features")
        print(f"Positive class: {y.sum()} ({y.mean():.2%})")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Create datasets
        train_dataset = InfectionDataset(X_train, y_train)
        test_dataset = InfectionDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        input_dim = X.shape[1]
        model = create_model(input_dim=input_dim, device=device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Loss function with class weighting
        pos_weight = torch.tensor([len(y_train) / y_train.sum() - 1]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        
        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_preds, val_labels = evaluate(model, test_loader, criterion, device)
            
            scheduler.step(val_loss)
            
            # Calculate metrics
            from sklearn.metrics import average_precision_score, roc_auc_score
            
            val_auroc = roc_auc_score(val_labels, val_preds)
            val_auprc = average_precision_score(val_labels, val_preds)
            
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUROC: {val_auroc:.4f} | "
                f"Val AUPRC: {val_auprc:.4f}"
            )
            
            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_auroc", val_auroc, step=epoch)
            mlflow.log_metric("val_auprc", val_auprc, step=epoch)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                model_dir = Path("models/temporal")
                model_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "input_dim": input_dim,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_auroc": val_auroc,
                    },
                    model_dir / "best_model.pt",
                )
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Log final model
        mlflow.pytorch.log_model(model, "model")
        
        print("\n✓ Training complete")
        print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

