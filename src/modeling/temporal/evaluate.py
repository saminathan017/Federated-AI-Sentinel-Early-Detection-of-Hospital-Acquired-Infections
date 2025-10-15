"""
Evaluate trained models on test data.

Reports AUROC, AUPRC, calibration error, and precision-recall at fixed recall.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from netcal.metrics import ECE
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.modeling.temporal.train import InfectionDataset, load_data
from src.modeling.temporal.transformer_model import create_model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
    
    Returns:
        Dictionary of metric name to value
    """
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    
    # Calibration error
    ece_metric = ECE(bins=10)
    calibration_error = ece_metric.measure(y_pred, y_true)
    
    # Precision at fixed recall
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    
    # Find threshold that gives recall >= 0.9
    target_recall = 0.9
    idx = np.where(recall >= target_recall)[0]
    if len(idx) > 0:
        precision_at_90_recall = precision[idx[-1]]
    else:
        precision_at_90_recall = 0.0
    
    return {
        "auroc": auroc,
        "auprc": auprc,
        "calibration_error": calibration_error,
        "precision_at_90_recall": precision_at_90_recall,
    }


def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"ROC curve saved to {output_path}")


def plot_pr_curve(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """Plot and save precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"PR curve saved to {output_path}")


def main() -> None:
    """Evaluate the trained temporal transformer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load test data
    data_dir = Path("data/labeled")
    X, y = load_data(data_dir)
    
    # Use last 20% as test set (same split as training)
    from sklearn.model_selection import train_test_split
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    test_dataset = InfectionDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load trained model
    model_path = Path("models/temporal/best_model.pt")
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run 'make train_temporal' first")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = create_model(input_dim=checkpoint["input_dim"], device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Generate predictions
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            features = features.transpose(0, 1)
            
            logits = model(features)
            probs = torch.sigmoid(logits)
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    
    print("\n" + "=" * 50)
    print("EVALUATION METRICS")
    print("=" * 50)
    for name, value in metrics.items():
        print(f"{name:25s}: {value:.4f}")
    print("=" * 50)
    
    # Plot curves
    plot_dir = Path("models/temporal/evaluation")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    plot_roc_curve(y_true, y_pred, plot_dir / "roc_curve.png")
    plot_pr_curve(y_true, y_pred, plot_dir / "pr_curve.png")
    
    # Save predictions
    results_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    results_df.to_csv(plot_dir / "predictions.csv", index=False)
    print(f"Predictions saved to {plot_dir / 'predictions.csv'}")
    
    print("\n✓ Evaluation complete")


if __name__ == "__main__":
    main()

