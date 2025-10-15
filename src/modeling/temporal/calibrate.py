"""
Calibrate model predictions to improve probability estimates.

Uses isotonic regression or Platt scaling on validation data.
"""

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ModelCalibrator:
    """Calibrate predicted probabilities using isotonic regression or Platt scaling."""

    def __init__(self, method: Literal["isotonic", "platt"] = "isotonic"):
        """
        Initialize the calibrator.
        
        Args:
            method: Calibration method - 'isotonic' or 'platt'
        """
        self.method = method
        
        if method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
        elif method == "platt":
            self.calibrator = LogisticRegression()
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """
        Fit the calibrator on validation data.
        
        Args:
            y_pred: Uncalibrated predicted probabilities
            y_true: Ground truth labels
        """
        if self.method == "isotonic":
            self.calibrator.fit(y_pred, y_true)
        elif self.method == "platt":
            # Platt scaling needs 2D input
            self.calibrator.fit(y_pred.reshape(-1, 1), y_true)

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Apply calibration to new predictions.
        
        Args:
            y_pred: Uncalibrated probabilities
        
        Returns:
            Calibrated probabilities
        """
        if self.method == "isotonic":
            return self.calibrator.predict(y_pred)
        elif self.method == "platt":
            return self.calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1]

    def save(self, path: Path) -> None:
        """Save calibrator to disk."""
        joblib.dump({"method": self.method, "calibrator": self.calibrator}, path)
        print(f"Calibrator saved to {path}")

    @staticmethod
    def load(path: Path) -> "ModelCalibrator":
        """Load calibrator from disk."""
        data = joblib.load(path)
        calibrator = ModelCalibrator(method=data["method"])
        calibrator.calibrator = data["calibrator"]
        return calibrator


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_uncalibrated: np.ndarray,
    y_pred_calibrated: np.ndarray,
    output_path: Path,
) -> None:
    """Plot calibration curve before and after calibration."""
    import matplotlib.pyplot as plt
    
    # Compute calibration curves
    frac_pos_uncal, mean_pred_uncal = calibration_curve(
        y_true, y_pred_uncalibrated, n_bins=10, strategy="uniform"
    )
    
    frac_pos_cal, mean_pred_cal = calibration_curve(
        y_true, y_pred_calibrated, n_bins=10, strategy="uniform"
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.plot(
        mean_pred_uncal,
        frac_pos_uncal,
        "s-",
        label="Uncalibrated",
        color="red",
        alpha=0.7,
    )
    plt.plot(
        mean_pred_cal,
        frac_pos_cal,
        "o-",
        label="Calibrated",
        color="green",
        alpha=0.7,
    )
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Calibration curve saved to {output_path}")


def main() -> None:
    """Example: calibrate model predictions."""
    # Load predictions from evaluation
    predictions_file = Path("models/temporal/evaluation/predictions.csv")
    
    if not predictions_file.exists():
        print(f"ERROR: Predictions not found at {predictions_file}")
        print("Run 'make eval' first")
        return
    
    import pandas as pd
    
    df = pd.read_csv(predictions_file)
    y_true = df["y_true"].values
    y_pred_uncalibrated = df["y_pred"].values
    
    # Split into calibration and test sets
    from sklearn.model_selection import train_test_split
    
    y_true_cal, y_true_test, y_pred_cal, y_pred_test = train_test_split(
        y_true, y_pred_uncalibrated, test_size=0.5, stratify=y_true, random_state=42
    )
    
    # Fit calibrator
    calibrator = ModelCalibrator(method="isotonic")
    calibrator.fit(y_pred_cal, y_true_cal)
    
    # Apply to test set
    y_pred_calibrated = calibrator.transform(y_pred_test)
    
    # Compute calibration error before and after
    from netcal.metrics import ECE
    
    ece_metric = ECE(bins=10)
    ece_before = ece_metric.measure(y_pred_test, y_true_test)
    ece_after = ece_metric.measure(y_pred_calibrated, y_true_test)
    
    print(f"Calibration error before: {ece_before:.4f}")
    print(f"Calibration error after: {ece_after:.4f}")
    print(f"Improvement: {(ece_before - ece_after):.4f}")
    
    # Plot
    plot_dir = Path("models/temporal/evaluation")
    plot_calibration_curve(
        y_true_test,
        y_pred_test,
        y_pred_calibrated,
        plot_dir / "calibration_curve.png",
    )
    
    # Save calibrator
    calibrator_path = Path("models/temporal/calibrator.pkl")
    calibrator.save(calibrator_path)
    
    print("\n✓ Calibration complete")


if __name__ == "__main__":
    main()

