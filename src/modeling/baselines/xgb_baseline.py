"""
XGBoost baseline model with calibrated probabilities.

Trains a gradient boosting classifier on tabular features.
Uses isotonic regression for probability calibration.
"""

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


class XGBoostBaseline:
    """XGBoost model with calibration for infection prediction."""

    def __init__(self, calibration_method: str = "isotonic"):
        """
        Initialize the baseline model.
        
        Args:
            calibration_method: 'isotonic' or 'sigmoid' for calibration
        """
        self.calibration_method = calibration_method
        
        self.base_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            min_child_weight=5,
            scale_pos_weight=10,  # Handle class imbalance
            random_state=42,
            n_jobs=-1,
        )
        
        self.model = None
        self.feature_names = None

    def load_data(self, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load labeled training data from all sites.
        
        Returns (features_df, labels_df)
        """
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
        
        X = combined[feature_cols]
        y = combined["infection_label"]
        
        # Handle missing values with median imputation
        X = X.fillna(X.median())
        
        self.feature_names = feature_cols
        
        return X, y

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train and calibrate the model."""
        print("Training XGBoost baseline...")
        
        # Train with calibration
        self.model = CalibratedClassifierCV(
            self.base_model,
            method=self.calibration_method,
            cv=3,
        )
        
        self.model.fit(X_train, y_train)
        
        print("✓ Training complete")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """Evaluate model performance."""
        y_pred_proba = self.predict_proba(X_test)
        
        auroc = roc_auc_score(y_test, y_pred_proba)
        auprc = average_precision_score(y_test, y_pred_proba)
        
        metrics = {
            "auroc": auroc,
            "auprc": auprc,
        }
        
        print(f"AUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
        
        return metrics

    def save(self, output_dir: Path) -> None:
        """Save model to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / "xgb_baseline.pkl"
        
        import joblib
        
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "calibration_method": self.calibration_method,
            },
            model_path,
        )
        
        print(f"Model saved to {model_path}")

    @staticmethod
    def load(model_path: Path) -> "XGBoostBaseline":
        """Load a saved model."""
        import joblib
        
        data = joblib.load(model_path)
        
        baseline = XGBoostBaseline(calibration_method=data["calibration_method"])
        baseline.model = data["model"]
        baseline.feature_names = data["feature_names"]
        
        return baseline


def main() -> None:
    """Train and evaluate the XGBoost baseline."""
    # Set MLflow tracking
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("xgboost_baseline")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", "xgboost_baseline")
        mlflow.log_param("calibration_method", "isotonic")
        
        # Initialize model
        baseline = XGBoostBaseline(calibration_method="isotonic")
        
        # Load data
        data_dir = Path("data/labeled")
        X, y = baseline.load_data(data_dir)
        
        print(f"Loaded {len(X)} samples with {len(baseline.feature_names)} features")
        print(f"Positive class: {y.sum()} ({y.mean():.2%})")
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Train
        baseline.train(X_train, y_train)
        
        # Evaluate
        metrics = baseline.evaluate(X_test, y_test)
        
        # Log metrics
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        
        # Save model
        model_dir = Path("models/baseline")
        baseline.save(model_dir)
        
        # Log model artifact
        mlflow.sklearn.log_model(baseline.model, "model")
        
        print("\n✓ XGBoost baseline training complete")


if __name__ == "__main__":
    main()

