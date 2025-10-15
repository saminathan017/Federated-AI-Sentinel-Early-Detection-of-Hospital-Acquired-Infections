"""
Automatic retrain triggers based on drift detection.

Monitors drift metrics and triggers retraining when thresholds are exceeded.
Creates tickets and can automatically kick off training jobs.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.drift.detectors import DriftDetector


class RetrainTrigger:
    """Automatically trigger model retraining when drift is detected."""

    def __init__(
        self,
        drift_detector: DriftDetector,
        trigger_log_path: Path = Path("logs/retrain_triggers.jsonl"),
        auto_retrain: bool = False,
    ):
        """
        Initialize retrain trigger.
        
        Args:
            drift_detector: Configured drift detector
            trigger_log_path: Where to log trigger events
            auto_retrain: If True, automatically start retraining (use with caution)
        """
        self.drift_detector = drift_detector
        self.trigger_log_path = trigger_log_path
        self.auto_retrain = auto_retrain
        
        # Ensure log directory exists
        trigger_log_path.parent.mkdir(parents=True, exist_ok=True)

    def check_and_trigger(
        self,
        current_data: pd.DataFrame,
        current_predictions: pd.Series | None = None,
        reference_predictions: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Check for drift and trigger retraining if needed.
        
        Args:
            current_data: Current production data
            current_predictions: Current model predictions
            reference_predictions: Reference predictions
        
        Returns:
            Dictionary with trigger decision and reason
        """
        # Detect feature drift
        feature_drift = self.drift_detector.detect_feature_drift(current_data)
        
        drifted_features = feature_drift[feature_drift["drift_detected"]]
        num_drifted = len(drifted_features)
        total_features = len(feature_drift)
        
        # Detect prediction drift if available
        pred_drift_detected = False
        pred_drift_psi = 0.0
        
        if current_predictions is not None and reference_predictions is not None:
            pred_drift = self.drift_detector.detect_prediction_drift(
                reference_predictions.values,
                current_predictions.values,
            )
            pred_drift_detected = pred_drift["drift_detected"]
            pred_drift_psi = pred_drift["psi"]
        
        # Decision logic
        should_retrain = False
        reason = []
        
        # Trigger if >25% of features have drifted
        if num_drifted / total_features > 0.25:
            should_retrain = True
            reason.append(f"{num_drifted}/{total_features} features drifted (>{25}%)")
        
        # Trigger if prediction drift is severe
        if pred_drift_detected and pred_drift_psi >= 0.2:
            should_retrain = True
            reason.append(f"Prediction PSI={pred_drift_psi:.3f} (threshold=0.2)")
        
        # Create trigger event
        trigger_event = {
            "timestamp": datetime.now().isoformat(),
            "should_retrain": should_retrain,
            "reason": "; ".join(reason) if reason else "No drift detected",
            "num_drifted_features": int(num_drifted),
            "total_features": int(total_features),
            "drifted_features": drifted_features["feature"].tolist(),
            "prediction_drift_psi": float(pred_drift_psi) if pred_drift_psi else None,
        }
        
        # Log the event
        self._log_trigger_event(trigger_event)
        
        # Execute retrain if needed
        if should_retrain:
            print(f"\n⚠ RETRAIN TRIGGERED: {trigger_event['reason']}")
            
            if self.auto_retrain:
                print("Starting automatic retraining...")
                self._execute_retrain()
            else:
                print("Manual retraining required. Run 'make train_temporal'")
        
        else:
            print("✓ No retraining needed")
        
        return trigger_event

    def _log_trigger_event(self, event: dict[str, Any]) -> None:
        """Append trigger event to log file."""
        with open(self.trigger_log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def _execute_retrain(self) -> None:
        """
        Execute retraining process.
        
        TODO: Implement automatic retraining workflow:
        1. Snapshot current production model
        2. Prepare new training data
        3. Kick off training job
        4. Validate new model
        5. Stage for A/B testing before full deployment
        """
        print("WARNING: Auto-retrain not fully implemented")
        print("In production, this would:")
        print("  1. Snapshot current model")
        print("  2. Launch training job")
        print("  3. Validate new model")
        print("  4. Stage for review before deployment")

    def get_trigger_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Retrieve recent trigger events.
        
        Args:
            limit: Maximum number of events to return
        
        Returns:
            List of trigger events
        """
        if not self.trigger_log_path.exists():
            return []
        
        events = []
        
        with open(self.trigger_log_path) as f:
            for line in f:
                events.append(json.loads(line))
        
        # Return most recent first
        return events[-limit:][::-1]


def main() -> None:
    """Test retrain trigger."""
    import numpy as np
    
    print("Testing retrain trigger...")
    
    # Create synthetic data with drift
    np.random.seed(42)
    
    n_samples = 1000
    
    ref_data = pd.DataFrame({
        "hr_mean": np.random.normal(80, 10, n_samples),
        "temp_mean": np.random.normal(37, 0.5, n_samples),
        "wbc_latest": np.random.normal(8, 2, n_samples),
    })
    
    # Current data with significant drift
    cur_data = pd.DataFrame({
        "hr_mean": np.random.normal(95, 12, n_samples),  # Shifted
        "temp_mean": np.random.normal(37.5, 0.6, n_samples),  # Shifted
        "wbc_latest": np.random.normal(11, 3, n_samples),  # Shifted
    })
    
    ref_preds = pd.Series(np.random.beta(2, 8, n_samples))
    cur_preds = pd.Series(np.random.beta(3, 7, n_samples))  # Shifted
    
    # Create detector and trigger
    detector = DriftDetector(
        reference_data=ref_data,
        feature_columns=["hr_mean", "temp_mean", "wbc_latest"],
        psi_threshold=0.1,
    )
    
    trigger = RetrainTrigger(
        drift_detector=detector,
        trigger_log_path=Path("logs/test_triggers.jsonl"),
        auto_retrain=False,
    )
    
    # Check and trigger
    result = trigger.check_and_trigger(
        current_data=cur_data,
        current_predictions=cur_preds,
        reference_predictions=ref_preds,
    )
    
    print(f"\nTrigger decision: {result['should_retrain']}")
    print(f"Reason: {result['reason']}")
    
    # View history
    history = trigger.get_trigger_history(limit=5)
    print(f"\nTrigger history: {len(history)} events")
    
    print("\n✓ Retrain trigger test complete")


if __name__ == "__main__":
    main()

