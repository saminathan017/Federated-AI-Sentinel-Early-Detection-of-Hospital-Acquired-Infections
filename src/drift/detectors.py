"""
Drift detection using statistical tests.

Monitors for distribution shifts in input data and model predictions.
Uses Kolmogorov-Smirnov test and Population Stability Index.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Any


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Compute Population Stability Index (PSI) between two distributions.
    
    PSI < 0.1: No significant shift
    0.1 <= PSI < 0.2: Moderate shift
    PSI >= 0.2: Significant shift, retrain recommended
    
    Args:
        reference: Reference distribution (e.g., training data)
        current: Current distribution (e.g., recent production data)
        bins: Number of bins for discretization
    
    Returns:
        PSI score
    """
    # Create bins based on reference distribution
    bin_edges = np.histogram_bin_edges(reference, bins=bins)
    
    # Compute histograms
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)
    
    # Convert to proportions
    ref_props = ref_counts / len(reference)
    cur_props = cur_counts / len(current)
    
    # Avoid log(0) by adding small epsilon
    eps = 1e-10
    ref_props = np.maximum(ref_props, eps)
    cur_props = np.maximum(cur_props, eps)
    
    # Compute PSI
    psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
    
    return psi


def kolmogorov_smirnov_test(
    reference: np.ndarray,
    current: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Perform two-sample Kolmogorov-Smirnov test.
    
    Tests null hypothesis that both samples come from same distribution.
    
    Args:
        reference: Reference sample
        current: Current sample
        alpha: Significance level
    
    Returns:
        Dictionary with test statistic, p-value, and drift detected flag
    """
    statistic, p_value = stats.ks_2samp(reference, current)
    
    drift_detected = p_value < alpha
    
    return {
        "statistic": statistic,
        "p_value": p_value,
        "alpha": alpha,
        "drift_detected": drift_detected,
    }


class DriftDetector:
    """Monitor for distribution drift in features and predictions."""

    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_columns: list[str],
        psi_threshold: float = 0.2,
        ks_alpha: float = 0.05,
    ):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference dataset (e.g., training data)
            feature_columns: Columns to monitor for drift
            psi_threshold: PSI threshold for drift alert
            ks_alpha: Significance level for KS test
        """
        self.reference_data = reference_data
        self.feature_columns = feature_columns
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha

    def detect_feature_drift(
        self,
        current_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Detect drift in each feature.
        
        Args:
            current_data: Current production data
        
        Returns:
            DataFrame with drift metrics per feature
        """
        results = []
        
        for col in self.feature_columns:
            if col not in current_data.columns:
                continue
            
            ref_values = self.reference_data[col].dropna().values
            cur_values = current_data[col].dropna().values
            
            if len(cur_values) == 0:
                continue
            
            # Compute PSI
            psi = population_stability_index(ref_values, cur_values)
            
            # Perform KS test
            ks_result = kolmogorov_smirnov_test(ref_values, cur_values, self.ks_alpha)
            
            # Determine drift status
            drift = psi >= self.psi_threshold or ks_result["drift_detected"]
            
            results.append({
                "feature": col,
                "psi": psi,
                "ks_statistic": ks_result["statistic"],
                "ks_p_value": ks_result["p_value"],
                "drift_detected": drift,
            })
        
        return pd.DataFrame(results).sort_values("psi", ascending=False)

    def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
    ) -> dict[str, Any]:
        """
        Detect drift in model predictions.
        
        Args:
            reference_predictions: Predictions on reference data
            current_predictions: Recent production predictions
        
        Returns:
            Dictionary with drift metrics
        """
        psi = population_stability_index(reference_predictions, current_predictions)
        
        ks_result = kolmogorov_smirnov_test(
            reference_predictions,
            current_predictions,
            self.ks_alpha,
        )
        
        drift = psi >= self.psi_threshold or ks_result["drift_detected"]
        
        return {
            "psi": psi,
            "ks_statistic": ks_result["statistic"],
            "ks_p_value": ks_result["p_value"],
            "drift_detected": drift,
            "severity": "HIGH" if psi >= 0.3 else "MODERATE" if psi >= 0.2 else "LOW",
        }


def drift_report(
    detector: DriftDetector,
    current_data: pd.DataFrame,
    current_predictions: np.ndarray | None = None,
    reference_predictions: np.ndarray | None = None,
) -> None:
    """
    Generate a comprehensive drift report.
    
    Args:
        detector: Configured drift detector
        current_data: Current production data
        current_predictions: Current model predictions (optional)
        reference_predictions: Reference predictions (optional)
    """
    print("\n" + "=" * 70)
    print("DRIFT DETECTION REPORT")
    print("=" * 70)
    
    # Feature drift
    feature_drift = detector.detect_feature_drift(current_data)
    
    drifted_features = feature_drift[feature_drift["drift_detected"]]
    
    print(f"\nFeatures Monitored: {len(feature_drift)}")
    print(f"Features with Drift: {len(drifted_features)}")
    
    if len(drifted_features) > 0:
        print("\nDrifted Features:")
        print(drifted_features[["feature", "psi", "ks_p_value"]].to_string(index=False))
    else:
        print("\n✓ No significant feature drift detected")
    
    # Prediction drift
    if current_predictions is not None and reference_predictions is not None:
        pred_drift = detector.detect_prediction_drift(
            reference_predictions,
            current_predictions,
        )
        
        print(f"\nPrediction Drift:")
        print(f"  PSI: {pred_drift['psi']:.4f}")
        print(f"  KS p-value: {pred_drift['ks_p_value']:.4f}")
        print(f"  Severity: {pred_drift['severity']}")
        
        if pred_drift["drift_detected"]:
            print("  ⚠ ALERT: Significant prediction drift detected")
        else:
            print("  ✓ No significant prediction drift")
    
    print("=" * 70)


def main() -> None:
    """Test drift detection."""
    print("Testing drift detection...")
    
    # Generate synthetic reference and current data
    np.random.seed(42)
    
    # Reference data: normal distribution
    n_samples = 1000
    ref_data = pd.DataFrame({
        "feature_1": np.random.normal(100, 15, n_samples),
        "feature_2": np.random.normal(50, 10, n_samples),
        "feature_3": np.random.exponential(2, n_samples),
    })
    
    # Current data: shifted distribution (simulating drift)
    cur_data = pd.DataFrame({
        "feature_1": np.random.normal(105, 15, n_samples),  # Mean shift
        "feature_2": np.random.normal(50, 15, n_samples),  # Variance shift
        "feature_3": np.random.exponential(2, n_samples),  # No shift
    })
    
    # Create detector
    detector = DriftDetector(
        reference_data=ref_data,
        feature_columns=["feature_1", "feature_2", "feature_3"],
        psi_threshold=0.1,
        ks_alpha=0.05,
    )
    
    # Generate predictions
    ref_predictions = np.random.beta(2, 5, n_samples)
    cur_predictions = np.random.beta(3, 5, n_samples)  # Slight shift
    
    # Run drift report
    drift_report(
        detector=detector,
        current_data=cur_data,
        current_predictions=cur_predictions,
        reference_predictions=ref_predictions,
    )
    
    print("\n✓ Drift detection test complete")


if __name__ == "__main__":
    main()

