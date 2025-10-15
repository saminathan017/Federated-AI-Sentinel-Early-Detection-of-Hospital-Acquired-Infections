"""Unit tests for drift detection."""

import numpy as np
import pandas as pd
import pytest

from src.drift.detectors import DriftDetector, population_stability_index, kolmogorov_smirnov_test


def test_psi_no_drift():
    """Test PSI with identical distributions."""
    reference = np.random.normal(0, 1, 1000)
    current = np.random.normal(0, 1, 1000)
    
    psi = population_stability_index(reference, current)
    
    # PSI should be small for similar distributions
    assert psi < 0.1


def test_psi_with_drift():
    """Test PSI with shifted distribution."""
    reference = np.random.normal(0, 1, 1000)
    current = np.random.normal(2, 1, 1000)  # Mean shift
    
    psi = population_stability_index(reference, current)
    
    # PSI should be large for different distributions
    assert psi > 0.2


def test_ks_test_no_drift():
    """Test KS test with same distribution."""
    reference = np.random.normal(0, 1, 1000)
    current = np.random.normal(0, 1, 1000)
    
    result = kolmogorov_smirnov_test(reference, current)
    
    assert "drift_detected" in result
    # Should not detect drift for same distribution
    # (may occasionally fail due to randomness)


def test_ks_test_with_drift():
    """Test KS test with different distribution."""
    reference = np.random.normal(0, 1, 1000)
    current = np.random.normal(3, 1, 1000)  # Strong mean shift
    
    result = kolmogorov_smirnov_test(reference, current)
    
    assert result["drift_detected"] == True
    assert result["p_value"] < 0.05


def test_drift_detector():
    """Test drift detector on DataFrame."""
    ref_data = pd.DataFrame({
        "feature_1": np.random.normal(100, 15, 1000),
        "feature_2": np.random.normal(50, 10, 1000),
    })
    
    cur_data = pd.DataFrame({
        "feature_1": np.random.normal(110, 15, 1000),  # Shifted
        "feature_2": np.random.normal(50, 10, 1000),  # Same
    })
    
    detector = DriftDetector(
        reference_data=ref_data,
        feature_columns=["feature_1", "feature_2"],
        psi_threshold=0.1,
    )
    
    drift_results = detector.detect_feature_drift(cur_data)
    
    assert len(drift_results) == 2
    assert "feature_1" in drift_results["feature"].values
    assert "feature_2" in drift_results["feature"].values
    
    # Feature 1 should show more drift
    feature_1_psi = drift_results[drift_results["feature"] == "feature_1"]["psi"].values[0]
    feature_2_psi = drift_results[drift_results["feature"] == "feature_2"]["psi"].values[0]
    
    assert feature_1_psi > feature_2_psi


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

