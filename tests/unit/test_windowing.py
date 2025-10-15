"""Unit tests for time windowing functionality."""

import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.features.windowing import TimeWindowBuilder


def test_time_window_builder_init():
    """Test initialization of window builder."""
    builder = TimeWindowBuilder(window_hours=24, stride_hours=4, lookback_hours=48)
    
    assert builder.window_hours == 24
    assert builder.stride_hours == 4
    assert builder.lookback_hours == 48


def test_aggregate_vitals_in_window():
    """Test vital signs aggregation."""
    builder = TimeWindowBuilder()
    
    # Create mock vitals data
    vitals_data = {
        "patient_id": ["P001"] * 10,
        "encounter_id": ["E001"] * 10,
        "timestamp": [datetime.now() + timedelta(hours=i) for i in range(10)],
        "code": ["heart_rate"] * 10,
        "value": [80 + i for i in range(10)],
    }
    
    vitals = pd.DataFrame(vitals_data)
    
    window_end = datetime.now() + timedelta(hours=9)
    
    features = builder.aggregate_vitals_in_window(vitals, "P001", "E001", window_end)
    
    assert "heart_rate_mean" in features
    assert "heart_rate_min" in features
    assert "heart_rate_max" in features
    assert "heart_rate_std" in features
    
    # Check values are reasonable
    assert features["heart_rate_mean"] > 0
    assert features["heart_rate_min"] <= features["heart_rate_mean"]
    assert features["heart_rate_max"] >= features["heart_rate_mean"]


def test_aggregate_labs_in_window():
    """Test lab results aggregation."""
    builder = TimeWindowBuilder()
    
    # Create mock labs data
    labs_data = {
        "patient_id": ["P001"] * 5,
        "encounter_id": ["E001"] * 5,
        "timestamp": [datetime.now() + timedelta(hours=i * 12) for i in range(5)],
        "code": ["wbc_count"] * 5,
        "value": [8.0, 9.0, 10.0, 11.0, 12.0],
    }
    
    labs = pd.DataFrame(labs_data)
    
    window_end = datetime.now() + timedelta(hours=48)
    
    features = builder.aggregate_labs_in_window(labs, "P001", "E001", window_end)
    
    assert "wbc_count_latest" in features
    assert "wbc_count_delta" in features
    
    # Latest value should be 12.0
    assert features["wbc_count_latest"] == 12.0
    
    # Delta should be positive (increasing trend)
    assert features["wbc_count_delta"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

