"""
Build sliding time windows from patient observations.

Aggregates vitals and labs into fixed-length sequences for temporal modeling.
Each window represents the patient state at a specific point in time.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


class TimeWindowBuilder:
    """Create sliding windows from patient time series data."""

    def __init__(
        self,
        window_hours: int = 24,
        stride_hours: int = 4,
        lookback_hours: int = 48,
    ):
        """
        Initialize the window builder.
        
        Args:
            window_hours: Length of each prediction window
            stride_hours: Time between consecutive windows
            lookback_hours: How far back to aggregate features
        """
        self.window_hours = window_hours
        self.stride_hours = stride_hours
        self.lookback_hours = lookback_hours

    def load_site_data(self, site_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load vitals, labs, and cultures from a site directory.
        
        Returns:
            Tuple of (vitals_df, labs_df, cultures_df)
        """
        vitals = pd.read_parquet(site_dir / "vitals.parquet")
        labs = pd.read_parquet(site_dir / "labs.parquet")
        cultures = pd.read_parquet(site_dir / "cultures.parquet")
        
        # Parse timestamps
        vitals["timestamp"] = pd.to_datetime(vitals["timestamp"])
        labs["timestamp"] = pd.to_datetime(labs["timestamp"])
        cultures["culture_timestamp"] = pd.to_datetime(cultures["culture_timestamp"])
        
        return vitals, labs, cultures

    def aggregate_vitals_in_window(
        self,
        vitals: pd.DataFrame,
        patient_id: str,
        encounter_id: str,
        window_end: datetime,
    ) -> dict[str, float]:
        """
        Aggregate vital signs within a lookback window.
        
        Returns mean, min, max, std for each vital sign.
        """
        window_start = window_end - timedelta(hours=self.lookback_hours)
        
        mask = (
            (vitals["patient_id"] == patient_id)
            & (vitals["encounter_id"] == encounter_id)
            & (vitals["timestamp"] >= window_start)
            & (vitals["timestamp"] <= window_end)
        )
        
        windowed = vitals[mask]
        
        features = {}
        
        for vital_code in ["heart_rate", "respiratory_rate", "temperature", "oxygen_saturation"]:
            vital_values = windowed[windowed["code"] == vital_code]["value"]
            
            if len(vital_values) > 0:
                features[f"{vital_code}_mean"] = vital_values.mean()
                features[f"{vital_code}_min"] = vital_values.min()
                features[f"{vital_code}_max"] = vital_values.max()
                features[f"{vital_code}_std"] = vital_values.std() if len(vital_values) > 1 else 0.0
            else:
                # Missing data imputation with neutral values
                features[f"{vital_code}_mean"] = np.nan
                features[f"{vital_code}_min"] = np.nan
                features[f"{vital_code}_max"] = np.nan
                features[f"{vital_code}_std"] = np.nan
        
        return features

    def aggregate_labs_in_window(
        self,
        labs: pd.DataFrame,
        patient_id: str,
        encounter_id: str,
        window_end: datetime,
    ) -> dict[str, float]:
        """
        Aggregate lab results within a lookback window.
        
        Returns most recent value for each lab test.
        """
        window_start = window_end - timedelta(hours=self.lookback_hours)
        
        mask = (
            (labs["patient_id"] == patient_id)
            & (labs["encounter_id"] == encounter_id)
            & (labs["timestamp"] >= window_start)
            & (labs["timestamp"] <= window_end)
        )
        
        windowed = labs[mask]
        
        features = {}
        
        for lab_code in ["wbc_count", "c_reactive_protein", "procalcitonin", "lactate"]:
            lab_values = windowed[windowed["code"] == lab_code].sort_values("timestamp")
            
            if len(lab_values) > 0:
                # Use most recent value
                features[f"{lab_code}_latest"] = lab_values.iloc[-1]["value"]
                # Also track trend: change from first to last
                if len(lab_values) > 1:
                    features[f"{lab_code}_delta"] = (
                        lab_values.iloc[-1]["value"] - lab_values.iloc[0]["value"]
                    )
                else:
                    features[f"{lab_code}_delta"] = 0.0
            else:
                features[f"{lab_code}_latest"] = np.nan
                features[f"{lab_code}_delta"] = np.nan
        
        return features

    def build_windows_for_encounter(
        self,
        vitals: pd.DataFrame,
        labs: pd.DataFrame,
        patient_id: str,
        encounter_id: str,
        admission_time: datetime,
        discharge_time: datetime,
    ) -> list[dict[str, Any]]:
        """
        Build all windows for a single patient encounter.
        
        Returns list of feature dictionaries, one per window.
        """
        windows = []
        
        # Start windowing after enough lookback data is available
        first_window_end = admission_time + timedelta(hours=self.lookback_hours)
        
        current_time = first_window_end
        
        while current_time <= discharge_time:
            vital_features = self.aggregate_vitals_in_window(
                vitals, patient_id, encounter_id, current_time
            )
            
            lab_features = self.aggregate_labs_in_window(
                labs, patient_id, encounter_id, current_time
            )
            
            window_features = {
                "patient_id": patient_id,
                "encounter_id": encounter_id,
                "window_end_time": current_time,
                "hours_since_admission": (current_time - admission_time).total_seconds() / 3600,
                **vital_features,
                **lab_features,
            }
            
            windows.append(window_features)
            
            current_time += timedelta(hours=self.stride_hours)
        
        return windows

    def build_dataset(self, site_dir: Path) -> pd.DataFrame:
        """
        Build a complete windowed dataset for a hospital site.
        
        Returns DataFrame with one row per time window.
        """
        vitals, labs, cultures = self.load_site_data(site_dir)
        
        # Get unique encounters
        encounters = vitals[["patient_id", "encounter_id"]].drop_duplicates()
        
        all_windows = []
        
        for _, row in encounters.iterrows():
            patient_id = row["patient_id"]
            encounter_id = row["encounter_id"]
            
            # Get admission and discharge times
            encounter_vitals = vitals[
                (vitals["patient_id"] == patient_id) & (vitals["encounter_id"] == encounter_id)
            ]
            
            admission_time = encounter_vitals["timestamp"].min()
            discharge_time = encounter_vitals["timestamp"].max()
            
            windows = self.build_windows_for_encounter(
                vitals, labs, patient_id, encounter_id, admission_time, discharge_time
            )
            
            all_windows.extend(windows)
        
        df = pd.DataFrame(all_windows)
        
        print(f"Built {len(df)} windows from {len(encounters)} encounters")
        
        return df


def main() -> None:
    """Example: build windowed features for all sites."""
    builder = TimeWindowBuilder(window_hours=24, stride_hours=4, lookback_hours=48)
    
    data_root = Path("data/synthetic")
    output_root = Path("data/windowed")
    output_root.mkdir(parents=True, exist_ok=True)
    
    for site in ["site_a", "site_b", "site_c"]:
        site_dir = data_root / site
        if site_dir.exists():
            print(f"\nProcessing {site}...")
            windowed_df = builder.build_dataset(site_dir)
            
            output_file = output_root / f"{site}_windows.parquet"
            windowed_df.to_parquet(output_file)
            print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()

