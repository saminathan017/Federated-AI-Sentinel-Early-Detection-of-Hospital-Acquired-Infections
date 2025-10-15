"""
Label time windows with infection outcomes.

A window is labeled positive if a hospital-acquired infection develops
within the prediction horizon (e.g., next 24 hours).
"""

from datetime import timedelta
from pathlib import Path

import pandas as pd


class InfectionLabeler:
    """Assign infection labels to time windows based on culture results."""

    def __init__(self, prediction_horizon_hours: int = 24):
        """
        Initialize the labeler.
        
        Args:
            prediction_horizon_hours: How far ahead to predict infection risk
        """
        self.prediction_horizon_hours = prediction_horizon_hours

    def load_cultures(self, site_dir: Path) -> pd.DataFrame:
        """Load microbiology culture results."""
        cultures = pd.read_parquet(site_dir / "cultures.parquet")
        cultures["culture_timestamp"] = pd.to_datetime(cultures["culture_timestamp"])
        return cultures

    def label_windows(
        self,
        windows_df: pd.DataFrame,
        cultures_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add infection labels to windows.
        
        A window is positive if a hospital-acquired culture is positive
        within the prediction horizon after the window end time.
        
        Args:
            windows_df: DataFrame of time windows
            cultures_df: DataFrame of culture results
        
        Returns:
            windows_df with 'infection_label' and 'hours_to_infection' columns
        """
        windows_df = windows_df.copy()
        windows_df["infection_label"] = 0
        windows_df["hours_to_infection"] = None
        
        # Filter to hospital-acquired infections only
        hai_cultures = cultures_df[cultures_df["is_hospital_acquired"] == True]
        
        for idx, window in windows_df.iterrows():
            patient_id = window["patient_id"]
            encounter_id = window["encounter_id"]
            window_end = window["window_end_time"]
            
            # Find cultures for this encounter
            encounter_cultures = hai_cultures[
                (hai_cultures["patient_id"] == patient_id)
                & (hai_cultures["encounter_id"] == encounter_id)
            ]
            
            if len(encounter_cultures) == 0:
                continue
            
            # Check if any culture falls within prediction horizon
            for _, culture in encounter_cultures.iterrows():
                culture_time = culture["culture_timestamp"]
                hours_until_culture = (culture_time - window_end).total_seconds() / 3600
                
                # Positive label if culture is within horizon and in the future
                if 0 < hours_until_culture <= self.prediction_horizon_hours:
                    windows_df.at[idx, "infection_label"] = 1
                    windows_df.at[idx, "hours_to_infection"] = hours_until_culture
                    break  # Label once per window
        
        label_counts = windows_df["infection_label"].value_counts()
        print(f"Label distribution:\n{label_counts}")
        
        if 1 in label_counts.index:
            prevalence = label_counts[1] / len(windows_df)
            print(f"Infection prevalence: {prevalence:.2%}")
        
        return windows_df


def main() -> None:
    """Example: label windowed data for all sites."""
    labeler = InfectionLabeler(prediction_horizon_hours=24)
    
    data_root = Path("data/synthetic")
    windowed_root = Path("data/windowed")
    labeled_root = Path("data/labeled")
    labeled_root.mkdir(parents=True, exist_ok=True)
    
    for site in ["site_a", "site_b", "site_c"]:
        site_dir = data_root / site
        windowed_file = windowed_root / f"{site}_windows.parquet"
        
        if not windowed_file.exists():
            print(f"Skipping {site}: windowed data not found")
            continue
        
        print(f"\nLabeling {site}...")
        
        windows = pd.read_parquet(windowed_file)
        windows["window_end_time"] = pd.to_datetime(windows["window_end_time"])
        
        cultures = labeler.load_cultures(site_dir)
        
        labeled_windows = labeler.label_windows(windows, cultures)
        
        output_file = labeled_root / f"{site}_labeled.parquet"
        labeled_windows.to_parquet(output_file)
        print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()

