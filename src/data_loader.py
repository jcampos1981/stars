"""
Data loading utilities for MALLORN TDE Classification Challenge

This module handles loading the complex MALLORN dataset structure:
- Log files (train_log.csv, test_log.csv) with metadata
- Lightcurve files across 20 split directories
- Time series data with multiple filters (u, g, r, i, z, y)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Load and prepare astronomical lightcurve data for TDE classification.

    The MALLORN dataset structure:
    - train_log.csv / test_log.csv: Object metadata (redshift, extinction, target)
    - split_01 to split_20: Directories containing lightcurve CSV files
    - Lightcurves: Time series with Time(MJD), Flux, Flux_err, Filter columns
    """

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize DataLoader.

        Args:
            data_dir: Directory containing the CSV files and split folders
        """
        self.data_dir = Path(data_dir)
        self.filters = ['u', 'g', 'r', 'i', 'z', 'y']  # LSST filters

    def load_log_file(self, filepath: str, is_training: bool = True) -> pd.DataFrame:
        """
        Load log file (train_log.csv or test_log.csv).

        Args:
            filepath: Path to log CSV file
            is_training: Whether this is training data

        Returns:
            DataFrame with object metadata
        """
        filepath = Path(filepath)
        print(f"Loading {'training' if is_training else 'test'} log from {filepath}...")

        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} objects")

        if is_training and 'target' in df.columns:
            print(f"\nClass distribution:")
            print(f"  Non-TDE (0): {(df['target'] == 0).sum()} ({100 * (df['target'] == 0).mean():.2f}%)")
            print(f"  TDE (1): {(df['target'] == 1).sum()} ({100 * (df['target'] == 1).mean():.2f}%)")

        print(f"\nLog file columns: {df.columns.tolist()}")
        return df

    def load_training_log(self) -> pd.DataFrame:
        """Load training log file."""
        filepath = self.data_dir / "train_log.csv"
        return self.load_log_file(filepath, is_training=True)

    def load_test_log(self) -> pd.DataFrame:
        """Load test log file."""
        filepath = self.data_dir / "test_log.csv"
        return self.load_log_file(filepath, is_training=False)

    def load_lightcurve(self, object_id: str, split_name: str, is_training: bool = True) -> pd.DataFrame:
        """
        Load lightcurve for a specific object.

        Args:
            object_id: Object identifier
            split_name: Split directory name (e.g., 'split_01')
            is_training: Whether to load from training or test lightcurves

        Returns:
            DataFrame with lightcurve time series data
        """
        filename = "train_full_lightcurves.csv" if is_training else "test_full_lightcurves.csv"
        filepath = self.data_dir / split_name / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Lightcurve file not found: {filepath}")

        # Load the full lightcurves file and filter for this object
        # Note: This is memory efficient for individual objects but inefficient if loading many
        # For production, consider loading all lightcurves at once and caching
        lc_df = pd.read_csv(filepath)
        object_lc = lc_df[lc_df['object_id'] == object_id].copy()

        return object_lc

    def load_all_lightcurves_from_split(self, split_name: str, is_training: bool = True) -> pd.DataFrame:
        """
        Load all lightcurves from a specific split.

        Args:
            split_name: Split directory name (e.g., 'split_01')
            is_training: Whether to load from training or test lightcurves

        Returns:
            DataFrame with all lightcurve data from this split
        """
        filename = "train_full_lightcurves.csv" if is_training else "test_full_lightcurves.csv"
        filepath = self.data_dir / split_name / filename

        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping...")
            return pd.DataFrame()

        lc_df = pd.read_csv(filepath)
        return lc_df

    def load_sample_submission(self) -> pd.DataFrame:
        """Load sample submission file."""
        filepath = self.data_dir / "sample_submission.csv"
        if filepath.exists():
            return pd.read_csv(filepath)
        else:
            print(f"Warning: sample_submission.csv not found at {filepath}")
            return None


if __name__ == "__main__":
    # Test the data loader
    print("=" * 60)
    print("Testing MALLORN Data Loader")
    print("=" * 60)

    loader = DataLoader()

    # Test loading log files
    print("\n1. Loading training log...")
    try:
        train_log = loader.load_training_log()
        print(f"Training log shape: {train_log.shape}")
        print(f"Sample object_ids: {train_log['object_id'].head(3).tolist()}")
        print(f"Splits present: {train_log['split'].unique()[:5].tolist()}...")
    except Exception as e:
        print(f"Error loading training log: {e}")

    print("\n2. Loading test log...")
    try:
        test_log = loader.load_test_log()
        print(f"Test log shape: {test_log.shape}")
        print(f"Sample object_ids: {test_log['object_id'].head(3).tolist()}")
    except Exception as e:
        print(f"Error loading test log: {e}")

    # Test loading a lightcurve
    print("\n3. Testing lightcurve loading...")
    try:
        if 'train_log' in locals():
            sample_obj = train_log.iloc[0]
            obj_id = sample_obj['object_id']
            split_name = sample_obj['split']

            print(f"Loading lightcurve for {obj_id} from {split_name}...")
            lc = loader.load_lightcurve(obj_id, split_name, is_training=True)
            print(f"Lightcurve shape: {lc.shape}")
            print(f"Filters present: {lc['Filter'].unique().tolist()}")
            print(f"Time range: {lc['Time (MJD)'].min():.1f} to {lc['Time (MJD)'].max():.1f}")
    except Exception as e:
        print(f"Error loading lightcurve: {e}")

    print("\n" + "=" * 60)
    print("Data loader test complete!")
    print("=" * 60)
