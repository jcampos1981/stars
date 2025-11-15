"""
Data loading utilities for MALLORN TDE Classification Challenge
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class DataLoader:
    """
    Load and prepare astronomical lightcurve data for TDE classification.
    """

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize DataLoader.

        Args:
            data_dir: Directory containing the CSV files
        """
        self.data_dir = Path(data_dir)

    def load_training_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load training data.

        Args:
            filepath: Path to training CSV. If None, uses default location.

        Returns:
            DataFrame with training data
        """
        if filepath is None:
            filepath = self.data_dir / "training_log.csv"
        else:
            filepath = Path(filepath)

        print(f"Loading training data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} training samples")

        if 'label' in df.columns or 'class' in df.columns or 'target' in df.columns:
            label_col = 'label' if 'label' in df.columns else ('class' if 'class' in df.columns else 'target')
            print(f"Class distribution:\n{df[label_col].value_counts()}")

        return df

    def load_test_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load test data.

        Args:
            filepath: Path to test CSV. If None, uses default location.

        Returns:
            DataFrame with test data
        """
        if filepath is None:
            filepath = self.data_dir / "test_log.csv"
        else:
            filepath = Path(filepath)

        print(f"Loading test data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} test samples")

        return df

    def get_feature_columns(self, df: pd.DataFrame, exclude_cols: list = None) -> list:
        """
        Get feature column names, excluding ID and label columns.

        Args:
            df: DataFrame
            exclude_cols: Additional columns to exclude

        Returns:
            List of feature column names
        """
        if exclude_cols is None:
            exclude_cols = []

        # Common non-feature columns
        non_features = ['id', 'source_id', 'object_id', 'label', 'class', 'target'] + exclude_cols

        feature_cols = [col for col in df.columns if col.lower() not in [c.lower() for c in non_features]]

        return feature_cols

    def prepare_data(self, df: pd.DataFrame, is_training: bool = True) -> Tuple:
        """
        Prepare data for modeling.

        Args:
            df: Input DataFrame
            is_training: Whether this is training data (has labels)

        Returns:
            Tuple of (X, y, feature_names) if training, else (X, feature_names)
        """
        # Identify ID column
        id_cols = [col for col in df.columns if 'id' in col.lower()]

        # Get feature columns
        feature_cols = self.get_feature_columns(df)

        print(f"Using {len(feature_cols)} features")

        X = df[feature_cols].copy()

        # Handle missing values
        if X.isnull().any().any():
            print(f"Found {X.isnull().sum().sum()} missing values, filling with median")
            X = X.fillna(X.median())

        if is_training:
            # Find label column
            label_col = None
            for col in ['label', 'class', 'target']:
                if col in df.columns:
                    label_col = col
                    break

            if label_col is None:
                raise ValueError("Could not find label column in training data")

            y = df[label_col].values
            return X, y, feature_cols
        else:
            return X, feature_cols


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()

    try:
        train_df = loader.load_training_data()
        print("\nTraining data loaded successfully!")
        print(f"Shape: {train_df.shape}")
        print(f"\nColumns: {train_df.columns.tolist()}")
        print(f"\nFirst few rows:\n{train_df.head()}")
    except FileNotFoundError:
        print("Training file not found. Place training_log.csv in data/raw/")

    try:
        test_df = loader.load_test_data()
        print("\nTest data loaded successfully!")
        print(f"Shape: {test_df.shape}")
    except FileNotFoundError:
        print("Test file not found. Place test_log.csv in data/raw/")
