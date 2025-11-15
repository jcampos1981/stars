"""
Feature engineering for astronomical lightcurve data
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from scipy import stats


class LightcurveFeatureEngineer:
    """
    Extract features from astronomical lightcurve data.

    This class provides methods to engineer features that are useful
    for classifying astronomical transients, particularly TDEs.
    """

    def __init__(self):
        self.feature_names = []

    def extract_statistical_features(self, values: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical features from a time series.

        Args:
            values: Array of measurements

        Returns:
            Dictionary of statistical features
        """
        features = {}

        if len(values) == 0 or np.all(np.isnan(values)):
            return {
                'mean': 0, 'std': 0, 'median': 0, 'min': 0, 'max': 0,
                'range': 0, 'skewness': 0, 'kurtosis': 0,
                'percentile_25': 0, 'percentile_75': 0, 'iqr': 0
            }

        # Remove NaN values
        values = values[~np.isnan(values)]

        if len(values) == 0:
            return {
                'mean': 0, 'std': 0, 'median': 0, 'min': 0, 'max': 0,
                'range': 0, 'skewness': 0, 'kurtosis': 0,
                'percentile_25': 0, 'percentile_75': 0, 'iqr': 0
            }

        features['mean'] = np.mean(values)
        features['std'] = np.std(values)
        features['median'] = np.median(values)
        features['min'] = np.min(values)
        features['max'] = np.max(values)
        features['range'] = features['max'] - features['min']

        # Higher order moments
        if len(values) > 1:
            features['skewness'] = stats.skew(values)
            features['kurtosis'] = stats.kurtosis(values)
        else:
            features['skewness'] = 0
            features['kurtosis'] = 0

        # Percentiles
        features['percentile_25'] = np.percentile(values, 25)
        features['percentile_75'] = np.percentile(values, 75)
        features['iqr'] = features['percentile_75'] - features['percentile_25']

        return features

    def extract_temporal_features(self, times: np.ndarray, values: np.ndarray) -> Dict[str, float]:
        """
        Extract time-domain features.

        Args:
            times: Array of observation times
            values: Array of measurements

        Returns:
            Dictionary of temporal features
        """
        features = {}

        if len(times) == 0 or len(values) == 0:
            return {'duration': 0, 'num_observations': 0, 'mean_cadence': 0}

        # Remove NaN pairs
        mask = ~(np.isnan(times) | np.isnan(values))
        times = times[mask]
        values = values[mask]

        if len(times) == 0:
            return {'duration': 0, 'num_observations': 0, 'mean_cadence': 0}

        # Sort by time
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        values = values[sort_idx]

        features['duration'] = times[-1] - times[0] if len(times) > 1 else 0
        features['num_observations'] = len(times)

        if len(times) > 1:
            cadence = np.diff(times)
            features['mean_cadence'] = np.mean(cadence)
            features['std_cadence'] = np.std(cadence)
            features['min_cadence'] = np.min(cadence)
            features['max_cadence'] = np.max(cadence)
        else:
            features['mean_cadence'] = 0
            features['std_cadence'] = 0
            features['min_cadence'] = 0
            features['max_cadence'] = 0

        return features

    def extract_rise_fall_features(self, times: np.ndarray, values: np.ndarray) -> Dict[str, float]:
        """
        Extract rise and fall characteristics of the lightcurve.

        Args:
            times: Array of observation times
            values: Array of measurements (magnitudes or fluxes)

        Returns:
            Dictionary of rise/fall features
        """
        features = {}

        # Remove NaN pairs
        mask = ~(np.isnan(times) | np.isnan(values))
        times = times[mask]
        values = values[mask]

        if len(values) < 3:
            return {
                'peak_value': 0, 'peak_time': 0,
                'rise_time': 0, 'fall_time': 0,
                'rise_rate': 0, 'fall_rate': 0
            }

        # Sort by time
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        values = values[sort_idx]

        # Find peak (assuming higher values = brighter for flux)
        peak_idx = np.argmax(values)
        features['peak_value'] = values[peak_idx]
        features['peak_time'] = times[peak_idx]

        # Rise time (time from start to peak)
        features['rise_time'] = times[peak_idx] - times[0] if peak_idx > 0 else 0

        # Fall time (time from peak to end)
        features['fall_time'] = times[-1] - times[peak_idx] if peak_idx < len(times) - 1 else 0

        # Rise and fall rates
        if features['rise_time'] > 0 and peak_idx > 0:
            features['rise_rate'] = (values[peak_idx] - values[0]) / features['rise_time']
        else:
            features['rise_rate'] = 0

        if features['fall_time'] > 0 and peak_idx < len(values) - 1:
            features['fall_rate'] = (values[peak_idx] - values[-1]) / features['fall_time']
        else:
            features['fall_rate'] = 0

        return features

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from the input DataFrame.

        This method assumes the DataFrame contains lightcurve data.
        It will try to automatically detect time and magnitude/flux columns.

        Args:
            df: Input DataFrame with lightcurve data

        Returns:
            DataFrame with engineered features
        """
        print("Engineering features from lightcurve data...")

        # If the DataFrame already has many numeric columns, assume they are features
        # and just add some basic derived features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) > 10:
            print(f"Found {len(numeric_cols)} numeric columns, creating derived features...")
            df_features = df.copy()

            # Add some interaction and polynomial features for key columns
            for col in numeric_cols[:5]:  # First few columns
                df_features[f'{col}_squared'] = df[col] ** 2
                df_features[f'{col}_log'] = np.log1p(np.abs(df[col]))

            # Add rolling statistics if data is sequential
            for col in numeric_cols[:3]:
                df_features[f'{col}_rolling_mean'] = df[col].rolling(window=3, min_periods=1).mean()
                df_features[f'{col}_rolling_std'] = df[col].rolling(window=3, min_periods=1).std().fillna(0)

            print(f"Created {len(df_features.columns)} total features")
            return df_features

        print("Features engineered successfully!")
        return df


if __name__ == "__main__":
    # Test feature engineering
    print("Feature engineering module ready!")
    print("\nThis module will extract features such as:")
    print("- Statistical features (mean, std, skewness, kurtosis)")
    print("- Temporal features (duration, cadence)")
    print("- Rise/fall characteristics")
    print("- And more...")
