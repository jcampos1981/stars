"""
Feature engineering for MALLORN TDE Classification Challenge

Extracts features from:
- Lightcurve time series data (per filter)
- Object metadata (redshift, extinction)
- Applies de-extinction correction to flux values
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from extinction import fitzpatrick99
    HAS_EXTINCTION = True
except ImportError:
    HAS_EXTINCTION = False
    print("Warning: 'extinction' package not found. De-extinction will be skipped.")


class LightcurveFeatureEngineer:
    """
    Extract features from astronomical lightcurve time series.

    Features extracted per filter (u, g, r, i, z, y):
    - Statistical: mean, std, median, min, max, range, skewness, kurtosis
    - Temporal: duration, number of observations, cadence statistics
    - Rise/Fall: peak time, rise/fall rates, rise/fall times
    - Flux-based: peak flux, total variation, SNR
    - De-extinction: Corrects flux for galactic dust extinction
    """

    def __init__(self, apply_deextinction: bool = True):
        """
        Initialize feature engineer.

        Args:
            apply_deextinction: Whether to apply de-extinction correction to fluxes
        """
        self.filters = ['u', 'g', 'r', 'i', 'z', 'y']
        self.feature_names = []
        self.apply_deextinction = apply_deextinction and HAS_EXTINCTION

        # Effective wavelengths for each LSST filter (in Angstroms)
        # Source: SVO Filter Profile Service
        self.filter_wavelengths = {
            'u': 3641.0,
            'g': 4704.0,
            'r': 6155.0,
            'i': 7504.0,
            'z': 8695.0,
            'y': 10056.0
        }

        if self.apply_deextinction:
            print("De-extinction correction will be applied to flux values")
        elif not HAS_EXTINCTION:
            print("De-extinction skipped: 'extinction' package not installed")

    def deextinct_flux(self, flux: np.ndarray, ebv: float, filter_name: str) -> np.ndarray:
        """
        Apply de-extinction correction to flux values.

        Uses Fitzpatrick99 extinction law to correct for Milky Way dust.

        Args:
            flux: Flux measurements (microjansky)
            ebv: E(B-V) extinction coefficient
            filter_name: Filter name (u, g, r, i, z, y)

        Returns:
            De-extincted flux values
        """
        if not self.apply_deextinction or ebv == 0:
            return flux

        # Get effective wavelength for this filter
        eff_wavelength = np.array([self.filter_wavelengths[filter_name]])

        # Calculate extinction in magnitudes using Fitzpatrick99 law
        # R_V = 3.1 is the standard Milky Way value
        A_lambda = fitzpatrick99(eff_wavelength, ebv * 3.1)

        # Convert magnitude extinction to flux correction
        # De-extincted flux = observed flux * 10^(A_lambda / 2.5)
        flux_deextincted = flux * 10**(A_lambda[0] / 2.5)

        return flux_deextincted

    def extract_filter_features(self, times: np.ndarray, fluxes: np.ndarray,
                                flux_errs: np.ndarray, filter_name: str) -> Dict[str, float]:
        """
        Extract features for a single filter's lightcurve.

        Args:
            times: Observation times (MJD)
            fluxes: Flux measurements (microjansky)
            flux_errs: Flux uncertainties
            filter_name: Filter name (u, g, r, i, z, y)

        Returns:
            Dictionary of features with filter-specific names
        """
        features = {}
        prefix = f"{filter_name}_"

        # Handle empty or all-NaN data
        if len(times) == 0 or np.all(np.isnan(fluxes)):
            return self._get_empty_features(prefix)

        # Remove NaN entries
        mask = ~(np.isnan(times) | np.isnan(fluxes) | np.isnan(flux_errs))
        times = times[mask]
        fluxes = fluxes[mask]
        flux_errs = flux_errs[mask]

        if len(times) == 0:
            return self._get_empty_features(prefix)

        # Sort by time
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        fluxes = fluxes[sort_idx]
        flux_errs = flux_errs[sort_idx]

        # Basic counting
        features[f'{prefix}n_obs'] = len(times)

        # Statistical features
        features[f'{prefix}flux_mean'] = np.mean(fluxes)
        features[f'{prefix}flux_std'] = np.std(fluxes)
        features[f'{prefix}flux_median'] = np.median(fluxes)
        features[f'{prefix}flux_min'] = np.min(fluxes)
        features[f'{prefix}flux_max'] = np.max(fluxes)
        features[f'{prefix}flux_range'] = features[f'{prefix}flux_max'] - features[f'{prefix}flux_min']

        if len(fluxes) > 1:
            features[f'{prefix}flux_skew'] = stats.skew(fluxes)
            features[f'{prefix}flux_kurtosis'] = stats.kurtosis(fluxes)
        else:
            features[f'{prefix}flux_skew'] = 0
            features[f'{prefix}flux_kurtosis'] = 0

        # Percentiles
        features[f'{prefix}flux_p25'] = np.percentile(fluxes, 25)
        features[f'{prefix}flux_p75'] = np.percentile(fluxes, 75)
        features[f'{prefix}flux_iqr'] = features[f'{prefix}flux_p75'] - features[f'{prefix}flux_p25']

        # Temporal features
        features[f'{prefix}duration'] = times[-1] - times[0] if len(times) > 1 else 0

        if len(times) > 1:
            cadence = np.diff(times)
            features[f'{prefix}cadence_mean'] = np.mean(cadence)
            features[f'{prefix}cadence_std'] = np.std(cadence)
            features[f'{prefix}cadence_min'] = np.min(cadence)
            features[f'{prefix}cadence_max'] = np.max(cadence)
        else:
            features[f'{prefix}cadence_mean'] = 0
            features[f'{prefix}cadence_std'] = 0
            features[f'{prefix}cadence_min'] = 0
            features[f'{prefix}cadence_max'] = 0

        # Peak and rise/fall features
        peak_idx = np.argmax(fluxes)
        features[f'{prefix}peak_flux'] = fluxes[peak_idx]
        features[f'{prefix}peak_time'] = times[peak_idx]
        features[f'{prefix}peak_flux_rel'] = (times[peak_idx] - times[0]) / features[f'{prefix}duration'] if features[f'{prefix}duration'] > 0 else 0.5

        # Rise time (first to peak)
        if peak_idx > 0:
            features[f'{prefix}rise_time'] = times[peak_idx] - times[0]
            features[f'{prefix}rise_rate'] = (fluxes[peak_idx] - fluxes[0]) / features[f'{prefix}rise_time'] if features[f'{prefix}rise_time'] > 0 else 0
        else:
            features[f'{prefix}rise_time'] = 0
            features[f'{prefix}rise_rate'] = 0

        # Fall time (peak to last)
        if peak_idx < len(fluxes) - 1:
            features[f'{prefix}fall_time'] = times[-1] - times[peak_idx]
            features[f'{prefix}fall_rate'] = (fluxes[peak_idx] - fluxes[-1]) / features[f'{prefix}fall_time'] if features[f'{prefix}fall_time'] > 0 else 0
        else:
            features[f'{prefix}fall_time'] = 0
            features[f'{prefix}fall_rate'] = 0

        # Signal-to-noise
        snr = fluxes / flux_errs
        features[f'{prefix}snr_mean'] = np.mean(snr)
        features[f'{prefix}snr_median'] = np.median(snr)
        features[f'{prefix}snr_max'] = np.max(snr)

        # Variability features
        if features[f'{prefix}flux_mean'] != 0:
            features[f'{prefix}coef_var'] = features[f'{prefix}flux_std'] / abs(features[f'{prefix}flux_mean'])
        else:
            features[f'{prefix}coef_var'] = 0

        # Total absolute variation
        if len(fluxes) > 1:
            features[f'{prefix}total_var'] = np.sum(np.abs(np.diff(fluxes)))
        else:
            features[f'{prefix}total_var'] = 0

        return features

    def _get_empty_features(self, prefix: str) -> Dict[str, float]:
        """Return zero-filled features for missing data."""
        feature_names = [
            'n_obs', 'flux_mean', 'flux_std', 'flux_median', 'flux_min', 'flux_max',
            'flux_range', 'flux_skew', 'flux_kurtosis', 'flux_p25', 'flux_p75', 'flux_iqr',
            'duration', 'cadence_mean', 'cadence_std', 'cadence_min', 'cadence_max',
            'peak_flux', 'peak_time', 'peak_flux_rel', 'rise_time', 'rise_rate',
            'fall_time', 'fall_rate', 'snr_mean', 'snr_median', 'snr_max',
            'coef_var', 'total_var'
        ]
        return {f'{prefix}{name}': 0.0 for name in feature_names}

    def extract_object_features(self, lightcurve_df: pd.DataFrame,
                                metadata: pd.Series) -> Dict[str, float]:
        """
        Extract features for a single object.

        Args:
            lightcurve_df: DataFrame with columns [Time (MJD), Flux, Flux_err, Filter]
            metadata: Series with object metadata (Z, EBV, etc.)

        Returns:
            Dictionary of all features for this object
        """
        features = {}

        # Add metadata features
        features['Z'] = metadata.get('Z', 0)
        features['Z_err'] = metadata.get('Z_err', 0)
        ebv = metadata.get('EBV', 0)
        features['EBV'] = ebv

        # Extract features for each filter
        for filter_name in self.filters:
            filter_data = lightcurve_df[lightcurve_df['Filter'] == filter_name]

            if len(filter_data) > 0:
                times = filter_data['Time (MJD)'].values
                fluxes = filter_data['Flux'].values
                flux_errs = filter_data['Flux_err'].values

                # Apply de-extinction if enabled
                if self.apply_deextinction:
                    fluxes = self.deextinct_flux(fluxes, ebv, filter_name)
                    # Also correct flux errors
                    flux_errs = self.deextinct_flux(flux_errs, ebv, filter_name)

                filter_features = self.extract_filter_features(times, fluxes, flux_errs, filter_name)
                features.update(filter_features)
            else:
                # No data for this filter
                empty_features = self._get_empty_features(f"{filter_name}_")
                features.update(empty_features)

        # Cross-filter features (on de-extincted data if applicable)
        if self.apply_deextinction:
            # Create de-extincted copy of lightcurve for cross-filter features
            lc_deext = lightcurve_df.copy()
            for filter_name in self.filters:
                mask = lc_deext['Filter'] == filter_name
                if mask.any():
                    lc_deext.loc[mask, 'Flux'] = self.deextinct_flux(
                        lc_deext.loc[mask, 'Flux'].values, ebv, filter_name
                    )
            features.update(self._extract_cross_filter_features(lc_deext))
        else:
            features.update(self._extract_cross_filter_features(lightcurve_df))

        return features

    def _extract_cross_filter_features(self, lightcurve_df: pd.DataFrame) -> Dict[str, float]:
        """Extract features across multiple filters."""
        features = {}

        # Total number of observations across all filters
        features['total_n_obs'] = len(lightcurve_df)

        # Number of filters with data
        features['n_filters_with_data'] = lightcurve_df['Filter'].nunique()

        # Overall flux statistics
        if len(lightcurve_df) > 0:
            features['overall_flux_mean'] = lightcurve_df['Flux'].mean()
            features['overall_flux_std'] = lightcurve_df['Flux'].std()
            features['overall_flux_range'] = lightcurve_df['Flux'].max() - lightcurve_df['Flux'].min()

            # Overall duration
            features['overall_duration'] = lightcurve_df['Time (MJD)'].max() - lightcurve_df['Time (MJD)'].min()
        else:
            features['overall_flux_mean'] = 0
            features['overall_flux_std'] = 0
            features['overall_flux_range'] = 0
            features['overall_duration'] = 0

        return features

    def process_multiple_objects(self, log_df: pd.DataFrame,
                                 lightcurve_loader_func,
                                 is_training: bool = True) -> pd.DataFrame:
        """
        Process multiple objects and extract features.

        Args:
            log_df: DataFrame with object metadata (from train_log.csv or test_log.csv)
            lightcurve_loader_func: Function to load lightcurve for an object
                                   Signature: func(object_id, split_name, is_training) -> DataFrame
            is_training: Whether processing training or test data

        Returns:
            DataFrame with extracted features for all objects
        """
        print(f"\nExtracting features from {len(log_df)} objects...")

        all_features = []

        for idx, row in tqdm(log_df.iterrows(), total=len(log_df), desc="Processing objects"):
            try:
                object_id = row['object_id']
                split_name = row['split']

                # Load lightcurve
                lc_df = lightcurve_loader_func(object_id, split_name, is_training)

                # Extract features
                obj_features = self.extract_object_features(lc_df, row)
                obj_features['object_id'] = object_id

                # Add target if training
                if is_training and 'target' in row:
                    obj_features['target'] = row['target']

                all_features.append(obj_features)

            except Exception as e:
                print(f"\nWarning: Error processing {object_id}: {e}")
                # Create empty features for this object
                obj_features = {'object_id': object_id}
                if is_training and 'target' in row:
                    obj_features['target'] = row['target']
                all_features.append(obj_features)

        features_df = pd.DataFrame(all_features)

        # Fill any missing values with 0
        features_df = features_df.fillna(0)

        print(f"\nFeature extraction complete!")
        print(f"Features shape: {features_df.shape}")
        print(f"Number of features: {len(features_df.columns) - 1 - ('target' in features_df.columns)}")

        return features_df


if __name__ == "__main__":
    print("Feature Engineering module for MALLORN TDE Classification")
    print("\nThis module extracts features from lightcurve time series data.")
    print("\nFeatures extracted per filter:")
    print("  - Statistical: mean, std, median, skewness, kurtosis, percentiles")
    print("  - Temporal: duration, observation count, cadence statistics")
    print("  - Rise/Fall: peak flux, rise/fall times and rates")
    print("  - Signal quality: SNR, variability")
    print("  - Cross-filter: overall statistics")
    print("\nDe-extinction:")
    print("  - Applies Fitzpatrick99 extinction law to correct for galactic dust")
    print("  - Uses EBV coefficient from metadata")
    print("  - Filter-specific correction based on effective wavelengths")
    print(f"\nDe-extinction package available: {HAS_EXTINCTION}")
