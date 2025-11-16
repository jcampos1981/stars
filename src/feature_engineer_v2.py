"""
Enhanced Feature Engineering for MALLORN TDE Classification Challenge
Version 2: With advanced color, temporal, and cross-filter features
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


class LightcurveFeatureEngineerV2:
    """
    Enhanced feature extraction from astronomical lightcurve time series.

    New features in V2:
    - Color features (flux ratios between filters)
    - Color evolution (temporal changes in colors)
    - Advanced statistical features
    - Cross-filter correlation features
    - Improved rise/fall characterization
    """

    def __init__(self, apply_deextinction: bool = True):
        self.filters = ['u', 'g', 'r', 'i', 'z', 'y']
        self.feature_names = []
        self.apply_deextinction = apply_deextinction and HAS_EXTINCTION

        # Effective wavelengths for each LSST filter (in Angstroms)
        self.filter_wavelengths = {
            'u': 3641,
            'g': 4704,
            'r': 6155,
            'i': 7504,
            'z': 8695,
            'y': 10056
        }

        # Color pairs for color features
        self.color_pairs = [
            ('u', 'g'),
            ('g', 'r'),
            ('r', 'i'),
            ('i', 'z'),
            ('z', 'y'),
            ('g', 'i'),  # Wide baseline
            ('r', 'z'),  # Wide baseline
        ]

        if self.apply_deextinction:
            print("De-extinction correction will be applied to flux values")
        else:
            print("WARNING: De-extinction correction is DISABLED")

    def deextinct_flux(self, flux: np.ndarray, ebv: float, filter_name: str) -> np.ndarray:
        """Apply de-extinction correction using Fitzpatrick99 law"""
        if not self.apply_deextinction or ebv == 0:
            return flux

        # Get effective wavelength for this filter
        eff_wavelength = np.array([self.filter_wavelengths[filter_name]])

        # Calculate extinction (A_lambda) using Fitzpatrick99 law
        # Rv = 3.1 (standard value for Milky Way)
        A_lambda = fitzpatrick99(eff_wavelength, ebv * 3.1)

        # Correct flux: flux_corrected = flux_observed * 10^(A_lambda/2.5)
        flux_deextincted = flux * 10**(A_lambda[0] / 2.5)

        return flux_deextincted

    def extract_filter_features(self, times, fluxes, flux_errs, filter_name):
        """Extract comprehensive features from a single filter's lightcurve"""
        features = {}

        if len(fluxes) == 0:
            # Return NaN features if no data
            return {f'{filter_name}_{key}': np.nan for key in [
                'flux_mean', 'flux_std', 'flux_median', 'flux_min', 'flux_max',
                'flux_range', 'flux_skew', 'flux_kurt', 'flux_p25', 'flux_p75',
                'n_obs', 'duration', 'cadence_mean', 'cadence_std', 'cadence_min', 'cadence_max',
                'peak_flux', 'peak_time', 'rise_time', 'fall_time', 'rise_rate', 'fall_rate',
                'snr_mean', 'snr_median', 'snr_max', 'coef_var', 'total_var',
                'flux_iqr', 'flux_mad', 'flux_above_median', 'flux_below_median',
                'early_mean', 'late_mean', 'mean_change'
            ]}

        # === Statistical features ===
        features[f'{filter_name}_flux_mean'] = np.mean(fluxes)
        features[f'{filter_name}_flux_std'] = np.std(fluxes)
        features[f'{filter_name}_flux_median'] = np.median(fluxes)
        features[f'{filter_name}_flux_min'] = np.min(fluxes)
        features[f'{filter_name}_flux_max'] = np.max(fluxes)
        features[f'{filter_name}_flux_range'] = np.max(fluxes) - np.min(fluxes)

        if len(fluxes) >= 3:
            features[f'{filter_name}_flux_skew'] = stats.skew(fluxes)
            features[f'{filter_name}_flux_kurt'] = stats.kurtosis(fluxes)
        else:
            features[f'{filter_name}_flux_skew'] = 0
            features[f'{filter_name}_flux_kurt'] = 0

        features[f'{filter_name}_flux_p25'] = np.percentile(fluxes, 25)
        features[f'{filter_name}_flux_p75'] = np.percentile(fluxes, 75)

        # NEW: Additional statistical features
        features[f'{filter_name}_flux_iqr'] = np.percentile(fluxes, 75) - np.percentile(fluxes, 25)
        features[f'{filter_name}_flux_mad'] = np.median(np.abs(fluxes - np.median(fluxes)))
        median_flux = np.median(fluxes)
        features[f'{filter_name}_flux_above_median'] = np.sum(fluxes > median_flux) / len(fluxes)
        features[f'{filter_name}_flux_below_median'] = np.sum(fluxes < median_flux) / len(fluxes)

        # === Temporal features ===
        features[f'{filter_name}_n_obs'] = len(times)
        features[f'{filter_name}_duration'] = np.max(times) - np.min(times)

        if len(times) > 1:
            time_diffs = np.diff(sorted(times))
            features[f'{filter_name}_cadence_mean'] = np.mean(time_diffs)
            features[f'{filter_name}_cadence_std'] = np.std(time_diffs)
            features[f'{filter_name}_cadence_min'] = np.min(time_diffs)
            features[f'{filter_name}_cadence_max'] = np.max(time_diffs)
        else:
            features[f'{filter_name}_cadence_mean'] = 0
            features[f'{filter_name}_cadence_std'] = 0
            features[f'{filter_name}_cadence_min'] = 0
            features[f'{filter_name}_cadence_max'] = 0

        # === Rise/Fall characteristics ===
        peak_idx = np.argmax(fluxes)
        features[f'{filter_name}_peak_flux'] = fluxes[peak_idx]
        features[f'{filter_name}_peak_time'] = times[peak_idx]

        if peak_idx > 0:
            rise_time = times[peak_idx] - times[0]
            rise_flux = fluxes[peak_idx] - fluxes[0]
            features[f'{filter_name}_rise_time'] = rise_time
            features[f'{filter_name}_rise_rate'] = rise_flux / rise_time if rise_time > 0 else 0
        else:
            features[f'{filter_name}_rise_time'] = 0
            features[f'{filter_name}_rise_rate'] = 0

        if peak_idx < len(times) - 1:
            fall_time = times[-1] - times[peak_idx]
            fall_flux = fluxes[peak_idx] - fluxes[-1]
            features[f'{filter_name}_fall_time'] = fall_time
            features[f'{filter_name}_fall_rate'] = fall_flux / fall_time if fall_time > 0 else 0
        else:
            features[f'{filter_name}_fall_time'] = 0
            features[f'{filter_name}_fall_rate'] = 0

        # === Signal-to-noise features ===
        snr = fluxes / flux_errs
        features[f'{filter_name}_snr_mean'] = np.mean(snr)
        features[f'{filter_name}_snr_median'] = np.median(snr)
        features[f'{filter_name}_snr_max'] = np.max(snr)

        # === Variability features ===
        mean_flux = np.mean(fluxes)
        features[f'{filter_name}_coef_var'] = np.std(fluxes) / mean_flux if mean_flux != 0 else 0
        features[f'{filter_name}_total_var'] = np.sum(np.abs(np.diff(fluxes)))

        # NEW: Early vs late evolution
        n_half = len(fluxes) // 2
        if n_half > 0:
            features[f'{filter_name}_early_mean'] = np.mean(fluxes[:n_half])
            features[f'{filter_name}_late_mean'] = np.mean(fluxes[n_half:])
            features[f'{filter_name}_mean_change'] = features[f'{filter_name}_late_mean'] - features[f'{filter_name}_early_mean']
        else:
            features[f'{filter_name}_early_mean'] = mean_flux
            features[f'{filter_name}_late_mean'] = mean_flux
            features[f'{filter_name}_mean_change'] = 0

        return features

    def extract_color_features(self, filter_data):
        """
        Extract color features (flux ratios) between filter pairs.

        Args:
            filter_data: Dict with keys as filter names and values as (times, fluxes, flux_errs)

        Returns:
            Dict of color features
        """
        color_features = {}

        for filter1, filter2 in self.color_pairs:
            if filter1 in filter_data and filter2 in filter_data:
                times1, fluxes1, _ = filter_data[filter1]
                times2, fluxes2, _ = filter_data[filter2]

                if len(fluxes1) > 0 and len(fluxes2) > 0:
                    # Mean color
                    mean_flux1 = np.mean(fluxes1)
                    mean_flux2 = np.mean(fluxes2)
                    if mean_flux2 > 0:
                        color_features[f'color_{filter1}_{filter2}_mean'] = mean_flux1 / mean_flux2
                    else:
                        color_features[f'color_{filter1}_{filter2}_mean'] = np.nan

                    # Peak color
                    peak_flux1 = np.max(fluxes1)
                    peak_flux2 = np.max(fluxes2)
                    if peak_flux2 > 0:
                        color_features[f'color_{filter1}_{filter2}_peak'] = peak_flux1 / peak_flux2
                    else:
                        color_features[f'color_{filter1}_{filter2}_peak'] = np.nan

                    # Color evolution (change over time)
                    if len(fluxes1) >= 2 and len(fluxes2) >= 2:
                        early_color = fluxes1[0] / fluxes2[0] if fluxes2[0] > 0 else np.nan
                        late_color = fluxes1[-1] / fluxes2[-1] if fluxes2[-1] > 0 else np.nan
                        if not np.isnan(early_color) and not np.isnan(late_color):
                            color_features[f'color_{filter1}_{filter2}_evolution'] = late_color - early_color
                        else:
                            color_features[f'color_{filter1}_{filter2}_evolution'] = np.nan
                    else:
                        color_features[f'color_{filter1}_{filter2}_evolution'] = np.nan
                else:
                    color_features[f'color_{filter1}_{filter2}_mean'] = np.nan
                    color_features[f'color_{filter1}_{filter2}_peak'] = np.nan
                    color_features[f'color_{filter1}_{filter2}_evolution'] = np.nan
            else:
                color_features[f'color_{filter1}_{filter2}_mean'] = np.nan
                color_features[f'color_{filter1}_{filter2}_peak'] = np.nan
                color_features[f'color_{filter1}_{filter2}_evolution'] = np.nan

        return color_features

    def extract_cross_filter_features(self, filter_data):
        """
        Extract features that aggregate information across all filters.

        Args:
            filter_data: Dict with keys as filter names and values as (times, fluxes, flux_errs)

        Returns:
            Dict of cross-filter features
        """
        cross_features = {}

        # Total observations across all filters
        total_obs = sum(len(data[0]) for data in filter_data.values())
        cross_features['total_n_obs'] = total_obs

        # Filters with observations
        n_filters_with_data = sum(1 for data in filter_data.values() if len(data[0]) > 0)
        cross_features['n_filters_with_data'] = n_filters_with_data

        # Peak flux across all filters
        all_peak_fluxes = []
        for times, fluxes, _ in filter_data.values():
            if len(fluxes) > 0:
                all_peak_fluxes.append(np.max(fluxes))

        if len(all_peak_fluxes) > 0:
            cross_features['cross_peak_flux_max'] = np.max(all_peak_fluxes)
            cross_features['cross_peak_flux_mean'] = np.mean(all_peak_fluxes)
            cross_features['cross_peak_flux_std'] = np.std(all_peak_fluxes)
        else:
            cross_features['cross_peak_flux_max'] = np.nan
            cross_features['cross_peak_flux_mean'] = np.nan
            cross_features['cross_peak_flux_std'] = np.nan

        # Time span across all filters
        all_times = np.concatenate([data[0] for data in filter_data.values() if len(data[0]) > 0])
        if len(all_times) > 0:
            cross_features['cross_time_span'] = np.max(all_times) - np.min(all_times)
            cross_features['cross_first_obs'] = np.min(all_times)
            cross_features['cross_last_obs'] = np.max(all_times)
        else:
            cross_features['cross_time_span'] = np.nan
            cross_features['cross_first_obs'] = np.nan
            cross_features['cross_last_obs'] = np.nan

        # NEW: Cross-filter variability correlation
        # Check if different filters show correlated variability
        if len(filter_data) >= 2:
            variability_list = []
            for times, fluxes, _ in filter_data.values():
                if len(fluxes) > 1:
                    variability = np.std(fluxes) / np.mean(fluxes) if np.mean(fluxes) > 0 else 0
                    variability_list.append(variability)

            if len(variability_list) > 1:
                cross_features['cross_var_std'] = np.std(variability_list)
                cross_features['cross_var_mean'] = np.mean(variability_list)
            else:
                cross_features['cross_var_std'] = np.nan
                cross_features['cross_var_mean'] = np.nan
        else:
            cross_features['cross_var_std'] = np.nan
            cross_features['cross_var_mean'] = np.nan

        return cross_features

    def process_object(self, lightcurve_df: pd.DataFrame, metadata: Dict) -> pd.Series:
        """
        Process a single object's lightcurve and extract all features.

        Args:
            lightcurve_df: DataFrame with columns [Time (MJD), Flux, Flux_err, Filter]
            metadata: Dict with object metadata (Z, EBV, etc.)

        Returns:
            Series with all extracted features
        """
        features = {}

        # Normalize column names
        lightcurve_df = lightcurve_df.rename(columns={
            'Time (MJD)': 'time',
            'Flux': 'flux',
            'Flux_err': 'flux_err',
            'Filter': 'filter'
        })

        # Extract EBV for de-extinction
        ebv = metadata.get('EBV', 0)

        # Group by filter
        filter_data = {}
        for filter_name in self.filters:
            filter_lc = lightcurve_df[lightcurve_df['filter'] == filter_name]

            if len(filter_lc) > 0:
                times = filter_lc['time'].values
                fluxes = filter_lc['flux'].values
                flux_errs = filter_lc['flux_err'].values

                # Apply de-extinction correction
                fluxes = self.deextinct_flux(fluxes, ebv, filter_name)
                flux_errs = self.deextinct_flux(flux_errs, ebv, filter_name)

                filter_data[filter_name] = (times, fluxes, flux_errs)

                # Extract per-filter features
                filter_features = self.extract_filter_features(times, fluxes, flux_errs, filter_name)
                features.update(filter_features)
            else:
                # No data for this filter
                filter_data[filter_name] = (np.array([]), np.array([]), np.array([]))
                empty_features = self.extract_filter_features(
                    np.array([]), np.array([]), np.array([]), filter_name
                )
                features.update(empty_features)

        # Extract color features
        color_features = self.extract_color_features(filter_data)
        features.update(color_features)

        # Extract cross-filter features
        cross_features = self.extract_cross_filter_features(filter_data)
        features.update(cross_features)

        # Add metadata features
        features['Z'] = metadata.get('Z', np.nan)
        features['EBV'] = ebv

        return pd.Series(features)

    def process_multiple_objects(self,
                                 log_df: pd.DataFrame,
                                 lightcurve_loader_func,
                                 is_training: bool = True) -> pd.DataFrame:
        """
        Process multiple objects and extract features.

        Args:
            log_df: DataFrame with object metadata
            lightcurve_loader_func: Function that loads lightcurve for an object
            is_training: Whether this is training data

        Returns:
            DataFrame with all features
        """
        all_features = []

        print(f"Processing {len(log_df)} objects...")
        for idx, row in tqdm(log_df.iterrows(), total=len(log_df), desc="Extracting features"):
            object_id = row['object_id']
            split_name = row['split']

            # Load lightcurve
            lc_df = lightcurve_loader_func(object_id, split_name, is_training)

            # Extract metadata
            metadata = {
                'Z': row.get('Z', np.nan),
                'EBV': row.get('EBV', 0)
            }

            # Extract features
            obj_features = self.process_object(lc_df, metadata)
            obj_features['object_id'] = object_id

            # Add target if training
            if is_training and 'is_tde' in row:
                obj_features['is_tde'] = row['is_tde']

            all_features.append(obj_features)

        features_df = pd.DataFrame(all_features)

        # Reorder columns
        first_cols = ['object_id']
        if 'is_tde' in features_df.columns:
            first_cols.append('is_tde')

        other_cols = [col for col in features_df.columns if col not in first_cols]
        features_df = features_df[first_cols + other_cols]

        return features_df
