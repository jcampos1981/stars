"""
Hybrid Feature Engineering: Base features + Top 7 color features
Strategy: Best of both worlds with minimal overfitting risk
"""

import numpy as np
import pandas as pd
from typing import Dict
from feature_engineer import LightcurveFeatureEngineer


class HybridFeatureEngineer(LightcurveFeatureEngineer):
    """
    Extends base feature engineer with selective color features.

    Adds only the top 7 most important color features:
    1. color_u_g_peak
    2. color_u_g_evolution
    3. color_r_z_peak
    4. color_g_r_peak
    5. color_g_r_mean
    6. color_i_z_peak
    7. color_i_z_evolution
    """

    def __init__(self, apply_deextinction: bool = True):
        super().__init__(apply_deextinction)

        # Color pairs for selected features
        self.color_pairs = [
            ('u', 'g'),  # For u_g features
            ('g', 'r'),  # For g_r features
            ('r', 'i'),  # For r_i features (evolution only)
            ('i', 'z'),  # For i_z features
            ('r', 'z'),  # For r_z features
        ]

        print("Hybrid Feature Engineer initialized")
        print("Base features + Top 7 color features")

    def extract_color_features(self, filter_data):
        """
        Extract only the top 7 most important color features.

        Args:
            filter_data: Dict with keys as filter names and values as (times, fluxes, flux_errs)

        Returns:
            Dict of selected color features
        """
        color_features = {}

        # u-g features (peak, evolution)
        if 'u' in filter_data and 'g' in filter_data:
            times_u, fluxes_u, _ = filter_data['u']
            times_g, fluxes_g, _ = filter_data['g']

            if len(fluxes_u) > 0 and len(fluxes_g) > 0:
                # Peak color
                peak_u = np.max(fluxes_u)
                peak_g = np.max(fluxes_g)
                color_features['color_u_g_peak'] = peak_u / peak_g if peak_g > 0 else np.nan

                # Evolution
                if len(fluxes_u) >= 2 and len(fluxes_g) >= 2:
                    early_color = fluxes_u[0] / fluxes_g[0] if fluxes_g[0] > 0 else np.nan
                    late_color = fluxes_u[-1] / fluxes_g[-1] if fluxes_g[-1] > 0 else np.nan
                    if not np.isnan(early_color) and not np.isnan(late_color):
                        color_features['color_u_g_evolution'] = late_color - early_color
                    else:
                        color_features['color_u_g_evolution'] = np.nan
                else:
                    color_features['color_u_g_evolution'] = np.nan
            else:
                color_features['color_u_g_peak'] = np.nan
                color_features['color_u_g_evolution'] = np.nan
        else:
            color_features['color_u_g_peak'] = np.nan
            color_features['color_u_g_evolution'] = np.nan

        # g-r features (peak, mean)
        if 'g' in filter_data and 'r' in filter_data:
            times_g, fluxes_g, _ = filter_data['g']
            times_r, fluxes_r, _ = filter_data['r']

            if len(fluxes_g) > 0 and len(fluxes_r) > 0:
                # Peak color
                peak_g = np.max(fluxes_g)
                peak_r = np.max(fluxes_r)
                color_features['color_g_r_peak'] = peak_g / peak_r if peak_r > 0 else np.nan

                # Mean color
                mean_g = np.mean(fluxes_g)
                mean_r = np.mean(fluxes_r)
                color_features['color_g_r_mean'] = mean_g / mean_r if mean_r > 0 else np.nan
            else:
                color_features['color_g_r_peak'] = np.nan
                color_features['color_g_r_mean'] = np.nan
        else:
            color_features['color_g_r_peak'] = np.nan
            color_features['color_g_r_mean'] = np.nan

        # i-z features (peak, evolution)
        if 'i' in filter_data and 'z' in filter_data:
            times_i, fluxes_i, _ = filter_data['i']
            times_z, fluxes_z, _ = filter_data['z']

            if len(fluxes_i) > 0 and len(fluxes_z) > 0:
                # Peak color
                peak_i = np.max(fluxes_i)
                peak_z = np.max(fluxes_z)
                color_features['color_i_z_peak'] = peak_i / peak_z if peak_z > 0 else np.nan

                # Evolution
                if len(fluxes_i) >= 2 and len(fluxes_z) >= 2:
                    early_color = fluxes_i[0] / fluxes_z[0] if fluxes_z[0] > 0 else np.nan
                    late_color = fluxes_i[-1] / fluxes_z[-1] if fluxes_z[-1] > 0 else np.nan
                    if not np.isnan(early_color) and not np.isnan(late_color):
                        color_features['color_i_z_evolution'] = late_color - early_color
                    else:
                        color_features['color_i_z_evolution'] = np.nan
                else:
                    color_features['color_i_z_evolution'] = np.nan
            else:
                color_features['color_i_z_peak'] = np.nan
                color_features['color_i_z_evolution'] = np.nan
        else:
            color_features['color_i_z_peak'] = np.nan
            color_features['color_i_z_evolution'] = np.nan

        # r-z features (peak only)
        if 'r' in filter_data and 'z' in filter_data:
            times_r, fluxes_r, _ = filter_data['r']
            times_z, fluxes_z, _ = filter_data['z']

            if len(fluxes_r) > 0 and len(fluxes_z) > 0:
                peak_r = np.max(fluxes_r)
                peak_z = np.max(fluxes_z)
                color_features['color_r_z_peak'] = peak_r / peak_z if peak_z > 0 else np.nan
            else:
                color_features['color_r_z_peak'] = np.nan
        else:
            color_features['color_r_z_peak'] = np.nan

        return color_features

    def process_object(self, lightcurve_df: pd.DataFrame, metadata: Dict) -> pd.Series:
        """
        Process object with base features + selective color features.

        Args:
            lightcurve_df: DataFrame with lightcurve time series
            metadata: Dict with object metadata

        Returns:
            Series with all extracted features
        """
        # Get base features from parent class
        base_features = super().process_object(lightcurve_df, metadata)

        # Extract EBV for de-extinction
        ebv = metadata.get('EBV', 0)

        # Group by filter for color features
        filter_data = {}
        for filter_name in self.filters:
            filter_lc = lightcurve_df[lightcurve_df['filter'] == filter_name]

            if len(filter_lc) > 0:
                times = filter_lc['time'].values
                fluxes = filter_lc['flux'].values
                flux_errs = filter_lc['flux_err'].values

                # Apply de-extinction
                fluxes = self.deextinct_flux(fluxes, ebv, filter_name)
                flux_errs = self.deextinct_flux(flux_errs, ebv, filter_name)

                filter_data[filter_name] = (times, fluxes, flux_errs)
            else:
                filter_data[filter_name] = (np.array([]), np.array([]), np.array([]))

        # Extract color features
        color_features = self.extract_color_features(filter_data)

        # Combine base + color features
        all_features = base_features.to_dict()
        all_features.update(color_features)

        return pd.Series(all_features)
