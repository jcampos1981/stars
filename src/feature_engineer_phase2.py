"""
Phase 2 Feature Engineer: Temperature & Color Evolution Features
Extends Phase 1 with spectral/thermal features

New features focus on:
1. Color evolution (TDEs cool: UV strong → optical)
2. UV/optical ratios (TDEs have high UV initially)
3. Temperature evolution (blackbody cooling)
4. Multi-band color changes over time
"""

import numpy as np
import pandas as pd
from typing import Dict
from feature_engineer_phase1 import Phase1FeatureEngineer
import warnings
warnings.filterwarnings('ignore')


class Phase2FeatureEngineer(Phase1FeatureEngineer):
    """
    Extended feature engineer with Phase 1 + Phase 2 features.

    Phase 2 adds temperature and color evolution features on top of Phase 1.
    """

    def __init__(self, apply_deextinction: bool = True):
        super().__init__(apply_deextinction)
        print("Phase 2 Feature Engineer initialized")
        print("Adding temperature & color evolution features")

        # Color pairs for analysis
        self.color_pairs = [
            ('u', 'g'),  # UV to blue
            ('g', 'r'),  # Blue to red
            ('r', 'i'),  # Red to near-IR
            ('i', 'z'),  # Near-IR to IR
            ('u', 'r'),  # UV to red (large baseline)
            ('g', 'i'),  # Blue to near-IR
        ]

    def extract_object_features(self, lightcurve_df: pd.DataFrame,
                                metadata: pd.Series) -> Dict[str, float]:
        """
        Extract base + Phase 1 + Phase 2 features for an object.
        """
        # Get base + Phase 1 features
        features = super().extract_object_features(lightcurve_df, metadata)

        # Organize lightcurve data by filter
        filter_data = {}
        for filt in self.filters:
            mask = lightcurve_df['Filter'] == filt
            if mask.any():
                times = lightcurve_df.loc[mask, 'Time (MJD)'].values
                fluxes = lightcurve_df.loc[mask, 'Flux'].values
                flux_errs = lightcurve_df.loc[mask, 'Flux_err'].values

                # Remove NaNs and sort
                valid = ~(np.isnan(times) | np.isnan(fluxes) | np.isnan(flux_errs))
                if valid.any():
                    times = times[valid]
                    fluxes = fluxes[valid]
                    flux_errs = flux_errs[valid]

                    sort_idx = np.argsort(times)
                    filter_data[filt] = {
                        'times': times[sort_idx],
                        'fluxes': fluxes[sort_idx],
                        'flux_errs': flux_errs[sort_idx]
                    }

        # === PHASE 2 FEATURE 1: Color Indices at Different Epochs ===
        # Extract early, peak, and late colors

        for filt1, filt2 in self.color_pairs:
            if filt1 in filter_data and filt2 in filter_data:
                data1 = filter_data[filt1]
                data2 = filter_data[filt2]

                # Early color (first 20% of observations)
                n1_early = max(1, len(data1['fluxes']) // 5)
                n2_early = max(1, len(data2['fluxes']) // 5)

                flux1_early = np.mean(data1['fluxes'][:n1_early])
                flux2_early = np.mean(data2['fluxes'][:n2_early])

                if flux2_early > 0 and flux1_early > 0:
                    features[f'color_{filt1}{filt2}_early'] = -2.5 * np.log10(flux1_early / flux2_early)
                else:
                    features[f'color_{filt1}{filt2}_early'] = 0.0

                # Peak color (around maximum flux)
                peak_idx1 = np.argmax(data1['fluxes'])
                peak_idx2 = np.argmax(data2['fluxes'])

                flux1_peak = data1['fluxes'][peak_idx1]
                flux2_peak = data2['fluxes'][peak_idx2]

                if flux2_peak > 0 and flux1_peak > 0:
                    features[f'color_{filt1}{filt2}_peak'] = -2.5 * np.log10(flux1_peak / flux2_peak)
                else:
                    features[f'color_{filt1}{filt2}_peak'] = 0.0

                # Late color (last 20% of observations)
                n1_late = max(1, len(data1['fluxes']) // 5)
                n2_late = max(1, len(data2['fluxes']) // 5)

                flux1_late = np.mean(data1['fluxes'][-n1_late:])
                flux2_late = np.mean(data2['fluxes'][-n2_late:])

                if flux2_late > 0 and flux1_late > 0:
                    features[f'color_{filt1}{filt2}_late'] = -2.5 * np.log10(flux1_late / flux2_late)
                else:
                    features[f'color_{filt1}{filt2}_late'] = 0.0

                # Color evolution (late - early)
                # TDEs: should become redder (positive evolution) as they cool
                features[f'color_{filt1}{filt2}_evolution'] = (
                    features[f'color_{filt1}{filt2}_late'] -
                    features[f'color_{filt1}{filt2}_early']
                )

                # Color change rate (evolution per day)
                if filt1 in filter_data and filt2 in filter_data:
                    time_span = max(data1['times'][-1] - data1['times'][0],
                                   data2['times'][-1] - data2['times'][0])
                    if time_span > 0:
                        features[f'color_{filt1}{filt2}_rate'] = (
                            features[f'color_{filt1}{filt2}_evolution'] / time_span
                        )
                    else:
                        features[f'color_{filt1}{filt2}_rate'] = 0.0
                else:
                    features[f'color_{filt1}{filt2}_rate'] = 0.0
            else:
                # Missing data for this color
                features[f'color_{filt1}{filt2}_early'] = 0.0
                features[f'color_{filt1}{filt2}_peak'] = 0.0
                features[f'color_{filt1}{filt2}_late'] = 0.0
                features[f'color_{filt1}{filt2}_evolution'] = 0.0
                features[f'color_{filt1}{filt2}_rate'] = 0.0

        # === PHASE 2 FEATURE 2: UV/Optical Ratios ===
        # TDEs have strong UV emission initially

        if 'u' in filter_data and 'r' in filter_data:
            u_data = filter_data['u']
            r_data = filter_data['r']

            # UV/optical ratio at peak
            u_max = np.max(u_data['fluxes'])
            r_max = np.max(r_data['fluxes'])

            if r_max > 0:
                features['uv_optical_ratio_peak'] = u_max / r_max
            else:
                features['uv_optical_ratio_peak'] = 0.0

            # UV/optical ratio early vs late
            u_early = np.mean(u_data['fluxes'][:max(1, len(u_data['fluxes'])//5)])
            r_early = np.mean(r_data['fluxes'][:max(1, len(r_data['fluxes'])//5)])

            u_late = np.mean(u_data['fluxes'][-max(1, len(u_data['fluxes'])//5):])
            r_late = np.mean(r_data['fluxes'][-max(1, len(r_data['fluxes'])//5):])

            if r_early > 0:
                features['uv_optical_ratio_early'] = u_early / r_early
            else:
                features['uv_optical_ratio_early'] = 0.0

            if r_late > 0:
                features['uv_optical_ratio_late'] = u_late / r_late
            else:
                features['uv_optical_ratio_late'] = 0.0

            # Evolution of UV/optical ratio (should decrease as TDE cools)
            features['uv_optical_evolution'] = (
                features['uv_optical_ratio_late'] - features['uv_optical_ratio_early']
            )
        else:
            features['uv_optical_ratio_peak'] = 0.0
            features['uv_optical_ratio_early'] = 0.0
            features['uv_optical_ratio_late'] = 0.0
            features['uv_optical_evolution'] = 0.0

        # === PHASE 2 FEATURE 3: Simplified Temperature Indicators ===
        # Instead of full blackbody fit, use color-temperature proxies

        # u-g color as temperature proxy (bluer = hotter)
        if 'u' in filter_data and 'g' in filter_data:
            # Temperature evolution from u-g
            # Hotter objects are bluer (smaller u-g)
            ug_early = features.get('color_ug_early', 0)
            ug_late = features.get('color_ug_late', 0)

            # Positive evolution = getting redder = cooling (TDE signature)
            features['temp_proxy_ug_cooling'] = ug_late - ug_early
        else:
            features['temp_proxy_ug_cooling'] = 0.0

        # g-r color as another temperature proxy
        if 'g' in filter_data and 'r' in filter_data:
            gr_early = features.get('color_gr_early', 0)
            gr_late = features.get('color_gr_late', 0)

            features['temp_proxy_gr_cooling'] = gr_late - gr_early
        else:
            features['temp_proxy_gr_cooling'] = 0.0

        # === PHASE 2 FEATURE 4: Multi-color Consistency ===
        # TDEs should show consistent cooling across all colors

        color_evolutions = []
        for filt1, filt2 in self.color_pairs:
            key = f'color_{filt1}{filt2}_evolution'
            if key in features and features[key] != 0:
                color_evolutions.append(features[key])

        if len(color_evolutions) > 0:
            # Average color evolution (should be positive for cooling)
            features['avg_color_evolution'] = np.mean(color_evolutions)

            # Consistency of color evolution (low std = consistent cooling)
            if len(color_evolutions) > 1:
                features['color_evolution_consistency'] = np.std(color_evolutions)
            else:
                features['color_evolution_consistency'] = 0.0

            # Fraction of colors showing cooling (evolution > 0)
            cooling_count = sum(1 for ev in color_evolutions if ev > 0)
            features['fraction_colors_cooling'] = cooling_count / len(color_evolutions)
        else:
            features['avg_color_evolution'] = 0.0
            features['color_evolution_consistency'] = 0.0
            features['fraction_colors_cooling'] = 0.0

        # === PHASE 2 FEATURE 5: Spectral Energy Distribution Shape ===
        # Compare flux levels across filters at peak

        peak_fluxes = {}
        for filt in self.filters:
            if filt in filter_data:
                peak_fluxes[filt] = np.max(filter_data[filt]['fluxes'])

        if len(peak_fluxes) >= 3:
            # Ratio of blue to red fluxes at peak
            if 'u' in peak_fluxes and 'z' in peak_fluxes and peak_fluxes['z'] > 0:
                features['sed_blue_red_ratio'] = peak_fluxes['u'] / peak_fluxes['z']
            else:
                features['sed_blue_red_ratio'] = 0.0

            # Check if SED is peaked in UV/blue (TDE signature)
            flux_values = list(peak_fluxes.values())
            max_flux = max(flux_values)
            max_filter_idx = list(peak_fluxes.keys()).index(
                [k for k, v in peak_fluxes.items() if v == max_flux][0]
            )

            # Normalized position of peak (0=u, 1=y)
            features['sed_peak_position'] = max_filter_idx / (len(self.filters) - 1)
        else:
            features['sed_blue_red_ratio'] = 0.0
            features['sed_peak_position'] = 0.5

        # === PHASE 2 FEATURE 6: Color Variability ===
        # TDEs should show smooth color evolution, AGN are erratic

        for filt1, filt2 in [('u', 'g'), ('g', 'r'), ('r', 'i')]:
            if filt1 in filter_data and filt2 in filter_data:
                data1 = filter_data[filt1]
                data2 = filter_data[filt2]

                # Compute colors over time
                min_len = min(len(data1['fluxes']), len(data2['fluxes']))
                if min_len > 2:
                    colors = []
                    for i in range(min_len):
                        if data2['fluxes'][i] > 0 and data1['fluxes'][i] > 0:
                            color = -2.5 * np.log10(data1['fluxes'][i] / data2['fluxes'][i])
                            colors.append(color)

                    if len(colors) > 1:
                        # Variability in color (low for TDEs, high for AGN)
                        features[f'color_{filt1}{filt2}_variability'] = np.std(colors)
                    else:
                        features[f'color_{filt1}{filt2}_variability'] = 0.0
                else:
                    features[f'color_{filt1}{filt2}_variability'] = 0.0
            else:
                features[f'color_{filt1}{filt2}_variability'] = 0.0

        return features


if __name__ == "__main__":
    print("Phase 2 Feature Engineer")
    print("=" * 70)
    print("\nAdded Phase 2 features:")
    print("\nColor evolution features (30 per color pair × 6 pairs = 30):")
    print("  For each color pair (u-g, g-r, r-i, i-z, u-r, g-i):")
    print("    - color_early: Early epoch color")
    print("    - color_peak: Color at peak brightness")
    print("    - color_late: Late epoch color")
    print("    - color_evolution: Late - early (TDEs positive = cooling)")
    print("    - color_rate: Evolution per day")
    print("\nUV/Optical features (4):")
    print("  - uv_optical_ratio_peak: u/r at peak")
    print("  - uv_optical_ratio_early: u/r early")
    print("  - uv_optical_ratio_late: u/r late")
    print("  - uv_optical_evolution: Change in u/r")
    print("\nTemperature proxy features (2):")
    print("  - temp_proxy_ug_cooling: u-g evolution (cooling indicator)")
    print("  - temp_proxy_gr_cooling: g-r evolution")
    print("\nMulti-color consistency (3):")
    print("  - avg_color_evolution: Average evolution across colors")
    print("  - color_evolution_consistency: Std of evolutions")
    print("  - fraction_colors_cooling: Fraction showing cooling")
    print("\nSED shape features (2):")
    print("  - sed_blue_red_ratio: u/z flux ratio")
    print("  - sed_peak_position: Where SED peaks (0=UV, 1=IR)")
    print("\nColor variability (3):")
    print("  - color_ug_variability: Smoothness of u-g")
    print("  - color_gr_variability: Smoothness of g-r")
    print("  - color_ri_variability: Smoothness of r-i")
    print("\nTotal new Phase 2 features: 30 + 4 + 2 + 3 + 2 + 3 = 44 features")
    print("Combined with Phase 1 (~95) + Base (~90): ~230 total features")
    print("=" * 70)
