"""
Phase 1 Feature Engineer: Temporal Evolution Features
Extends base features with TDE-specific temporal signatures

New features focus on:
1. Rise/decay asymmetry (TDEs have fast rise, slow decay)
2. Multi-band temporal coherence (TDEs evolve coherently)
3. Power law decay characteristics
4. Duration and event completeness
"""

import numpy as np
import pandas as pd
from typing import Dict
from feature_engineer import LightcurveFeatureEngineer
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')


class Phase1FeatureEngineer(LightcurveFeatureEngineer):
    """
    Extended feature engineer with Phase 1 temporal evolution features.

    Adds TDE-specific temporal signatures on top of base features.
    """

    def __init__(self, apply_deextinction: bool = True):
        super().__init__(apply_deextinction)
        print("Phase 1 Feature Engineer initialized")
        print("Adding temporal evolution features for TDE detection")

    def extract_filter_features(self, times: np.ndarray, fluxes: np.ndarray,
                                flux_errs: np.ndarray, filter_name: str) -> Dict[str, float]:
        """
        Extract base + Phase 1 features for a single filter.
        """
        # Get base features first
        features = super().extract_filter_features(times, fluxes, flux_errs, filter_name)

        # Add Phase 1 features
        prefix = f"{filter_name}_"

        # Handle empty data
        if len(times) == 0 or np.all(np.isnan(fluxes)):
            features.update(self._get_empty_phase1_features(prefix))
            return features

        # Remove NaN entries
        mask = ~(np.isnan(times) | np.isnan(fluxes) | np.isnan(flux_errs))
        times = times[mask]
        fluxes = fluxes[mask]
        flux_errs = flux_errs[mask]

        if len(times) < 3:
            features.update(self._get_empty_phase1_features(prefix))
            return features

        # Sort by time
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        fluxes = fluxes[sort_idx]
        flux_errs = flux_errs[sort_idx]

        # === PHASE 1 FEATURE 1: Rise/Decay Asymmetry ===
        # TDEs have fast rise + slow decay (asymmetry > 1)
        peak_idx = np.argmax(fluxes)

        rise_time = features.get(f'{prefix}rise_time', 0)
        fall_time = features.get(f'{prefix}fall_time', 0)

        if fall_time > 0 and rise_time > 0:
            # Asymmetry ratio: rise_time / fall_time
            # TDEs: < 0.5 (rise fast, decay slow)
            # Supernovae: ~1.0 (symmetric)
            features[f'{prefix}rise_fall_asymmetry'] = rise_time / fall_time
        else:
            features[f'{prefix}rise_fall_asymmetry'] = 1.0

        # Asymmetry in rates (inverse)
        rise_rate = features.get(f'{prefix}rise_rate', 0)
        fall_rate = features.get(f'{prefix}fall_rate', 0)

        if fall_rate > 0 and rise_rate > 0:
            # Rate asymmetry: rise_rate / fall_rate
            # TDEs: > 2 (rise fast, decay slow)
            features[f'{prefix}rise_fall_rate_ratio'] = rise_rate / fall_rate
        else:
            features[f'{prefix}rise_fall_rate_ratio'] = 1.0

        # === PHASE 1 FEATURE 2: Event Completeness ===
        # Did we capture the full event?

        # Number of observations before peak
        n_obs_rise = np.sum(times < times[peak_idx]) if peak_idx > 0 else 0
        features[f'{prefix}n_obs_rise'] = n_obs_rise

        # Number of observations after peak
        n_obs_decay = np.sum(times > times[peak_idx]) if peak_idx < len(times) - 1 else 0
        features[f'{prefix}n_obs_decay'] = n_obs_decay

        # Observation asymmetry (should correlate with time asymmetry)
        if n_obs_decay > 0 and n_obs_rise > 0:
            features[f'{prefix}obs_asymmetry'] = n_obs_rise / n_obs_decay
        else:
            features[f'{prefix}obs_asymmetry'] = 1.0

        # Peak captured quality
        # 1.0 if peak well-sampled, 0.0 if peak is at edge
        if len(times) > 2:
            peak_position = peak_idx / (len(times) - 1)
            # Penalty if peak is too early or too late
            features[f'{prefix}peak_captured_quality'] = 1.0 - abs(peak_position - 0.3)  # TDEs typically peak ~30% through
        else:
            features[f'{prefix}peak_captured_quality'] = 0.0

        # === PHASE 1 FEATURE 3: Power Law Decay Fit ===
        # TDEs decay as t^(-5/3) ~ t^(-1.67)
        # Try to fit decay phase to power law

        if n_obs_decay >= 3:
            # Get decay phase data
            decay_mask = times > times[peak_idx]
            t_decay = times[decay_mask] - times[peak_idx]  # Time since peak
            f_decay = fluxes[decay_mask]

            # Avoid negative or zero fluxes (can't take log)
            valid_decay = (f_decay > 0) & (t_decay > 0)

            if np.sum(valid_decay) >= 3:
                t_decay = t_decay[valid_decay]
                f_decay = f_decay[valid_decay]

                try:
                    # Fit: flux = A * t^alpha
                    # Log space: log(flux) = log(A) + alpha * log(t)
                    log_t = np.log(t_decay)
                    log_f = np.log(f_decay)

                    # Linear fit in log space
                    coeffs = np.polyfit(log_t, log_f, 1)
                    alpha = coeffs[0]  # Power law exponent

                    # TDEs: alpha ~ -5/3 ~ -1.67
                    features[f'{prefix}decay_power_law_alpha'] = alpha

                    # Distance from TDE value
                    features[f'{prefix}decay_alpha_tde_distance'] = abs(alpha + 1.67)

                    # R-squared of fit (goodness of fit)
                    predicted_log_f = coeffs[0] * log_t + coeffs[1]
                    ss_res = np.sum((log_f - predicted_log_f)**2)
                    ss_tot = np.sum((log_f - np.mean(log_f))**2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    features[f'{prefix}decay_power_law_r2'] = r_squared

                except:
                    features[f'{prefix}decay_power_law_alpha'] = 0.0
                    features[f'{prefix}decay_alpha_tde_distance'] = 999.0
                    features[f'{prefix}decay_power_law_r2'] = 0.0
            else:
                features[f'{prefix}decay_power_law_alpha'] = 0.0
                features[f'{prefix}decay_alpha_tde_distance'] = 999.0
                features[f'{prefix}decay_power_law_r2'] = 0.0
        else:
            features[f'{prefix}decay_power_law_alpha'] = 0.0
            features[f'{prefix}decay_alpha_tde_distance'] = 999.0
            features[f'{prefix}decay_power_law_r2'] = 0.0

        # === PHASE 1 FEATURE 4: Relative Durations ===
        # Fraction of event that is rise vs decay
        duration = features.get(f'{prefix}duration', 0)

        if duration > 0:
            features[f'{prefix}rise_fraction'] = rise_time / duration
            features[f'{prefix}decay_fraction'] = fall_time / duration
        else:
            features[f'{prefix}rise_fraction'] = 0.5
            features[f'{prefix}decay_fraction'] = 0.5

        # === PHASE 1 FEATURE 5: Brightness Evolution ===
        # How much does brightness change from start to peak to end?

        flux_start = fluxes[0]
        flux_peak = fluxes[peak_idx]
        flux_end = fluxes[-1]

        # Rise amplitude (normalized by peak)
        if flux_peak != 0:
            features[f'{prefix}rise_amplitude_norm'] = (flux_peak - flux_start) / abs(flux_peak)
        else:
            features[f'{prefix}rise_amplitude_norm'] = 0.0

        # Decay amplitude (normalized by peak)
        if flux_peak != 0:
            features[f'{prefix}decay_amplitude_norm'] = (flux_peak - flux_end) / abs(flux_peak)
        else:
            features[f'{prefix}decay_amplitude_norm'] = 0.0

        # Amplitude asymmetry
        if features[f'{prefix}decay_amplitude_norm'] > 0:
            features[f'{prefix}amplitude_asymmetry'] = features[f'{prefix}rise_amplitude_norm'] / features[f'{prefix}decay_amplitude_norm']
        else:
            features[f'{prefix}amplitude_asymmetry'] = 1.0

        return features

    def _get_empty_phase1_features(self, prefix: str) -> Dict[str, float]:
        """Return zero-filled Phase 1 features for missing data."""
        feature_names = [
            'rise_fall_asymmetry', 'rise_fall_rate_ratio',
            'n_obs_rise', 'n_obs_decay', 'obs_asymmetry', 'peak_captured_quality',
            'decay_power_law_alpha', 'decay_alpha_tde_distance', 'decay_power_law_r2',
            'rise_fraction', 'decay_fraction',
            'rise_amplitude_norm', 'decay_amplitude_norm', 'amplitude_asymmetry'
        ]
        return {f'{prefix}{name}': 0.0 for name in feature_names}

    def extract_object_features(self, lightcurve_df: pd.DataFrame,
                                metadata: pd.Series) -> Dict[str, float]:
        """
        Extract base + Phase 1 features for an object.
        """
        # Get base + per-filter Phase 1 features
        features = super().extract_object_features(lightcurve_df, metadata)

        # === CROSS-FILTER PHASE 1 FEATURES ===
        # Compare temporal evolution across filters

        # Time lags between peak times in different filters
        # TDEs: UV peaks before optical
        peak_times = {}
        for filt in self.filters:
            peak_time_key = f'{filt}_peak_time'
            if peak_time_key in features and features[peak_time_key] > 0:
                peak_times[filt] = features[peak_time_key]

        # u-r lag (UV vs red optical)
        if 'u' in peak_times and 'r' in peak_times:
            features['peak_time_lag_u_r'] = peak_times['u'] - peak_times['r']
        else:
            features['peak_time_lag_u_r'] = 0.0

        # g-i lag
        if 'g' in peak_times and 'i' in peak_times:
            features['peak_time_lag_g_i'] = peak_times['g'] - peak_times['i']
        else:
            features['peak_time_lag_g_i'] = 0.0

        # u-z lag (UV vs near-IR)
        if 'u' in peak_times and 'z' in peak_times:
            features['peak_time_lag_u_z'] = peak_times['u'] - peak_times['z']
        else:
            features['peak_time_lag_u_z'] = 0.0

        # === GLOBAL EVENT CHARACTERISTICS ===

        # Total event duration (max across all filters)
        durations = [features.get(f'{filt}_duration', 0) for filt in self.filters]
        features['global_duration'] = max(durations) if durations else 0

        # Average rise/fall asymmetry across all filters
        asymmetries = [features.get(f'{filt}_rise_fall_asymmetry', 1.0) for filt in self.filters]
        asymmetries = [a for a in asymmetries if a > 0]  # Remove zeros
        features['global_avg_asymmetry'] = np.mean(asymmetries) if asymmetries else 1.0
        features['global_std_asymmetry'] = np.std(asymmetries) if len(asymmetries) > 1 else 0.0

        # Average power law exponent across filters
        alphas = [features.get(f'{filt}_decay_power_law_alpha', 0) for filt in self.filters]
        alphas = [a for a in alphas if a != 0]  # Remove missing values
        features['global_avg_decay_alpha'] = np.mean(alphas) if alphas else 0.0
        features['global_std_decay_alpha'] = np.std(alphas) if len(alphas) > 1 else 0.0

        # How many filters have TDE-like alpha (close to -5/3)?
        tde_like_count = sum(1 for a in alphas if -2.5 < a < -1.0)
        features['n_filters_tde_like_decay'] = tde_like_count

        # Total number of observations across all filters
        n_obs_total = sum(features.get(f'{filt}_n_obs', 0) for filt in self.filters)
        features['global_n_obs'] = n_obs_total

        # Observation distribution (are observations evenly distributed?)
        n_obs_per_filter = [features.get(f'{filt}_n_obs', 0) for filt in self.filters]
        features['global_obs_std'] = np.std(n_obs_per_filter)

        return features


if __name__ == "__main__":
    print("Phase 1 Feature Engineer")
    print("=" * 70)
    print("\nAdded features:")
    print("\nPer-filter features (14 per filter):")
    print("  1. rise_fall_asymmetry - time ratio (TDEs < 0.5)")
    print("  2. rise_fall_rate_ratio - rate ratio (TDEs > 2)")
    print("  3. n_obs_rise - observations before peak")
    print("  4. n_obs_decay - observations after peak")
    print("  5. obs_asymmetry - observation ratio")
    print("  6. peak_captured_quality - how well peak is sampled")
    print("  7. decay_power_law_alpha - power law exponent (TDEs ~ -1.67)")
    print("  8. decay_alpha_tde_distance - distance from TDE value")
    print("  9. decay_power_law_r2 - goodness of power law fit")
    print(" 10. rise_fraction - rise time / total duration")
    print(" 11. decay_fraction - decay time / total duration")
    print(" 12. rise_amplitude_norm - normalized rise amplitude")
    print(" 13. decay_amplitude_norm - normalized decay amplitude")
    print(" 14. amplitude_asymmetry - amplitude ratio")
    print("\nCross-filter features (11):")
    print("  1. peak_time_lag_u_r - UV vs red lag")
    print("  2. peak_time_lag_g_i - green vs i-band lag")
    print("  3. peak_time_lag_u_z - UV vs near-IR lag")
    print("  4. global_duration - total event duration")
    print("  5. global_avg_asymmetry - average asymmetry")
    print("  6. global_std_asymmetry - asymmetry variability")
    print("  7. global_avg_decay_alpha - average power law alpha")
    print("  8. global_std_decay_alpha - alpha variability")
    print("  9. n_filters_tde_like_decay - # filters with TDE-like decay")
    print(" 10. global_n_obs - total observations")
    print(" 11. global_obs_std - observation distribution")
    print("\nTotal new features: 6 filters × 14 + 11 = 95 features")
    print("Combined with base features: ~185 total features")
    print("=" * 70)
