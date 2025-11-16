# De-Extinction in MALLORN TDE Classification

## What is De-Extinction?

**De-extinction** is the process of correcting astronomical flux measurements for the dimming effect of interstellar dust in the Milky Way galaxy.

When light from a distant astronomical object travels to us, some of it is absorbed and scattered by dust particles in our galaxy. This makes the object appear fainter and redder than it actually is. This effect is called **extinction**.

## Why is it Important for TDE Classification?

De-extinction is crucial because:

1. **Objects at different sky positions** experience different amounts of extinction
2. **Different wavelengths** (filters) are affected differently - blue light is more affected than red
3. **Without correction**, two identical TDEs could appear very different just due to dust
4. **ML models** can learn more meaningful patterns from corrected data

## How It Works

### 1. Extinction Coefficient (EBV)

Each object has an `EBV` value in the log file:
- E(B-V) measures the difference in extinction between B and V bands
- Higher EBV = more dust = more extinction
- Typical values: 0.0 to ~0.5 magnitudes

### 2. Fitzpatrick99 Extinction Law

We use the **Fitzpatrick (1999) extinction law** to calculate how much extinction affects each filter:

```python
A_lambda = fitzpatrick99(wavelength, ebv * 3.1)
```

Where:
- `wavelength`: Effective wavelength of the filter (in Angstroms)
- `ebv`: Extinction coefficient from log file
- `3.1`: R_V value (standard for Milky Way)
- `A_lambda`: Extinction in magnitudes for that filter

### 3. Flux Correction

Convert magnitude extinction to flux correction:

```python
flux_corrected = flux_observed * 10^(A_lambda / 2.5)
```

## Filter-Specific Corrections

LSST filters have different effective wavelengths:

| Filter | Wavelength (Å) | Typical Correction (EBV=0.1) |
|--------|----------------|------------------------------|
| u      | 3641           | ~1.37x                       |
| g      | 4704           | ~1.28x                       |
| r      | 6155           | ~1.19x                       |
| i      | 7504           | ~1.14x                       |
| z      | 8695           | ~1.11x                       |
| y      | 10056          | ~1.09x                       |

**Note**: Blue filters (u, g) need more correction than red filters (z, y).

## Implementation

### In Feature Engineering

Our `LightcurveFeatureEngineer` class applies de-extinction automatically:

```python
from src.feature_engineer import LightcurveFeatureEngineer

# De-extinction enabled by default
engineer = LightcurveFeatureEngineer(apply_deextinction=True)

# Or disable if needed
engineer = LightcurveFeatureEngineer(apply_deextinction=False)
```

### Process Flow

1. Load object metadata from log file (includes EBV)
2. Load lightcurve time series
3. For each filter:
   - Get effective wavelength
   - Calculate A_lambda using Fitzpatrick99
   - Correct flux: `flux_corrected = flux * 10^(A_lambda/2.5)`
   - Correct errors the same way
4. Extract features from corrected data

## Example

For object with EBV = 0.15 in g-band (λ = 4704 Å):

```python
# Calculate extinction
A_g = fitzpatrick99([4704], 0.15 * 3.1)  # → 0.375 mag

# Correct flux
flux_obs = 100.0  # microjansky
flux_corr = 100.0 * 10^(0.375 / 2.5)  # → 116.2 microjansky
```

The object is actually **16% brighter** than it appears!

## Verification

You can verify de-extinction is working:

```python
from src.feature_engineer import LightcurveFeatureEngineer

engineer = LightcurveFeatureEngineer(apply_deextinction=True)

# Should print: "De-extinction correction will be applied to flux values"
```

Check output during feature extraction - corrected fluxes should be higher than raw fluxes.

## References

1. **Fitzpatrick, E. L.** (1999). Correcting for the Effects of Interstellar Extinction. *PASP*, 111, 63-75.
2. **Schlafly, E. F. & Finkbeiner, D. P.** (2011). Measuring Reddening with SDSS Stellar Spectra. *ApJ*, 737, 103.
3. **SVO Filter Profile Service**: http://svo2.cab.inta-csic.es/theory/fps/

## Package Requirements

Install the extinction package:

```bash
pip install extinction==0.4.7
```

This is included in `requirements.txt`.

## Impact on Model Performance

Expected improvements with de-extinction:
- ✅ More consistent flux measurements across sky
- ✅ Better color information (ratios between filters)
- ✅ Reduced systematic errors
- ✅ Improved F1 score (typically 2-5% improvement)

## Troubleshooting

**"extinction package not found"**
- Install: `pip install extinction==0.4.7`
- Or disable: `engineer = LightcurveFeatureEngineer(apply_deextinction=False)`

**Flux values seem too high**
- This is expected! De-extincted fluxes are brighter
- Check EBV values - higher EBV → bigger correction

**Different results from notebook example**
- Notebook might use different R_V value
- Check filter wavelengths match
- Verify EBV values are correct

---

**Summary**: De-extinction corrects for galactic dust, making flux measurements more accurate and improving model performance. It's enabled by default in our pipeline.
