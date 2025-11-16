# Error Analysis - Key Insights for Reaching 0.60 Score

## 📊 Current Performance (CV)

**Score:** 0.47 (CV F1)
**Target:** 0.60
**Gap:** +0.13 (+27% improvement needed)

---

## 🔍 Error Distribution

### Confusion Matrix (Cross-Validation)
```
               Predicted
              Non-TDE  TDE
Actual Non-TDE  2808     87
       TDE        75     73
```

### Key Metrics
- **True Positives (TP):** 73/148 TDEs (49.3%) ✅ Correctamente identificados
- **False Negatives (FN):** 75/148 TDEs (50.7%) ❌ TDEs perdidos
- **True Negatives (TN):** 2808/2895 Non-TDEs (97.0%) ✅ Correctamente rechazados
- **False Positives (FP):** 87/2895 Non-TDEs (3.0%) ❌ Clasificados como TDEs

### 🎯 PROBLEMA PRINCIPAL

**Estamos perdiendo el 50.7% de los TDEs reales (75 de 148)**

Esto significa:
- **Recall muy bajo:** Solo encontramos la mitad de los TDEs
- **Precision moderada:** 46% de lo que clasificamos como TDE es correcto
- Para mejorar a 0.60, necesitamos **encontrar más TDEs sin generar muchos False Positives**

---

## 💡 Análisis de Probabilidades

### TDEs Correctos (TP)
- Probabilidad media: **0.70** (alta confianza)
- Mediana: 0.71
- Estos TDEs tienen señales claras

### TDEs Perdidos (FN) - ⚠️ CRÍTICO
- Probabilidad media: **0.14** (muy baja confianza)
- Mediana: 0.10
- **Problema:** El modelo no ve señales fuertes de TDE en estos objetos
- **Solución:** Necesitamos features que capturen características TDE más sutiles

### False Positives (FP)
- Probabilidad media: **0.62** (alta confianza)
- Estos Non-TDEs (principalmente AGN y SN Ia) se parecen mucho a TDEs
- Necesitamos features para diferenciarlos

---

## 🔬 Características de los TDEs Perdidos (FN) vs Encontrados (TP)

### Top 10 Diferencias Más Grandes

| Feature | FN (Perdidos) | TP (Encontrados) | Interpretación |
|---------|---------------|------------------|----------------|
| **g_coef_var** | 7.30 | 2.14 | FN son más variables/ruidosos en g |
| **u_coef_var** | 8.25 | 3.50 | FN más variables en u (UV) |
| **u_flux_p25** | 0.033 | -0.097 | FN son más brillantes en u |
| **u_flux_kurtosis** | 0.12 | 0.84 | FN tienen lightcurves más "normales" |
| **z_rise_rate** | 0.0078 | 0.0047 | FN suben más rápido en z |
| **y_fall_rate** | 0.0107 | 0.0065 | FN caen más rápido en y |
| **u_cadence_min** | 14.4 | 8.7 | FN tienen menos observaciones |
| **u_flux_median** | 0.39 | 0.25 | FN más brillantes en u |
| **u_flux_skew** | 0.45 | 0.86 | FN menos asimétricos |
| **EBV** | 0.061 | 0.040 | FN más extinción |

### 💡 Insights Clave sobre TDEs Perdidos

1. **Más variables/ruidosos** (alto coef_var en g, u)
2. **Rise/fall rates más rápidos** (no el perfil temporal típico de TDE)
3. **Menos observaciones** (u_cadence_min más alto)
4. **Lightcurves menos asimétricas** (u_flux_skew más bajo)
5. **Mayor extinción galáctica** (EBV más alto)

---

## 🔬 Características de False Positives (FP) vs True Negatives (TN)

### Top 10 Diferencias

| Feature | FP (Parecen TDE) | TN (Claramente No-TDE) | Interpretación |
|---------|------------------|------------------------|----------------|
| **u_flux_kurtosis** | 0.47 | -0.13 | FP tienen picos más pronunciados |
| **u_flux_skew** | 0.71 | 0.17 | FP más asimétricos (¡como TDEs!) |
| **u_coef_var** | 25.0 | 11.0 | FP muy variables |
| **g_flux_skew** | 1.63 | 0.73 | FP asimétricos en g |
| **g_flux_kurtosis** | 2.17 | 1.12 | FP con picos pronunciados en g |
| **y_fall_rate** | 0.029 | 0.015 | FP caen más lento |

### 💡 Insights sobre False Positives

Los **AGN** y **SN Ia** que confundimos con TDEs tienen:
1. **Alta asimetría** (como TDEs verdaderos)
2. **Picos pronunciados** en UV (u_flux_kurtosis alto)
3. **Alta variabilidad**
4. **Decay rates lentos**

**Problema:** Comparten características superficiales con TDEs, necesitamos features más específicas de TDEs.

---

## 🎯 RECOMENDACIONES DE NUEVAS FEATURES

### 1. Features de Evolución Temporal TDE-Específicas ⭐⭐⭐⭐⭐

TDEs tienen una evolución temporal muy característica:
- **Rise time:** ~2-4 semanas
- **Decay time:** ~months (t^{-5/3} power law)
- **Asimetría rise/decay:** Rise rápido, decay lento

**Nuevas features:**
```python
# Ratio de asimetría temporal
rise_decay_ratio = (peak_time - start_time) / (end_time - peak_time)
# TDEs típicos: ratio < 0.5 (rise rápido, decay lento)

# Ajuste a power law decay
# Fit lightcurve después del pico a: flux ∝ t^alpha
# TDEs típicos: alpha ≈ -5/3 ≈ -1.67

# Tiempo total del evento
duration = end_time - start_time
# TDEs: ~months a 1 año

# Rise rate vs decay rate en cada filtro
for filter in ['u', 'g', 'r', 'i', 'z']:
    rise_rate = max_flux / (time_at_max - time_at_start)
    decay_rate = (max_flux - final_flux) / (time_at_end - time_at_max)
    asymmetry = rise_rate / decay_rate
    # TDEs: asymmetry > 2 (rise más rápido que decay)
```

### 2. Features de Temperatura y Color ⭐⭐⭐⭐⭐

TDEs muestran evolución de temperatura característica:
- Temperatura inicial alta (UV brillante)
- Enfriamiento gradual con el tiempo

**Nuevas features:**
```python
# Evolución de color u-g
color_u_g_early = (u_flux_first / g_flux_first) if g_flux_first > 0 else nan
color_u_g_peak = (u_flux_max / g_flux_max) if g_flux_max > 0 else nan
color_u_g_late = (u_flux_last / g_flux_last) if g_flux_last > 0 else nan

# Evolución de color (enfriamiento)
color_evolution_u_g = color_u_g_late - color_u_g_early
# TDEs: negativo (se enfría, menos UV al final)

# Temperatura blackbody estimada
# Ajustar u, g, r a blackbody: flux ∝ nu^3 / (exp(h*nu/kT) - 1)
temp_peak = fit_blackbody_temp(u_peak, g_peak, r_peak)
temp_late = fit_blackbody_temp(u_late, g_late, r_late)
temp_evolution = temp_late - temp_peak
# TDEs: temp_evolution < 0 (se enfría)

# Ratio UV/óptico
uv_optical_ratio = u_flux_max / r_flux_max
# TDEs: típicamente alto (UV fuerte)
```

### 3. Features de Multi-banda Correlacionada ⭐⭐⭐⭐

TDEs muestran evolución coherente en múltiples bandas:

**Nuevas features:**
```python
# Diferencia en tiempos de pico entre bandas
# TDEs: pico en UV antes que en óptico
time_lag_u_r = time_at_max_u - time_at_max_r
# TDEs: negativo (u alcanza máximo antes)

# Correlación entre lightcurves
from scipy.stats import pearsonr
corr_u_r = pearsonr(u_flux_interpolated, r_flux_interpolated)[0]
corr_g_i = pearsonr(g_flux_interpolated, i_flux_interpolated)[0]
# TDEs: alta correlación (evolución coherente)

# Anchura del pico en diferentes bandas
peak_width_u = time_above_half_max(u_lightcurve)
peak_width_r = time_above_half_max(r_lightcurve)
width_ratio = peak_width_u / peak_width_r
```

### 4. Features de Regularidad vs Caos ⭐⭐⭐⭐

AGN son variables/caóticos, TDEs son eventos únicos suaves:

**Nuevas features:**
```python
# Número de picos secundarios
n_secondary_peaks = count_peaks_above_threshold(lightcurve, threshold=0.3*max_flux)
# TDEs: típicamente 0-1 (un solo evento)
# AGN: múltiples (variabilidad continua)

# Suavidad de la lightcurve
smoothness = np.mean(np.abs(np.diff(np.diff(flux))))
# TDEs: bajo (smooth)
# AGN: alto (variable)

# Autocorrelación
from statsmodels.tsa.stattools import acf
autocorr_lag1 = acf(flux, nlags=1)[1]
# TDEs: alta (correlación temporal fuerte)
# AGN: baja (variabilidad aleatoria)

# Entropía de la lightcurve
from scipy.stats import entropy
lc_entropy = entropy(flux_normalized)
# TDEs: baja (predecible)
# AGN: alta (caótica)
```

### 5. Features basadas en Redshift ⭐⭐⭐

TDEs a diferentes distancias se ven diferentes:

**Nuevas features:**
```python
# Luminosidad absoluta estimada
# distance_modulus = 5 * log10(luminosity_distance) + 25
# Con Z podemos estimar distancia
import astropy.cosmology as cosmo
lum_dist = cosmo.Planck15.luminosity_distance(Z).value  # Mpc

# Magnitud absoluta en cada banda
for filter in ['u', 'g', 'r', 'i', 'z']:
    app_mag = -2.5 * log10(flux_max)
    abs_mag = app_mag - 5 * log10(lum_dist) - 25
    # TDEs: típicamente M_g ≈ -20 a -22

# Rest-frame time scales
rest_frame_duration = observed_duration / (1 + Z)
rest_frame_rise_time = observed_rise_time / (1 + Z)
# Normaliza por time dilation cosmológica
```

### 6. Features de Cadencia y Calidad ⭐⭐⭐

Los TDEs perdidos tienen menos observaciones:

**Nuevas features:**
```python
# Número total de observaciones
n_obs_total = len(lightcurve)

# Número de observaciones en rise phase
n_obs_rise = len(lightcurve[lightcurve['time'] < time_at_max])

# Número de observaciones en decay phase
n_obs_decay = len(lightcurve[lightcurve['time'] > time_at_max])

# Ratio rise/decay observations
obs_asymmetry = n_obs_rise / n_obs_decay

# SNR promedio en cada banda
for filter in ['u', 'g', 'r', 'i', 'z']:
    avg_snr = np.mean(flux / flux_err)

# ¿Se capturó el pico?
peak_captured = (n_obs_rise > 2) and (n_obs_decay > 2)
```

---

## 📋 Plan de Implementación Priorizado

### **Fase 1: Features de Evolución Temporal** (Impacto Esperado: +0.05-0.08)
1. Rise/decay asymmetry ratio
2. Power law decay fit (alpha parameter)
3. Duration of event
4. Time lags between filters

**Por qué:** Los TDEs perdidos tienen rise/fall rates diferentes. Estas features los capturarán.

### **Fase 2: Features de Temperatura/Color** (Impacto Esperado: +0.03-0.06)
1. UV/optical ratio
2. Color evolution (u-g, g-r)
3. Blackbody temperature evolution

**Por qué:** Los False Positives (AGN, SN Ia) tienen evolución de color diferente a TDEs.

### **Fase 3: Features de Multi-banda** (Impacto Esperado: +0.02-0.04)
1. Peak time lags between filters
2. Cross-band correlations
3. Peak width ratios

**Por qué:** Discrimina entre TDEs (evolución coherente) y AGN (variabilidad independiente).

### **Fase 4: Features de Regularidad** (Impacto Esperado: +0.02-0.04)
1. Number of secondary peaks
2. Lightcurve smoothness
3. Autocorrelation
4. Entropy

**Por qué:** Separa TDEs (eventos únicos suaves) de AGN (variables caóticos).

---

## 🎯 Meta Realista

Con estas features nuevas:

| Fase | Features Añadidas | F1 Score Esperado | Mejora |
|------|-------------------|-------------------|--------|
| Actual | 92 | 0.47 | - |
| Fase 1 | +8-10 temporal | 0.52-0.55 | +0.05-0.08 |
| Fase 2 | +6-8 color/temp | 0.55-0.58 | +0.03-0.06 |
| Fase 3 | +4-6 multi-banda | 0.57-0.60 | +0.02-0.04 |
| Fase 4 | +4-5 regularidad | 0.59-0.62 | +0.02-0.04 |

**Estimación final:** 0.58-0.62 (alcanzar 0.60 es viable!)

---

## 📁 Archivos Generados

- `data/processed/error_analysis_cv.csv` - Análisis completo con todas las features
- `data/processed/false_positives_cv.csv` - 87 Non-TDEs mal clasificados
- `data/processed/false_negatives_cv.csv` - 75 TDEs perdidos

---

## ⚡ Siguiente Paso Inmediato

**Implementar Fase 1:** Features de Evolución Temporal

Estas features tienen el mayor impacto potencial porque:
1. Atacan directamente el problema de TDEs perdidos
2. Los TDEs perdidos tienen rise/fall rates muy diferentes
3. Son features basadas en física (no empíricas)
4. Discriminan TDEs de AGN/SN

**¿Quieres que implemente las features de Fase 1?**
