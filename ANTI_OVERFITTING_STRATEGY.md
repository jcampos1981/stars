# Estrategia Anti-Overfitting - MALLORN TDE Classification

## 📉 Problema Detectado

**Modelo Optimizado (V1):**
- Validation F1: **0.5455** ✅
- Competition Test F1: **0.4158** ❌
- **Gap:** -0.1297 (overfitting severo)

El modelo estaba **sobreajustado** al conjunto de validación, con muchas features complejas que no generalizaban bien.

---

## 🛡️ Solución Implementada: Modelo Robusto

### Estrategia de 5 Puntos

#### 1. **Features Más Simples**
- ❌ No usar features V2 (color evolution, advanced stats)
- ✅ Usar features originales más simples
- **Resultado:** 183 features base (vs 237 en V2)

#### 2. **Feature Selection Agresiva**
- Entrenar modelo simple (max_depth=3)
- Seleccionar solo features con importancia > mediana
- **Resultado:** 92 features seleccionadas (50% reducción)

**Top 10 features más importantes:**
```
1. g_flux_skew       0.0438
2. r_flux_skew       0.0395
3. r_flux_range      0.0277
4. g_fall_rate       0.0268
5. r_coef_var        0.0221
6. u_flux_median     0.0220
7. u_snr_max         0.0217
8. r_cadence_max     0.0206
9. u_flux_mean       0.0204
10. r_flux_p25       0.0202
```

#### 3. **Regularización Fuerte**
```python
XGBClassifier(
    max_depth=4,              # Shallow trees
    learning_rate=0.05,       # Slow learning
    n_estimators=200,         # Fewer trees
    min_child_weight=5,       # Min samples per leaf
    gamma=0.2,                # Pruning threshold
    subsample=0.7,            # 70% data sampling
    colsample_bytree=0.7,     # 70% feature sampling
    reg_alpha=0.5,            # L1 regularization
    reg_lambda=1.0            # L2 regularization
)
```

#### 4. **Cross-Validation Robusto**
- **10-fold Stratified CV** (antes era solo 3-fold)
- Resultados por fold:
  ```
  Fold 1:  0.3684
  Fold 2:  0.5185
  Fold 3:  0.5714
  Fold 4:  0.4103
  Fold 5:  0.3571
  Fold 6:  0.3448
  Fold 7:  0.6154
  Fold 8:  0.4118
  Fold 9:  0.4706
  Fold 10: 0.5000
  ──────────────
  Mean:    0.4568 ± 0.0887
  ```

#### 5. **No Threshold Optimization**
- ❌ No optimizar threshold (causa overfitting)
- ✅ Usar threshold=0.5 por defecto
- Menos agresivo pero más robusto

---

## 📊 Resultados Esperados

### Modelo Optimizado (V1) - Con Overfitting
```
Features:        237 (54 nuevas complejas)
CV F1:           0.5455 (optimista)
Test F1:         0.4158 (real)
Threshold:       0.4412 (optimizado)
TDEs predichos:  450 (6.31%)
```

### Modelo Robusto (V2) - Anti-Overfitting
```
Features:        92 (seleccionadas)
CV F1:           0.4568 ± 0.0887 (realista)
Test F1:         ??? (debería ser ~0.45-0.46)
Threshold:       0.5000 (default)
TDEs predichos:  439 (6.15%)
```

**Expectativa:** El modelo robusto debería tener un score de test entre **0.43-0.48**, mucho más cercano al CV score.

---

## 🎯 ¿Por Qué Este Modelo Debería Funcionar Mejor?

### Signos de Mejor Generalización:

1. **CV más realista:**
   - CV F1 = 0.4568
   - Train F1 = 0.8433
   - Gap = 0.3865 (mejor que antes, aunque todavía hay overfitting)

2. **Menos features = menos ruido:**
   - 92 vs 237 features
   - Solo las más importantes

3. **Regularización fuerte:**
   - Penaliza complejidad
   - Previene memorización

4. **10-fold CV:**
   - Más splits = evaluación más robusta
   - Menor varianza en estimación

5. **Threshold conservador:**
   - No optimizado = menos overfitting
   - 0.5 es más generalizable

---

## 📁 Archivos Disponibles en GitHub

**Nuevo modelo robusto:**
- `submission.csv` - Nueva predicción (439 TDEs)
- `src/train_robust.py` - Training con anti-overfitting
- `src/predict_robust.py` - Predicción robusta
- `models/feature_names_robust.txt` - 92 features seleccionadas

**Modelo anterior (para comparación):**
- `src/train_optimized.py` - Modelo con overfitting
- `src/feature_engineer_v2.py` - Features complejas

---

## 🔄 Próximos Pasos Si Todavía No Funciona

Si el score de test sigue siendo bajo (<0.43), considerar:

### Opción 1: Aún Más Conservador
- Reducir más features (top 50-70)
- Aumentar regularización (gamma=0.5, lambda=2.0)
- max_depth=3 (trees más simples)

### Opción 2: Modelo Más Simple
- Usar Logistic Regression con regularización
- Random Forest con árboles muy shallow
- Ensemble de modelos simples

### Opción 3: Análisis de Datos
- Revisar si hay data leakage
- Validar que train/test tienen misma distribución
- Analizar objetos mal clasificados

### Opción 4: Feature Engineering Diferente
- Solo usar features básicas (mean, std, max, min)
- Features basadas en domain knowledge de astronomía
- Evitar features derivadas complejas

---

## 🚀 Cómo Probar el Nuevo Modelo

El archivo `submission.csv` ya está actualizado con el modelo robusto.

Puedes descargarlo desde:
```
https://github.com/jcampos1981/stars/blob/claude/python-work-01F1ahgmFV5Ce9rjgwrXmsWS/submission.csv
```

---

## 📈 Comparación de Todos los Modelos

| Modelo | Features | CV F1 | Test F1 | TDEs | Estrategia |
|--------|----------|-------|---------|------|-----------|
| **Original** | 183 | ~0.49 | 0.4326 | 178 | Baseline XGBoost |
| **Optimizado** | 237 | 0.5455 | 0.4158 | 450 | ❌ Overfitting |
| **Robusto** | 92 | 0.4568 | ??? | 439 | ✅ Anti-overfitting |

---

## 💡 Lecciones Aprendidas

1. **Más features ≠ Mejor modelo**
   - Features complejas pueden introducir ruido
   - Feature selection es crítica

2. **Threshold optimization puede causar overfitting**
   - Optimizar en validation set no garantiza generalización
   - Threshold=0.5 es más seguro

3. **CV score debe ser realista**
   - Si CV >> Test, hay overfitting
   - Mejor un CV modesto pero confiable

4. **Regularización es tu amiga**
   - Penalizar complejidad ayuda a generalizar
   - gamma, alpha, lambda son importantes

5. **Simplicidad > Complejidad**
   - Modelo simple con buen CV a menudo gana
   - KISS principle en ML

---

**Creado:** 2025-11-16
**Versión:** 2.0 (Robust)
