# Comparación de Modelos - MALLORN TDE Classification

## 📊 Resumen de Todos los Modelos

| # | Modelo | Features | CV F1 | Test F1 | TDEs | Threshold | Gap CV-Test | Estado |
|---|--------|----------|-------|---------|------|-----------|-------------|--------|
| 1 | **Original** | 183 | ~0.49 | **0.4326** | 178 | 0.5 | -0.06 | ✅ Baseline |
| 2 | **Optimizado** | 237 | 0.5455 | **0.4158** | 450 | 0.4412 | -0.1297 | ❌ Overfitting |
| 3 | **Robusto** | 92 | 0.4568 | **0.4764** | 439 | 0.5 | **+0.0196** | ✅ Mejor |
| 4 | **Fine-Tuned** | 92 | 0.4259 | **???** | 372 | 0.3946 | ??? | 🔄 Pendiente |

---

## 📈 Evolución del Score

```
Baseline:    0.4326 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    │
Optimizado:  0.4158 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ❌ (-0.0168)
                    │
Robusto:     0.4764 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ✅ (+0.0438)
                    │
Fine-Tuned:  ???    🔄 Pendiente
```

---

## 🔍 Análisis Detallado

### 1️⃣ Modelo Original (Baseline)
**Score:** 0.4326

**Características:**
- 183 features (básicas)
- XGBoost con scale_pos_weight=10
- Threshold=0.5 (default)
- No feature selection

**Resultado:** Baseline sólido para comparación

---

### 2️⃣ Modelo Optimizado (Overfitting)
**Score:** 0.4158 ❌

**Características:**
- 237 features (54 nuevas complejas)
  - Color features
  - Color evolution
  - Advanced stats
- Hyperparameter tuning
- Threshold optimizado: 0.4412
- Ensemble methods

**Problema:**
- CV F1 = 0.5455
- Test F1 = 0.4158
- **Gap = -0.1297 (overfitting severo)**

**Lecciones:**
- ❌ Más features ≠ mejor modelo
- ❌ Threshold optimization en validation set → overfitting
- ❌ Features complejas introducen ruido

---

### 3️⃣ Modelo Robusto (Mejor hasta ahora)
**Score:** 0.4764 ✅ (+10.1% vs baseline)

**Características:**
- 92 features (feature selection agresiva)
- Features simples (originales)
- Regularización fuerte:
  ```python
  max_depth=4
  learning_rate=0.05
  gamma=0.2
  reg_alpha=0.5
  reg_lambda=1.0
  subsample=0.7
  ```
- Threshold=0.5 (default, no optimizado)
- 10-fold stratified CV

**Resultados:**
- CV F1: 0.4568
- Test F1: 0.4764
- **Gap: +0.0196 (generaliza excelente!)**

**Por qué funciona:**
- ✅ Features simples → menos ruido
- ✅ Feature selection → solo las importantes
- ✅ Regularización fuerte → previene memorización
- ✅ Threshold conservador → mejor generalización
- ✅ CV realista → evaluación confiable

---

### 4️⃣ Modelo Fine-Tuned (Pendiente evaluación)
**Score:** ??? 🔄

**Características:**
- 92 features (mismas que robusto)
- Ajustes finos:
  ```python
  max_depth: 4 → 5
  n_estimators: 200 → 300
  ```
- Threshold CV-optimizado conservador: 0.3946
  - Óptimo en CV: 0.2892
  - Conservador: (0.5 + 0.2892) / 2 = 0.3946

**Resultados:**
- CV F1: 0.4259 ± 0.1032
- Test F1: ??? (pendiente)
- TDEs predichos: 372 (5.21%)

**Expectativas:**
- Threshold más bajo → más agresivo en detectar TDEs
- Árboles más profundos → mejor fit
- CV-optimized threshold → podría mejorar o causar overfitting leve

**Riesgo:** El threshold 0.3946 es agresivo, podría causar algo de overfitting

---

## 🎯 Predicciones Comparadas

| Modelo | TDEs | % | Estrategia |
|--------|------|---|-----------|
| Original | 178 | 2.49% | Conservador |
| Optimizado | 450 | 6.31% | Muy agresivo (threshold=0.44) |
| Robusto | 439 | 6.15% | Balanceado (threshold=0.5) |
| Fine-Tuned | 372 | 5.21% | Moderado (threshold=0.39) |

**Real (ground truth):** ~5% TDEs esperados

El modelo **Fine-Tuned** con 372 TDEs (5.21%) está muy cerca del valor esperado.

---

## 🏆 Ranking de Modelos

### Por Score (Test F1)
1. **Robusto:** 0.4764 🥇
2. **Original:** 0.4326 🥈
3. **Optimizado:** 0.4158 🥉
4. **Fine-Tuned:** ??? 🔄

### Por Generalización (Gap CV-Test)
1. **Robusto:** +0.0196 (test > CV) 🥇
2. **Original:** ~-0.06 🥈
3. **Optimizado:** -0.1297 (overfitting) 🥉
4. **Fine-Tuned:** ??? 🔄

### Por Simplicidad
1. **Robusto:** 92 features, simple 🥇
2. **Fine-Tuned:** 92 features, simple 🥇
3. **Original:** 183 features 🥈
4. **Optimizado:** 237 features, complejo 🥉

---

## 💡 Lecciones Clave

### ✅ Qué Funcionó
1. **Feature selection agresiva** (183 → 92)
2. **Regularización fuerte** (gamma, L1, L2)
3. **Features simples** (no complejas)
4. **Threshold conservador** (0.5, no optimizado)
5. **10-fold CV** (evaluación robusta)

### ❌ Qué No Funcionó
1. **Features complejas** (color evolution)
2. **Threshold optimization en validation set**
3. **Modelo muy complejo** (237 features)
4. **3-fold CV** (no suficiente)
5. **Optimizar todo** (hyperparams + threshold + ensemble)

### 🤔 Por Probar
1. **Threshold CV-optimizado conservador** (Fine-Tuned)
2. **Ensemble conservador** (XGB + LightGBM simple)
3. **Más regularización** (gamma=0.5)
4. **Menos features** (top 50-70)

---

## 🚀 Próximos Pasos

### Si Fine-Tuned < 0.47
El threshold agresivo (0.3946) causó overfitting. **Recomendación:**
- Volver al modelo **Robusto** (0.4764)
- O usar threshold=0.5 en Fine-Tuned

### Si 0.47 ≤ Fine-Tuned < 0.48
Mejora marginal. **Recomendación:**
- Probar ensemble conservador
- O quedarse con Robusto

### Si Fine-Tuned ≥ 0.48
¡Éxito! **Recomendación:**
- Usar Fine-Tuned como modelo final
- Documentar y finalizar

---

## 📁 Archivos Disponibles

### Modelo Robusto (Mejor hasta ahora)
```bash
submission.csv                          # Predicciones (439 TDEs)
src/train_robust.py                     # Training script
src/predict_robust.py                   # Prediction script
models/tde_classifier_robust.pkl        # Modelo guardado
models/feature_names_robust.txt         # 92 features
```

### Modelo Fine-Tuned (En evaluación)
```bash
submission.csv                          # Predicciones (372 TDEs)
src/train_finetuned.py                  # Training script
src/predict_finetuned.py                # Prediction script
models/tde_classifier_finetuned.pkl     # Modelo guardado
models/feature_names_finetuned.txt      # 92 features
```

---

## 🎓 Resumen Ejecutivo

**Mejor modelo actual:** **Robusto** (Test F1 = 0.4764)

**Por qué:**
- ✅ Mejor score en competencia
- ✅ Excelente generalización (CV ≈ Test)
- ✅ Simplicidad (92 features)
- ✅ Robusto y confiable

**Candidato alternativo:** **Fine-Tuned** (pendiente evaluación)

**Diferencias:**
- Threshold más agresivo (0.39 vs 0.5)
- Árboles un poco más profundos (5 vs 4)
- Más estimators (300 vs 200)
- Menos TDEs predichos (372 vs 439)

---

**Última actualización:** 2025-11-16
**Mejor score:** 0.4764 (Modelo Robusto)
**Mejora sobre baseline:** +10.1%
