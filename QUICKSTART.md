# Quick Start Guide - MALLORN TDE Classification

## Configuración Rápida (5 minutos)

### 1. Instalar dependencias

```bash
python setup.py
```

O manualmente:

```bash
pip install -r requirements.txt
```

### 2. Colocar tus datos

Coloca tus archivos CSV en `data/raw/`:
- `training_log.csv` - Datos de entrenamiento con etiquetas
- `test_log.csv` - Datos de prueba para predicciones

```bash
# Ejemplo si tus archivos están en otra ubicación
cp /ruta/a/tus/datos/training_log.csv data/raw/
cp /ruta/a/tus/datos/test_log.csv data/raw/
```

### 3. Ejecutar el pipeline completo

```bash
python run_pipeline.py
```

Este comando:
1. ✓ Carga los datos de entrenamiento
2. ✓ Ingeniería de características
3. ✓ Entrena el modelo
4. ✓ Genera predicciones
5. ✓ Crea el archivo `submission.csv`

## Opción: Ejecutar pasos individuales

### Solo entrenar el modelo

```bash
python src/train.py
```

### Solo generar predicciones (requiere modelo entrenado)

```bash
python src/predict.py
```

## Exploración de datos (opcional)

Abre el notebook de exploración:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Estructura de archivos de salida

Después de ejecutar el pipeline, encontrarás:

```
stars/
├── submission.csv                          # ← Archivo para subir a la competencia
├── submission_YYYYMMDD_HHMMSS.csv         # ← Copia con timestamp
├── predictions_with_probabilities.csv      # ← Predicciones detalladas con probabilidades
└── models/
    └── tde_classifier.pkl                 # ← Modelo entrenado
```

## Formato del archivo de submission

El archivo `submission.csv` tendrá este formato:

```csv
id,prediction
0,0
1,1
2,0
...
```

Donde:
- `id`: Identificador del objeto
- `prediction`: 0 = No-TDE, 1 = TDE

## Personalización

### Cambiar el tipo de modelo

Edita `src/train.py`, línea ~180:

```python
# Opciones: 'xgboost', 'lightgbm', 'balanced_rf', 'gradient_boosting'
classifier = TDEClassifier(model_type='xgboost')
```

### Ajustar hiperparámetros

Edita el método `create_model()` en `src/train.py`:

```python
model = xgb.XGBClassifier(
    n_estimators=200,      # ← Número de árboles
    max_depth=6,           # ← Profundidad máxima
    learning_rate=0.05,    # ← Tasa de aprendizaje
    ...
)
```

### Usar SMOTE para balancear clases

En `src/train.py`, cambia:

```python
classifier.train(X_train, y_train, use_smote=True)
```

## Solución de problemas

### Error: "Training file not found"

Verifica que `data/raw/training_log.csv` existe:

```bash
ls -la data/raw/
```

### Error: "ModuleNotFoundError"

Instala las dependencias:

```bash
pip install -r requirements.txt
```

### El modelo tiene bajo rendimiento

1. Explora los datos con el notebook
2. Ajusta los hiperparámetros
3. Prueba diferentes modelos
4. Agrega más features en `feature_engineer.py`

## Métricas de evaluación

El desafío usa **F1 Score** para evaluación:

```
F1 = 2 × (precision × recall) / (precision + recall)
```

Donde:
- **Precision**: Proporción de TDEs correctamente identificados entre todos los clasificados como TDE
- **Recall**: Proporción de TDEs reales que fueron identificados

El F1 Score es ideal para datasets desbalanceados como este (pocos TDEs vs muchos No-TDEs).

## Próximos pasos

1. ✓ Ejecuta el pipeline básico
2. Explora los datos con el notebook
3. Experimenta con diferentes modelos
4. Mejora la ingeniería de características
5. Ajusta hiperparámetros
6. Envía tu mejor `submission.csv` a la competencia

## Recursos adicionales

- [Sobre TDEs (Tidal Disruption Events)](https://en.wikipedia.org/wiki/Tidal_disruption_event)
- [Documentación de XGBoost](https://xgboost.readthedocs.io/)
- [Documentación de scikit-learn](https://scikit-learn.org/)

¡Buena suerte en el desafío! 🌟
