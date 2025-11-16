"""
Optimized training script with:
- Enhanced features (V2)
- Hyperparameter tuning
- Ensemble models
- Threshold optimization
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from data_loader import DataLoader
from feature_engineer_v2 import LightcurveFeatureEngineerV2

print("="*70)
print(" MALLORN TDE Classification - Optimized Training Pipeline")
print("="*70)

# Setup paths
data_dir = Path('data/raw')
processed_dir = Path('data/processed')
models_dir = Path('models')

# Create directories
processed_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

# Initialize data loader and feature engineer
loader = DataLoader(data_dir)
engineer = LightcurveFeatureEngineerV2(apply_deextinction=True)

print("\n[1/7] Loading training log...")
train_log = loader.load_training_log()
print(f"Loaded {len(train_log)} training objects")

# Rename 'target' to 'is_tde' for consistency
if 'target' in train_log.columns:
    train_log['is_tde'] = train_log['target']

print(f"TDE: {train_log['is_tde'].sum()}, Non-TDE: {(~train_log['is_tde']).sum()}")

# Load or extract features
features_path = processed_dir / 'train_features_v2.csv'
if features_path.exists():
    print(f"\n[2/7] Loading existing features from {features_path}...")
    features_df = pd.read_csv(features_path)
    print(f"Loaded features for {len(features_df)} objects")
else:
    print(f"\n[2/7] Extracting features (this will take a while)...")
    features_df = engineer.process_multiple_objects(
        log_df=train_log,
        lightcurve_loader_func=loader.load_lightcurve,
        is_training=True
    )

    # Save features
    features_df.to_csv(features_path, index=False)
    print(f"Features saved to {features_path}")

print(f"\n[3/7] Preparing training data...")
print(f"Feature shape: {features_df.shape}")

# Separate features and target
X = features_df.drop(['object_id', 'is_tde'], axis=1)
y = features_df['is_tde'].astype(int)

# Handle missing values
X = X.fillna(X.median())

print(f"Training set size: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print(f"Class distribution: TDE={y.sum()}, Non-TDE={(~y.astype(bool)).sum()}")

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {len(X_train)}, Validation size: {len(X_val)}")

# Save feature names
feature_names_path = models_dir / 'feature_names_v2.txt'
with open(feature_names_path, 'w') as f:
    for fname in X.columns:
        f.write(f"{fname}\n")
print(f"Feature names saved to {feature_names_path}")

# === HYPERPARAMETER TUNING ===
print(f"\n[4/7] Hyperparameter tuning for XGBoost...")

# Calculate scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class imbalance ratio: {scale_pos_weight:.2f}:1")

# Define parameter grid
param_distributions = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 500],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
}

# Base XGBoost model
base_xgb = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

# Randomized search (faster than GridSearch)
print("Running randomized search (20 iterations)...")
random_search = RandomizedSearchCV(
    base_xgb,
    param_distributions=param_distributions,
    n_iter=20,  # Number of parameter settings sampled
    scoring='f1',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print(f"\nBest parameters found:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"Best CV F1 score: {random_search.best_score_:.4f}")

best_xgb = random_search.best_estimator_

# === TRAIN ENSEMBLE ===
print(f"\n[5/7] Training ensemble models...")

# Train Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Create voting ensemble
print("Creating voting ensemble...")
ensemble = VotingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('rf', rf_model)
    ],
    voting='soft',  # Use predicted probabilities
    weights=[2, 1]  # Give more weight to XGBoost
)

ensemble.fit(X_train, y_train)

# === EVALUATE MODELS ===
print(f"\n[6/7] Evaluating models on validation set...")

models = {
    'XGBoost (Tuned)': best_xgb,
    'Random Forest': rf_model,
    'Ensemble': ensemble
}

best_model = None
best_f1 = 0
best_model_name = None

for model_name, model in models.items():
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    f1 = f1_score(y_val, y_pred)

    print(f"\n{model_name}:")
    print(f"  F1 Score: {f1:.4f}")
    print(classification_report(y_val, y_pred, target_names=['Non-TDE', 'TDE']))

    # Track best model
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = model_name

print(f"\nBest model: {best_model_name} (F1={best_f1:.4f})")

# === THRESHOLD OPTIMIZATION ===
print(f"\n[7/7] Optimizing classification threshold...")

y_proba = best_model.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_proba)

# Calculate F1 for each threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]
best_f1_optimized = f1_scores[best_threshold_idx]

print(f"Default threshold (0.5): F1 = {best_f1:.4f}")
print(f"Optimized threshold ({best_threshold:.3f}): F1 = {best_f1_optimized:.4f}")
print(f"Improvement: {best_f1_optimized - best_f1:.4f}")

# Apply optimized threshold
y_pred_optimized = (y_proba >= best_threshold).astype(int)

print(f"\nOptimized predictions:")
print(classification_report(y_val, y_pred_optimized, target_names=['Non-TDE', 'TDE']))

# === SAVE MODELS ===
print(f"\nSaving models...")

# Save best model with optimized threshold
model_data = {
    'model': best_model,
    'model_name': best_model_name,
    'threshold': best_threshold,
    'f1_score': best_f1_optimized,
    'feature_names': list(X.columns)
}

model_path = models_dir / 'tde_classifier_optimized.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"Optimized model saved to: {model_path}")

# Also save XGBoost separately (for comparison)
xgb_path = models_dir / 'tde_xgboost_tuned.pkl'
with open(xgb_path, 'wb') as f:
    pickle.dump(best_xgb, f)
print(f"XGBoost model saved to: {xgb_path}")

print("\n" + "="*70)
print(" TRAINING COMPLETE!")
print("="*70)
print(f"\nFinal Results:")
print(f"  Model: {best_model_name}")
print(f"  F1 Score: {best_f1_optimized:.4f}")
print(f"  Optimal Threshold: {best_threshold:.3f}")
print(f"  Features: {X.shape[1]}")
print("="*70)
