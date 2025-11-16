"""
Robust training with anti-overfitting measures
Focus on generalization instead of validation performance
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from data_loader import DataLoader
from feature_engineer import LightcurveFeatureEngineer  # Use original, simpler features

print("="*70)
print(" MALLORN TDE Classification - Robust Training Pipeline")
print("="*70)
print("Focus: Generalization over validation performance")
print("Strategy: Simpler features + Strong regularization + CV")
print("="*70)

# Setup paths
data_dir = Path('data/raw')
processed_dir = Path('data/processed')
models_dir = Path('models')

# Initialize data loader and feature engineer (using ORIGINAL simpler version)
loader = DataLoader(data_dir)
engineer = LightcurveFeatureEngineer(apply_deextinction=True)

print("\n[1/6] Loading training log...")
train_log = loader.load_training_log()
print(f"Loaded {len(train_log)} training objects")

# Rename 'target' to 'is_tde' for consistency
if 'target' in train_log.columns:
    train_log['is_tde'] = train_log['target']

print(f"TDE: {train_log['is_tde'].sum()}, Non-TDE: {(~train_log['is_tde']).sum()}")

# Load or extract features (using ORIGINAL features)
features_path = processed_dir / 'train_features.csv'
if features_path.exists():
    print(f"\n[2/6] Loading existing features from {features_path}...")
    features_df = pd.read_csv(features_path)
    print(f"Loaded features for {len(features_df)} objects")
else:
    print(f"\n[2/6] Extracting ORIGINAL features (simpler, less overfitting)...")
    features_df = engineer.process_multiple_objects(
        log_df=train_log,
        lightcurve_loader_func=loader.load_lightcurve,
        is_training=True
    )
    features_df.to_csv(features_path, index=False)
    print(f"Features saved to {features_path}")

print(f"\n[3/6] Preparing training data...")
print(f"Feature shape: {features_df.shape}")

# Determine target column name
target_col = 'is_tde' if 'is_tde' in features_df.columns else 'target'

# Separate features and target
X = features_df.drop(['object_id', target_col], axis=1)
y = features_df[target_col].astype(int)

# Handle missing values
X = X.fillna(X.median())

print(f"Training set size: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print(f"Class distribution: TDE={y.sum()}, Non-TDE={(~y.astype(bool)).sum()}")

# Calculate scale_pos_weight
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
print(f"Class imbalance ratio: {scale_pos_weight:.2f}:1")

# === FEATURE SELECTION ===
print(f"\n[4/6] Feature selection to reduce overfitting...")

# Train a simple model for feature selection
selector_model = xgb.XGBClassifier(
    max_depth=3,  # Very shallow to avoid overfitting
    learning_rate=0.1,
    n_estimators=100,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

selector_model.fit(X, y)

# Select top features based on importance
selector = SelectFromModel(selector_model, threshold='median', prefit=True)
X_selected = selector.transform(X)

selected_features = X.columns[selector.get_support()].tolist()
print(f"Features selected: {len(selected_features)} out of {X.shape[1]}")
print(f"Top 10 selected features:")
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': selector_model.feature_importances_
}).sort_values('importance', ascending=False)
print(importances.head(10))

# Update X with selected features only
X = pd.DataFrame(X_selected, columns=selected_features)

# === ROBUST MODEL WITH STRONG REGULARIZATION ===
print(f"\n[5/6] Training robust model with strong regularization...")

# Use conservative hyperparameters to prevent overfitting
model = xgb.XGBClassifier(
    max_depth=4,              # Limit tree depth
    learning_rate=0.05,       # Lower learning rate
    n_estimators=200,         # Fewer trees
    min_child_weight=5,       # Higher minimum samples per leaf
    gamma=0.2,                # Minimum loss reduction (regularization)
    subsample=0.7,            # Use only 70% of data per tree
    colsample_bytree=0.7,     # Use only 70% of features per tree
    reg_alpha=0.5,            # L1 regularization
    reg_lambda=1.0,           # L2 regularization
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

# === ROBUST CROSS-VALIDATION ===
print(f"\nPerforming 10-fold stratified cross-validation...")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)

print(f"\nCross-validation F1 scores:")
for i, score in enumerate(cv_scores):
    print(f"  Fold {i+1}: {score:.4f}")
print(f"  Mean: {cv_scores.mean():.4f}")
print(f"  Std:  {cv_scores.std():.4f}")

# Train final model on all data
print(f"\nTraining final model on all data...")
model.fit(X, y)

# === EVALUATE ON TRAINING SET (should be modest due to regularization) ===
y_pred = model.predict(X)
train_f1 = f1_score(y, y_pred)
print(f"\nTraining F1 score: {train_f1:.4f}")
print("(Should be close to CV mean for good generalization)")

print("\nClassification report on training set:")
print(classification_report(y, y_pred, target_names=['Non-TDE', 'TDE']))

# === SAVE MODEL ===
print(f"\n[6/6] Saving robust model...")

model_data = {
    'model': model,
    'model_name': 'XGBoost (Robust)',
    'threshold': 0.5,  # Use default threshold, no optimization
    'cv_f1_mean': cv_scores.mean(),
    'cv_f1_std': cv_scores.std(),
    'train_f1': train_f1,
    'feature_names': selected_features,
    'n_features': len(selected_features)
}

model_path = models_dir / 'tde_classifier_robust.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"Robust model saved to: {model_path}")

# Save selected features
features_path = models_dir / 'feature_names_robust.txt'
with open(features_path, 'w') as f:
    for fname in selected_features:
        f.write(f"{fname}\n")
print(f"Selected feature names saved to: {features_path}")

print("\n" + "="*70)
print(" TRAINING COMPLETE!")
print("="*70)
print(f"\nFinal Results:")
print(f"  Model: XGBoost (Robust with strong regularization)")
print(f"  CV F1 Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Train F1 Score: {train_f1:.4f}")
print(f"  Features: {len(selected_features)} (selected from {features_df.shape[1]-2})")
print(f"  Threshold: 0.5 (default, not optimized)")
print("\nStrategy:")
print("  - Using original simpler features (no complex color evolution)")
print("  - Feature selection (median importance threshold)")
print("  - Strong regularization (gamma=0.2, alpha=0.5, lambda=1.0)")
print("  - Conservative hyperparameters (max_depth=4, subsample=0.7)")
print("  - 10-fold CV for robust evaluation")
print("  - Default threshold (0.5) to avoid overfitting")
print("="*70)
