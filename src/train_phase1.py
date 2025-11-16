"""
Phase 1 Training: Temporal Evolution Features
Train model with base + Phase 1 features for improved TDE recall
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from data_loader import DataLoader
from feature_engineer_phase1 import Phase1FeatureEngineer

print("="*70)
print(" MALLORN TDE Classification - Phase 1 Training")
print("="*70)
print("Strategy: Base features + Temporal Evolution Features")
print("Goal: Improve recall by capturing TDEs with different temporal profiles")
print("="*70)

# Setup paths
data_dir = Path('data/raw')
processed_dir = Path('data/processed')
models_dir = Path('models')

# Initialize data loader and Phase 1 feature engineer
loader = DataLoader(data_dir)
engineer = Phase1FeatureEngineer(apply_deextinction=True)

print("\n[1/7] Loading training log...")
train_log = loader.load_training_log()
print(f"Loaded {len(train_log)} training objects")

# Rename 'target' to 'is_tde' for consistency
if 'target' in train_log.columns:
    train_log['is_tde'] = train_log['target']

print(f"TDE: {train_log['is_tde'].sum()}, Non-TDE: {(~train_log['is_tde']).sum()}")

# Load or extract Phase 1 features
features_path = processed_dir / 'train_features_phase1.csv'
if features_path.exists():
    print(f"\n[2/7] Loading existing Phase 1 features from {features_path}...")
    features_df = pd.read_csv(features_path)
    print(f"Loaded features for {len(features_df)} objects")
else:
    print(f"\n[2/7] Extracting Phase 1 features (base + temporal evolution)...")
    features_df = engineer.process_multiple_objects(
        log_df=train_log,
        lightcurve_loader_func=loader.load_lightcurve,
        is_training=True
    )
    features_df.to_csv(features_path, index=False)
    print(f"Phase 1 features saved to {features_path}")

print(f"\n[3/7] Preparing training data...")
target_col = 'is_tde' if 'is_tde' in features_df.columns else 'target'
X = features_df.drop(['object_id', target_col], axis=1)
y = features_df[target_col].astype(int)
X = X.fillna(X.median())

print(f"Training set size: {len(X)}")
print(f"Number of features: {X.shape[1]}")

# Calculate scale_pos_weight
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
print(f"Class imbalance ratio: {scale_pos_weight:.2f}:1")

# === FEATURE SELECTION (same as fine-tuned) ===
print(f"\n[4/7] Feature selection...")
selector_model = xgb.XGBClassifier(
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)
selector_model.fit(X, y)

selector = SelectFromModel(selector_model, threshold='median', prefit=True)
X_selected = selector.transform(X)
selected_features = X.columns[selector.get_support()].tolist()

print(f"Features selected: {len(selected_features)} out of {X.shape[1]}")

# Check how many Phase 1 features were selected
phase1_features = [f for f in selected_features if any(keyword in f for keyword in [
    'asymmetry', 'power_law', 'n_obs_rise', 'n_obs_decay', 'peak_captured',
    'rise_fraction', 'decay_fraction', 'amplitude', 'peak_time_lag', 'global_'
])]

print(f"Phase 1 features selected: {len(phase1_features)}")
if len(phase1_features) > 0:
    print(f"Top Phase 1 features selected:")
    for f in phase1_features[:10]:
        print(f"  - {f}")

X = pd.DataFrame(X_selected, columns=selected_features)

# === PHASE 1 MODEL (same params as fine-tuned for fair comparison) ===
print(f"\n[5/7] Training Phase 1 model...")
print("Using fine-tuned model parameters for fair comparison:")
print("  - max_depth: 5")
print("  - n_estimators: 300")
print("  - Strong regularization")

model = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.05,
    n_estimators=300,
    min_child_weight=5,
    gamma=0.2,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

# === ROBUST CROSS-VALIDATION ===
print(f"\n[6/7] Performing 10-fold stratified cross-validation...")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Get out-of-fold predictions for threshold optimization
print("Getting out-of-fold predictions for threshold optimization...")
y_proba_cv = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]

# Calculate CV F1 scores
cv_scores = []
for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    fold_model = xgb.XGBClassifier(
        max_depth=5, learning_rate=0.05, n_estimators=300,
        min_child_weight=5, gamma=0.2, subsample=0.7,
        colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight, random_state=42,
        eval_metric='logloss'
    )
    fold_model.fit(X_train_fold, y_train_fold)
    y_pred_fold = fold_model.predict(X_val_fold)
    fold_f1 = f1_score(y_val_fold, y_pred_fold)
    cv_scores.append(fold_f1)
    print(f"  Fold {fold_idx+1}: {fold_f1:.4f}")

print(f"  Mean: {np.mean(cv_scores):.4f}")
print(f"  Std:  {np.std(cv_scores):.4f}")

# === THRESHOLD OPTIMIZATION USING CV PREDICTIONS ===
print(f"\nOptimizing threshold using CV predictions...")
precision, recall, thresholds = precision_recall_curve(y, y_proba_cv)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_threshold_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[best_threshold_idx]
optimal_f1 = f1_scores[best_threshold_idx]

print(f"Default threshold (0.5): F1 = {f1_score(y, y_proba_cv >= 0.5):.4f}")
print(f"Optimal threshold ({optimal_threshold:.3f}): F1 = {optimal_f1:.4f}")
print(f"Improvement: {optimal_f1 - f1_score(y, y_proba_cv >= 0.5):.4f}")

# Use a conservative threshold (average of default and optimal)
conservative_threshold = (0.5 + optimal_threshold) / 2
print(f"\nUsing conservative threshold: {conservative_threshold:.3f}")
print(f"(Average of default 0.5 and optimal {optimal_threshold:.3f})")

# Train final model on all data
print(f"\n[7/7] Training final model on all data...")
model.fit(X, y)

# Evaluate with conservative threshold
y_pred = (model.predict_proba(X)[:, 1] >= conservative_threshold).astype(int)
train_f1 = f1_score(y, y_pred)

print(f"\nTraining F1 score (conservative threshold): {train_f1:.4f}")
print(f"CV F1 score (mean): {np.mean(cv_scores):.4f}")
print(f"Gap: {train_f1 - np.mean(cv_scores):.4f}")

print("\nClassification report on training set:")
print(classification_report(y, y_pred, target_names=['Non-TDE', 'TDE']))

# === SAVE MODEL ===
print(f"\nSaving Phase 1 model...")

model_data = {
    'model': model,
    'model_name': 'XGBoost (Phase 1 - Temporal Evolution)',
    'threshold': conservative_threshold,
    'optimal_threshold': optimal_threshold,
    'cv_f1_mean': np.mean(cv_scores),
    'cv_f1_std': np.std(cv_scores),
    'train_f1': train_f1,
    'feature_names': selected_features,
    'n_features': len(selected_features),
    'n_phase1_features': len(phase1_features),
    'phase1_features': phase1_features
}

model_path = models_dir / 'tde_classifier_phase1.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"Phase 1 model saved to: {model_path}")

# Save selected features
features_path = models_dir / 'feature_names_phase1.txt'
with open(features_path, 'w') as f:
    for fname in selected_features:
        f.write(f"{fname}\n")
print(f"Selected feature names saved to: {features_path}")

print("\n" + "="*70)
print(" TRAINING COMPLETE!")
print("="*70)
print(f"\nFinal Results:")
print(f"  Model: XGBoost (Phase 1 - Temporal Evolution Features)")
print(f"  CV F1 Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
print(f"  Train F1 Score: {train_f1:.4f}")
print(f"  Total Features: {len(selected_features)}")
print(f"  Phase 1 Features: {len(phase1_features)}")
print(f"  Conservative Threshold: {conservative_threshold:.4f}")
print(f"  Optimal Threshold: {optimal_threshold:.4f}")
print("\nPhase 1 Strategy:")
print("  - Base features from LightcurveFeatureEngineer")
print("  - Plus temporal evolution features:")
print("    * Rise/decay asymmetry ratios")
print("    * Power law decay fits (alpha ~ -5/3 for TDEs)")
print("    * Event completeness (n_obs rise/decay)")
print("    * Cross-filter time lags")
print("    * Global event characteristics")
print("  - Feature selection (median threshold)")
print("  - Fine-tuned parameters (max_depth=5, n_estimators=300)")
print("  - Strong regularization (gamma=0.2, alpha=0.5, lambda=1.0)")
print("  - CV-optimized conservative threshold")
print("\nExpected Impact:")
print("  - Target: Improve recall on TDEs with non-standard temporal profiles")
print("  - Goal: CV F1 0.50-0.55, Test F1 0.54-0.58")
print("="*70)
