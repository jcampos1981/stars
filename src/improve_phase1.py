#!/usr/bin/env python3
"""
Opción A - Improve Phase 1 Model
Strategy: Phase 1 achieved best test score (0.5105)
Goal: Optimize to reach 0.60 target

Improvement strategies:
1. Threshold optimization
2. Hyperparameter tuning
3. Feature selection refinement
4. Ensemble approaches
"""

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_selection import SelectFromModel
from pathlib import Path

print("="*80)
print(" PHASE 1 IMPROVEMENT - Opción A")
print("="*80)
print("Current best: Phase 1 with test score 0.5105")
print("Goal: Reach 0.60 through optimization")
print("="*80)

# Load pre-processed Phase 1 features
print("\n[1/6] Loading pre-processed Phase 1 features...")
train_features = pd.read_csv('data/processed/train_features_phase1.csv')
print(f"Loaded features for {len(train_features)} objects")
print(f"Feature shape: {train_features.shape}")

# Prepare data
target_col = 'is_tde' if 'is_tde' in train_features.columns else 'target'
X = train_features.drop(['object_id', target_col], axis=1)
y = train_features[target_col].astype(int)

# Fill NaN values
X = X.fillna(X.median())

print(f"\nFeature matrix: {X.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Strategy 1: Hyperparameter Optimization
print("\n[2/6] Strategy 1 - Hyperparameter Optimization")
print("-" * 80)

# Calculate class weight
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
print(f"Class imbalance ratio: {scale_pos_weight:.2f}")

# Test multiple hyperparameter configurations
configs = [
    {
        'name': 'Current (Phase 1)',
        'params': {
            'max_depth': 5,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'min_child_weight': 5,
            'gamma': 0.2,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
            'scale_pos_weight': scale_pos_weight,
        }
    },
    {
        'name': 'Deeper Trees',
        'params': {
            'max_depth': 6,
            'learning_rate': 0.03,
            'n_estimators': 400,
            'min_child_weight': 3,
            'gamma': 0.3,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
            'scale_pos_weight': scale_pos_weight,
        }
    },
    {
        'name': 'More Regularization',
        'params': {
            'max_depth': 5,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'min_child_weight': 7,
            'gamma': 0.5,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'reg_alpha': 1.0,
            'reg_lambda': 2.0,
            'scale_pos_weight': scale_pos_weight,
        }
    },
    {
        'name': 'More Estimators',
        'params': {
            'max_depth': 5,
            'learning_rate': 0.03,
            'n_estimators': 500,
            'min_child_weight': 5,
            'gamma': 0.2,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
            'scale_pos_weight': scale_pos_weight,
        }
    },
    {
        'name': 'Balanced',
        'params': {
            'max_depth': 6,
            'learning_rate': 0.04,
            'n_estimators': 400,
            'min_child_weight': 4,
            'gamma': 0.3,
            'subsample': 0.65,
            'colsample_bytree': 0.65,
            'reg_alpha': 0.7,
            'reg_lambda': 1.5,
            'scale_pos_weight': scale_pos_weight,
        }
    },
]

# Evaluate each configuration with cross-validation
cv_results = []
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for config in configs:
    print(f"\nTesting configuration: {config['name']}")
    print(f"Parameters: {config['params']}")

    fold_scores = []
    fold_thresholds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Train model
        model = xgb.XGBClassifier(
            **config['params'],
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train_fold, y_train_fold)

        # Get probabilities
        y_val_proba = model.predict_proba(X_val_fold)[:, 1]

        # Find optimal threshold for this fold
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_f1 = 0

        for thresh in thresholds:
            y_val_pred = (y_val_proba >= thresh).astype(int)
            f1 = f1_score(y_val_fold, y_val_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

        fold_scores.append(best_f1)
        fold_thresholds.append(best_threshold)

    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)
    mean_threshold = np.mean(fold_thresholds)

    cv_results.append({
        'name': config['name'],
        'params': config['params'],
        'cv_f1_mean': mean_f1,
        'cv_f1_std': std_f1,
        'mean_threshold': mean_threshold,
        'fold_scores': fold_scores
    })

    print(f"  CV F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  Mean optimal threshold: {mean_threshold:.4f}")

# Find best configuration
print("\n" + "="*80)
print(" CONFIGURATION COMPARISON")
print("="*80)

for result in sorted(cv_results, key=lambda x: x['cv_f1_mean'], reverse=True):
    print(f"{result['name']:25s} | CV F1: {result['cv_f1_mean']:.4f} ± {result['cv_f1_std']:.4f} | Threshold: {result['mean_threshold']:.4f}")

best_config = max(cv_results, key=lambda x: x['cv_f1_mean'])
print(f"\n✓ Best configuration: {best_config['name']}")
print(f"  CV F1: {best_config['cv_f1_mean']:.4f} ± {best_config['cv_f1_std']:.4f}")
print(f"  Optimal threshold: {best_config['mean_threshold']:.4f}")

# Strategy 2: Feature Selection Optimization
print("\n[3/6] Strategy 2 - Feature Selection Optimization")
print("-" * 80)

# Try different feature selection thresholds
print("Testing different feature selection strategies...")

selection_strategies = [
    {'name': 'Current (median)', 'threshold': 'median'},
    {'name': 'Mean importance', 'threshold': 'mean'},
    {'name': 'Top 100 features', 'threshold': -100},  # negative means top N features
    {'name': 'Top 120 features', 'threshold': -120},
    {'name': '0.75 * mean', 'threshold': '0.75*mean'},
]

selection_results = []

for strategy in selection_strategies:
    print(f"\nTesting: {strategy['name']}")

    # Train selector model
    selector_model = xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    selector_model.fit(X, y)

    # Select features
    if strategy['threshold'] == -100:
        # Top 100 features
        importances = selector_model.feature_importances_
        top_indices = np.argsort(importances)[-100:]
        selected_features = X.columns[top_indices].tolist()
    elif strategy['threshold'] == -120:
        # Top 120 features
        importances = selector_model.feature_importances_
        top_indices = np.argsort(importances)[-120:]
        selected_features = X.columns[top_indices].tolist()
    else:
        selector = SelectFromModel(selector_model, threshold=strategy['threshold'], prefit=True)
        selected_features = X.columns[selector.get_support()].tolist()

    X_selected = X[selected_features]

    print(f"  Selected {len(selected_features)} features")

    # Evaluate with cross-validation using best hyperparameters
    fold_scores = []
    for train_idx, val_idx in skf.split(X_selected, y):
        X_train_fold, X_val_fold = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(
            **best_config['params'],
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train_fold, y_train_fold)

        y_val_proba = model.predict_proba(X_val_fold)[:, 1]
        y_val_pred = (y_val_proba >= best_config['mean_threshold']).astype(int)

        f1 = f1_score(y_val_fold, y_val_pred, zero_division=0)
        fold_scores.append(f1)

    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)

    selection_results.append({
        'name': strategy['name'],
        'n_features': len(selected_features),
        'cv_f1_mean': mean_f1,
        'cv_f1_std': std_f1,
        'features': selected_features
    })

    print(f"  CV F1: {mean_f1:.4f} ± {std_f1:.4f}")

# Find best feature selection
best_selection = max(selection_results, key=lambda x: x['cv_f1_mean'])
print(f"\n✓ Best feature selection: {best_selection['name']}")
print(f"  Features: {best_selection['n_features']}")
print(f"  CV F1: {best_selection['cv_f1_mean']:.4f} ± {best_selection['cv_f1_std']:.4f}")

# Train final improved model
print("\n[4/6] Training final improved Phase 1 model...")
print("-" * 80)

X_final = X[best_selection['features']]

print(f"Using best configuration: {best_config['name']}")
print(f"Using best feature selection: {best_selection['name']} ({best_selection['n_features']} features)")

# Train on full training set
final_model = xgb.XGBClassifier(
    **best_config['params'],
    random_state=42,
    eval_metric='logloss'
)

final_model.fit(X_final, y)

# Get training performance
y_train_proba = final_model.predict_proba(X_final)[:, 1]
y_train_pred = (y_train_proba >= best_config['mean_threshold']).astype(int)
train_f1 = f1_score(y, y_train_pred, zero_division=0)

print(f"\nFinal model trained!")
print(f"Training F1: {train_f1:.4f}")
print(f"CV F1: {best_config['cv_f1_mean']:.4f} ± {best_config['cv_f1_std']:.4f}")
print(f"Threshold: {best_config['mean_threshold']:.4f}")

# Save improved model
model_data = {
    'model': final_model,
    'feature_names': best_selection['features'],
    'threshold': best_config['mean_threshold'],
    'cv_f1_mean': best_config['cv_f1_mean'],
    'cv_f1_std': best_config['cv_f1_std'],
    'train_f1': train_f1,
    'config_name': best_config['name'],
    'selection_name': best_selection['name'],
    'n_features': best_selection['n_features'],
    'phase1_features': [f for f in best_selection['features'] if any(kw in f for kw in ['rise', 'decay', 'asymmetry', 'amplitude', 'power_law'])]
}

output_file = 'models/tde_classifier_phase1_improved.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✓ Model saved to: {output_file}")

# Save feature names
feature_file = 'models/feature_names_phase1_improved.txt'
with open(feature_file, 'w') as f:
    for feat in best_selection['features']:
        f.write(f"{feat}\n")

print(f"✓ Feature names saved to: {feature_file}")

# Summary
print("\n" + "="*80)
print(" IMPROVEMENT SUMMARY")
print("="*80)
print(f"Original Phase 1 CV F1: ~0.4850")
print(f"Improved Phase 1 CV F1: {best_config['cv_f1_mean']:.4f} ± {best_config['cv_f1_std']:.4f}")
print(f"Configuration: {best_config['name']}")
print(f"Features: {best_selection['n_features']} selected")
print(f"Threshold: {best_config['mean_threshold']:.4f}")
print("="*80)

if best_config['cv_f1_mean'] > 0.485:
    improvement = ((best_config['cv_f1_mean'] - 0.485) / 0.485) * 100
    print(f"✓ CV improvement: +{improvement:.2f}%")
else:
    print("⚠ CV performance did not improve significantly")

print("\nNext step: Generate predictions with improved model")
print("Run: python src/predict_phase1_improved.py")
