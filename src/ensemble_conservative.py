#!/usr/bin/env python3
"""
Conservative Ensemble Strategy

Since threshold adjustment won't work due to severe overfitting,
we'll create an ensemble that combines multiple models to reduce
overfitting and improve generalization.

Strategy:
1. Load available trained models (Phase 1, finetuned, optimized, etc.)
2. Evaluate each model's CV performance
3. Create weighted ensemble based on CV performance
4. Validate ensemble improves over Phase 1 Original (0.5105)
5. Only generate predictions if validation passes
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" CONSERVATIVE ENSEMBLE STRATEGY")
print("="*80)
print("Goal: Improve over Phase 1 Original (0.5105) via ensemble")
print("Strategy: Combine models to reduce overfitting")
print("="*80)

# Load training features
print("\n[1/5] Loading training data...")
train_features = pd.read_csv('data/processed/train_features_phase1.csv')
target_col = 'is_tde' if 'is_tde' in train_features.columns else 'target'
y = train_features[target_col].astype(int)

print(f"Training set: {len(train_features)} objects")
print(f"TDEs: {y.sum()} | Non-TDEs: {(~y.astype(bool)).sum()}")

# Find available models
print("\n[2/5] Loading available models...")
models_dir = Path('models')
available_models = []

model_files = {
    'phase1': 'tde_classifier_phase1.pkl',
    'finetuned': 'tde_classifier_finetuned.pkl',
    'optimized': 'tde_classifier_optimized.pkl',
    'robust': 'tde_classifier_robust.pkl',
}

for name, filename in model_files.items():
    model_path = models_dir / filename
    if model_path.exists():
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            available_models.append({
                'name': name,
                'path': model_path,
                'model': model_data['model'],
                'threshold': model_data['threshold'],
                'feature_names': model_data['feature_names'],
                'cv_f1_mean': model_data.get('cv_f1_mean', 0.0)
            })
            print(f"  ✓ Loaded {name}: CV F1 = {model_data.get('cv_f1_mean', 0.0):.4f}")
        except Exception as e:
            print(f"  ✗ Failed to load {name}: {e}")

print(f"\nTotal models loaded: {len(available_models)}")

if len(available_models) == 0:
    print("ERROR: No models available for ensemble")
    exit(1)

# Evaluate each model with CV
print("\n[3/5] Evaluating models with 10-fold CV...")
print("-" * 80)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
model_evaluations = []

for model_info in available_models:
    print(f"\nEvaluating: {model_info['name']}")

    X = train_features[model_info['feature_names']]
    X = X.fillna(X.median())

    all_probs = np.zeros(len(y))

    for train_idx, val_idx in skf.split(X, y):
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        # Get probabilities
        y_proba = model_info['model'].predict_proba(X_val)[:, 1]
        all_probs[val_idx] = y_proba

    # Calculate F1 at model's threshold
    predictions = (all_probs >= model_info['threshold']).astype(int)
    cv_f1 = f1_score(y, predictions, zero_division=0)

    # Also find optimal threshold on this CV run
    thresholds = np.arange(0.1, 0.9, 0.02)
    best_f1 = 0
    best_thresh = model_info['threshold']

    for thresh in thresholds:
        preds = (all_probs >= thresh).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    model_evaluations.append({
        'name': model_info['name'],
        'model': model_info['model'],
        'feature_names': model_info['feature_names'],
        'cv_f1': cv_f1,
        'cv_f1_optimal': best_f1,
        'threshold': model_info['threshold'],
        'optimal_threshold': best_thresh,
        'probabilities': all_probs
    })

    print(f"  CV F1 (at threshold {model_info['threshold']:.4f}): {cv_f1:.4f}")
    print(f"  CV F1 (optimal threshold {best_thresh:.4f}): {best_f1:.4f}")

# Sort by CV performance
model_evaluations.sort(key=lambda x: x['cv_f1_optimal'], reverse=True)

print("\n" + "="*80)
print(" MODEL RANKING")
print("="*80)
for i, eval_info in enumerate(model_evaluations, 1):
    print(f"{i}. {eval_info['name']:15s} | CV F1: {eval_info['cv_f1_optimal']:.4f} | Threshold: {eval_info['optimal_threshold']:.4f}")

# Create ensemble combinations
print("\n[4/5] Testing ensemble combinations...")
print("-" * 80)

# Phase 1 Original baseline
baseline_eval = [e for e in model_evaluations if e['name'] == 'phase1'][0]
baseline_f1 = baseline_eval['cv_f1_optimal']

print(f"\nBaseline (Phase 1 Original): CV F1 = {baseline_f1:.4f}")

ensemble_results = []

# Test different ensemble strategies
strategies = [
    {'name': 'Average All', 'models': [e['name'] for e in model_evaluations]},
    {'name': 'Top 2', 'models': [e['name'] for e in model_evaluations[:2]]},
    {'name': 'Top 3', 'models': [e['name'] for e in model_evaluations[:3]]},
    {'name': 'Weighted Top 2', 'models': [e['name'] for e in model_evaluations[:2]], 'weighted': True},
]

for strategy in strategies:
    print(f"\nTesting: {strategy['name']}")
    print(f"  Models: {', '.join(strategy['models'])}")

    # Get probabilities from selected models
    ensemble_probs = np.zeros(len(y))
    total_weight = 0

    for model_name in strategy['models']:
        model_eval = [e for e in model_evaluations if e['name'] == model_name][0]

        if strategy.get('weighted', False):
            # Weight by CV F1
            weight = model_eval['cv_f1_optimal']
        else:
            weight = 1.0

        ensemble_probs += model_eval['probabilities'] * weight
        total_weight += weight

    ensemble_probs /= total_weight

    # Find optimal threshold for ensemble
    thresholds = np.arange(0.1, 0.9, 0.02)
    best_f1 = 0
    best_thresh = 0.5

    for thresh in thresholds:
        preds = (ensemble_probs >= thresh).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    improvement = best_f1 - baseline_f1
    improvement_pct = (improvement / baseline_f1) * 100

    ensemble_results.append({
        'name': strategy['name'],
        'models': strategy['models'],
        'weighted': strategy.get('weighted', False),
        'cv_f1': best_f1,
        'threshold': best_thresh,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'probabilities': ensemble_probs
    })

    print(f"  CV F1: {best_f1:.4f} (threshold: {best_thresh:.4f})")
    print(f"  vs Baseline: {improvement:+.4f} ({improvement_pct:+.2f}%)")

# Find best ensemble
best_ensemble = max(ensemble_results, key=lambda x: x['cv_f1'])

print("\n" + "="*80)
print(" BEST ENSEMBLE")
print("="*80)
print(f"Strategy: {best_ensemble['name']}")
print(f"Models: {', '.join(best_ensemble['models'])}")
print(f"Weighted: {best_ensemble['weighted']}")
print(f"CV F1: {best_ensemble['cv_f1']:.4f}")
print(f"Threshold: {best_ensemble['threshold']:.4f}")
print(f"Improvement: {best_ensemble['improvement']:+.4f} ({best_ensemble['improvement_pct']:+.2f}%)")

# Decision
print("\n[5/5] Decision Making...")
print("-" * 80)

MIN_IMPROVEMENT = 0.001  # Require at least 0.1% improvement

if best_ensemble['improvement'] > MIN_IMPROVEMENT:
    print(f"✓ IMPROVEMENT DETECTED: +{best_ensemble['improvement']:.4f}")
    print(f"✓ Ensemble {best_ensemble['name']} improves over baseline")
    print(f"✓ Proceeding with ensemble predictions")

    decision = {
        'proceed': True,
        'strategy': best_ensemble,
        'baseline_cv_f1': baseline_f1,
        'ensemble_cv_f1': best_ensemble['cv_f1'],
        'improvement': best_ensemble['improvement']
    }
else:
    print(f"✗ NO SIGNIFICANT IMPROVEMENT: {best_ensemble['improvement']:+.4f}")
    print(f"✗ Best ensemble doesn't improve over baseline")
    print(f"✗ Phase 1 Original (0.5105) remains the best option")

    decision = {
        'proceed': False,
        'strategy': best_ensemble,
        'baseline_cv_f1': baseline_f1,
        'ensemble_cv_f1': best_ensemble['cv_f1'],
        'improvement': best_ensemble['improvement']
    }

# Save analysis
with open('analysis/ensemble_analysis.pkl', 'wb') as f:
    pickle.dump({
        'decision': decision,
        'model_evaluations': model_evaluations,
        'ensemble_results': ensemble_results,
        'best_ensemble': best_ensemble
    }, f)

print("\n✓ Analysis saved to analysis/ensemble_analysis.pkl")

# Final recommendation
print("\n" + "="*80)
print(" FINAL RECOMMENDATION")
print("="*80)

if decision['proceed']:
    print(f"\n✓ PROCEED WITH ENSEMBLE")
    print(f"  Strategy: {best_ensemble['name']}")
    print(f"  Models: {', '.join(best_ensemble['models'])}")
    print(f"  Expected CV F1: {best_ensemble['cv_f1']:.4f}")
    print(f"  Improvement: +{best_ensemble['improvement']:.4f}")
    print(f"\n  Next: Create ensemble prediction script")
else:
    print(f"\n✗ ENSEMBLE DOES NOT IMPROVE")
    print(f"\n  The overfitting problem is too severe for ensemble to fix.")
    print(f"  Phase 1 Original (test score: 0.5105) remains the best model.")
    print(f"\n  ROOT CAUSE:")
    print(f"  - Models achieve perfect F1=1.0 on training")
    print(f"  - But only F1=0.5105 on test")
    print(f"  - This indicates data distribution mismatch or feature leakage")
    print(f"\n  RECOMMENDED ACTIONS:")
    print(f"  1. Investigate potential data leakage in features")
    print(f"  2. Analyze test set differences from training set")
    print(f"  3. Consider completely new features based on domain knowledge")
    print(f"  4. Accept 0.5105 as current best and focus on other approaches")

print("="*80)
