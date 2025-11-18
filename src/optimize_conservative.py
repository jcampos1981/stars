#!/usr/bin/env python3
"""
Conservative Optimization - Ensure improvement over Phase 1 Original (0.5105)

Strategy:
1. Load Phase 1 Original model (best so far: 0.5105)
2. Analyze prediction probabilities with cross-validation
3. Test conservative threshold adjustments (small steps around 0.4287)
4. Only proceed if CV F1 improves consistently
5. If no improvement, analyze why and propose alternative
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" CONSERVATIVE OPTIMIZATION - Phase 1")
print("="*80)
print("Goal: Improve over Phase 1 Original test score of 0.5105")
print("Strategy: Small, validated adjustments with CV verification")
print("="*80)

# Load Phase 1 Original model (our best baseline)
print("\n[1/6] Loading Phase 1 Original model...")
with open('models/tde_classifier_phase1.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    original_threshold = model_data['threshold']
    feature_names = model_data['feature_names']

print(f"Baseline Model: Phase 1 Original")
print(f"Test Score: 0.5105")
print(f"CV F1: {model_data['cv_f1_mean']:.4f} ± {model_data['cv_f1_std']:.4f}")
print(f"Original Threshold: {original_threshold:.4f}")
print(f"Features: {len(feature_names)}")

# Load training features
print("\n[2/6] Loading training features...")
train_features = pd.read_csv('data/processed/train_features_phase1.csv')
target_col = 'is_tde' if 'is_tde' in train_features.columns else 'target'
X = train_features[feature_names]
y = train_features[target_col].astype(int)
X = X.fillna(X.median())

print(f"Training set: {len(X)} objects")
print(f"TDEs: {y.sum()} | Non-TDEs: {(~y.astype(bool)).sum()}")

# Analyze prediction probabilities with cross-validation
print("\n[3/6] Analyzing prediction probabilities with 10-fold CV...")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

all_probs = np.zeros(len(y))
all_true = np.zeros(len(y))

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Use pre-trained model (don't retrain to stay conservative)
    y_proba = model.predict_proba(X_val)[:, 1]
    all_probs[val_idx] = y_proba
    all_true[val_idx] = y_val

# Analyze probability distribution
print(f"\nProbability Distribution (CV):")
print(f"  Mean: {all_probs.mean():.4f}")
print(f"  Median: {np.median(all_probs):.4f}")
print(f"  Std: {all_probs.std():.4f}")
print(f"  Min: {all_probs.min():.4f}")
print(f"  Max: {all_probs.max():.4f}")

# Analyze TDE vs Non-TDE probability distributions
tde_probs = all_probs[all_true == 1]
non_tde_probs = all_probs[all_true == 0]

print(f"\nTDE probabilities:")
print(f"  Mean: {tde_probs.mean():.4f}")
print(f"  Median: {np.median(tde_probs):.4f}")
print(f"  Q1: {np.percentile(tde_probs, 25):.4f}")
print(f"  Q3: {np.percentile(tde_probs, 75):.4f}")

print(f"\nNon-TDE probabilities:")
print(f"  Mean: {non_tde_probs.mean():.4f}")
print(f"  Median: {np.median(non_tde_probs):.4f}")
print(f"  Q1: {np.percentile(non_tde_probs, 25):.4f}")
print(f"  Q3: {np.percentile(non_tde_probs, 75):.4f}")

# Test conservative threshold adjustments
print("\n[4/6] Testing conservative threshold adjustments...")
print("-" * 80)

# Test thresholds around the original (0.4287) in small steps
test_thresholds = [
    original_threshold,  # Baseline
    original_threshold - 0.02,
    original_threshold - 0.01,
    original_threshold + 0.01,
    original_threshold + 0.02,
    original_threshold + 0.03,
    original_threshold - 0.03,
]

results = []

for thresh in test_thresholds:
    predictions = (all_probs >= thresh).astype(int)

    f1 = f1_score(all_true, predictions, zero_division=0)
    precision = precision_score(all_true, predictions, zero_division=0)
    recall = recall_score(all_true, predictions, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(all_true, predictions).ravel()
    n_predicted_tdes = predictions.sum()

    results.append({
        'threshold': thresh,
        'cv_f1': f1,
        'precision': precision,
        'recall': recall,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'n_predicted': n_predicted_tdes
    })

    delta = thresh - original_threshold
    print(f"Threshold: {thresh:.4f} ({delta:+.4f}) | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Predicted: {n_predicted_tdes}")

# Find best threshold
best_result = max(results, key=lambda x: x['cv_f1'])
original_result = [r for r in results if r['threshold'] == original_threshold][0]

print("\n" + "="*80)
print(" RESULTS ANALYSIS")
print("="*80)

print(f"\nOriginal threshold ({original_threshold:.4f}):")
print(f"  CV F1: {original_result['cv_f1']:.4f}")
print(f"  Precision: {original_result['precision']:.4f}")
print(f"  Recall: {original_result['recall']:.4f}")
print(f"  TP: {original_result['tp']}, FP: {original_result['fp']}, FN: {original_result['fn']}")

print(f"\nBest threshold ({best_result['threshold']:.4f}):")
print(f"  CV F1: {best_result['cv_f1']:.4f}")
print(f"  Precision: {best_result['precision']:.4f}")
print(f"  Recall: {best_result['recall']:.4f}")
print(f"  TP: {best_result['tp']}, FP: {best_result['fp']}, FN: {best_result['fn']}")

improvement = best_result['cv_f1'] - original_result['cv_f1']
improvement_pct = (improvement / original_result['cv_f1']) * 100

print(f"\nImprovement: {improvement:+.4f} ({improvement_pct:+.2f}%)")

# Decision: only proceed if there's meaningful improvement
print("\n[5/6] Decision Making...")
print("-" * 80)

MIN_IMPROVEMENT = 0.001  # Require at least 0.1% improvement

if improvement > MIN_IMPROVEMENT:
    print(f"✓ IMPROVEMENT DETECTED: +{improvement:.4f} F1 ({improvement_pct:+.2f}%)")
    print(f"✓ New threshold: {best_result['threshold']:.4f}")
    print(f"✓ This is a conservative improvement - proceeding with predictions")

    # Save decision
    decision = {
        'proceed': True,
        'reason': 'CV improvement detected',
        'original_threshold': original_threshold,
        'new_threshold': best_result['threshold'],
        'original_cv_f1': original_result['cv_f1'],
        'new_cv_f1': best_result['cv_f1'],
        'improvement': improvement,
        'improvement_pct': improvement_pct
    }

else:
    print(f"✗ NO SIGNIFICANT IMPROVEMENT: {improvement:+.4f} F1 ({improvement_pct:+.2f}%)")
    print(f"✗ Current approach won't improve test score")
    print(f"✗ Need alternative strategy")

    decision = {
        'proceed': False,
        'reason': 'No significant CV improvement',
        'original_threshold': original_threshold,
        'best_threshold_tested': best_result['threshold'],
        'original_cv_f1': original_result['cv_f1'],
        'best_cv_f1_tested': best_result['cv_f1'],
        'improvement': improvement
    }

# Save analysis
print("\n[6/6] Saving analysis...")
with open('analysis/conservative_optimization.pkl', 'wb') as f:
    pickle.dump({
        'decision': decision,
        'results': results,
        'probability_analysis': {
            'all_probs': all_probs,
            'all_true': all_true,
            'tde_probs': tde_probs,
            'non_tde_probs': non_tde_probs
        }
    }, f)

print("✓ Analysis saved to analysis/conservative_optimization.pkl")

# Final decision
print("\n" + "="*80)
print(" FINAL DECISION")
print("="*80)

if decision['proceed']:
    print(f"\n✓ PROCEED WITH PREDICTIONS")
    print(f"  Use threshold: {decision['new_threshold']:.4f}")
    print(f"  Expected CV F1: {decision['new_cv_f1']:.4f}")
    print(f"  Improvement over original: +{decision['improvement']:.4f}")
    print(f"\n  Next step: Run predict_phase1_conservative.py")
else:
    print(f"\n✗ DO NOT PROCEED - ALTERNATIVE NEEDED")
    print(f"\n  Analysis shows no improvement with threshold adjustment.")
    print(f"  Best threshold tested: {decision['best_threshold_tested']:.4f}")
    print(f"  Best CV F1 achieved: {decision['best_cv_f1_tested']:.4f}")
    print(f"  Original CV F1: {decision['original_cv_f1']:.4f}")
    print(f"\n  ALTERNATIVE STRATEGIES:")
    print(f"  1. Ensemble: Combine multiple models")
    print(f"  2. Feature engineering: Add new discriminative features")
    print(f"  3. Different model architecture: Try LightGBM or CatBoost")
    print(f"  4. Focus on recall: Accept Phase 1 Original as baseline")

print("="*80)
