#!/usr/bin/env python3
"""
Generate predictions using improved Phase 1 model (Opción A)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from data_loader import DataLoader
from feature_engineer_phase1 import Phase1FeatureEngineer

print("="*70)
print(" MALLORN TDE Classification - Improved Phase 1 Prediction")
print("="*70)
print("Model: Improved Phase 1 (Deeper Trees + Top 100 features)")
print("="*70)

# Setup paths
data_dir = Path('data/raw')
models_dir = Path('models')

# Load improved model
print("\n[1/5] Loading improved Phase 1 model...")
model_path = models_dir / 'tde_classifier_phase1_improved.pkl'
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    feature_names = model_data['feature_names']
    threshold = model_data['threshold']

print(f"Model: XGBoost (Improved Phase 1 - Deeper Trees)")
print(f"Features: {len(feature_names)}")
print(f"Threshold: {threshold:.4f}")
print(f"CV F1: {model_data['cv_f1_mean']:.4f} ± {model_data['cv_f1_std']:.4f}")
print(f"Configuration: {model_data['config_name']}")
print(f"Selection: {model_data['selection_name']}")

# Initialize data loader and Phase 1 feature engineer
loader = DataLoader(data_dir)
engineer = Phase1FeatureEngineer(apply_deextinction=True)

# Load test log
print(f"\n[2/5] Loading test log...")
test_log = loader.load_test_log()
print(f"Loaded {len(test_log)} test objects")

# Extract features
print(f"\n[3/5] Extracting Phase 1 features for test set...")
test_features = engineer.process_multiple_objects(
    log_df=test_log,
    lightcurve_loader_func=loader.load_lightcurve,
    is_training=False
)

print(f"Extracted features for {len(test_features)} objects")
print(f"Feature shape: {test_features.shape}")

# Prepare features for prediction
print(f"\n[4/5] Preparing features for prediction...")
X_test = test_features.drop(['object_id'], axis=1)

# Ensure we have the same features as training
missing_features = set(feature_names) - set(X_test.columns)
extra_features = set(X_test.columns) - set(feature_names)

if missing_features:
    print(f"WARNING: Missing {len(missing_features)} features. Adding with value 0.")
    for feat in missing_features:
        X_test[feat] = 0

if extra_features:
    print(f"Dropping {len(extra_features)} extra features not in training.")

# Reorder to match training features
X_test = X_test[feature_names]

# Handle missing values
X_test = X_test.fillna(X_test.median())

print(f"Test matrix: {X_test.shape}")

# Generate predictions
print("\n[5/5] Generating predictions...")
y_proba = model.predict_proba(X_test)[:, 1]
predictions = (y_proba >= threshold).astype(int)

# Create submission
submission = pd.DataFrame({
    'object_id': test_features['object_id'],
    'target': predictions
})

# Statistics
print("\nPrediction Statistics:")
print(f"  Total objects: {len(submission)}")
print(f"  Predicted TDEs: {predictions.sum()} ({predictions.sum()/len(predictions)*100:.2f}%)")
print(f"  Predicted Non-TDEs: {(~predictions.astype(bool)).sum()} ({(~predictions.astype(bool)).sum()/len(predictions)*100:.2f}%)")

print("\nProbability Distribution:")
print(f"  Mean: {y_proba.mean():.4f}")
print(f"  Median: {np.median(y_proba):.4f}")
print(f"  Std: {y_proba.std():.4f}")
print(f"  Min: {y_proba.min():.4f}")
print(f"  Max: {y_proba.max():.4f}")

# Save submission
output_file = 'submission.csv'
submission.to_csv(output_file, index=False)

print("\n" + "="*70)
print(" PREDICTION COMPLETE!")
print("="*70)
print(f"Submission saved to: {output_file}")
print("")
print("Model Info:")
print(f"  Strategy: Improved Phase 1 (Opción A)")
print(f"  Configuration: {model_data['config_name']}")
print(f"  Features: {len(feature_names)}")
print(f"  CV F1: {model_data['cv_f1_mean']:.4f} ± {model_data['cv_f1_std']:.4f}")
print(f"  Threshold: {threshold:.4f}")
print(f"  Predicted TDEs: {predictions.sum()} ({predictions.sum()/len(predictions)*100:.2f}%)")
print("="*70)
