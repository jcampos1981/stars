"""
Generate predictions using Phase 2 model (Phase 1 + temperature/color features)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from data_loader import DataLoader
from feature_engineer_phase2 import Phase2FeatureEngineer

print("="*70)
print(" MALLORN TDE Classification - Phase 2 Model Prediction")
print("="*70)
print("Model: Phase 1 + Temperature/Color Evolution Features")
print("="*70)

# Setup paths
data_dir = Path('data/raw')
models_dir = Path('models')
output_path = Path('submission.csv')

# Load model
model_path = models_dir / 'tde_classifier_phase2.pkl'
print(f"\n[1/5] Loading Phase 2 model from {model_path}...")
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
threshold = model_data['threshold']
feature_names = model_data['feature_names']

print(f"Model: {model_data['model_name']}")
print(f"Features: {model_data['n_features']}")
print(f"Phase 2 features: {model_data['n_phase2_features']}")
print(f"Threshold: {threshold:.4f}")
print(f"CV F1: {model_data['cv_f1_mean']:.4f} ± {model_data['cv_f1_std']:.4f}")

# Initialize data loader and Phase 2 feature engineer
loader = DataLoader(data_dir)
engineer = Phase2FeatureEngineer(apply_deextinction=True)

# Load test log
print(f"\n[2/5] Loading test log...")
test_log = loader.load_test_log()
print(f"Loaded {len(test_log)} test objects")

# Extract features
print(f"\n[3/5] Extracting Phase 2 features for test set...")
test_features = engineer.process_multiple_objects(
    log_df=test_log,
    lightcurve_loader_func=loader.load_lightcurve,
    is_training=False
)

print(f"Extracted features for {len(test_features)} objects")
print(f"Feature shape: {test_features.shape}")

# Prepare features
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

# Make predictions
print(f"\n[5/5] Generating predictions...")
y_proba = model.predict_proba(X_test)[:, 1]
predictions = (y_proba >= threshold).astype(int)

# Create submission file
submission = pd.DataFrame({
    'object_id': test_features['object_id'],
    'target': predictions
})

# Statistics
n_tde = predictions.sum()
n_total = len(predictions)
pct_tde = 100 * n_tde / n_total

print(f"\nPrediction Statistics:")
print(f"  Total objects: {n_total}")
print(f"  Predicted TDEs: {n_tde} ({pct_tde:.2f}%)")
print(f"  Predicted Non-TDEs: {n_total - n_tde} ({100-pct_tde:.2f}%)")

# Probability distribution
print(f"\nProbability Distribution:")
print(f"  Mean: {y_proba.mean():.4f}")
print(f"  Median: {np.median(y_proba):.4f}")
print(f"  Std: {y_proba.std():.4f}")
print(f"  Min: {y_proba.min():.4f}")
print(f"  Max: {y_proba.max():.4f}")

# Save submission
submission.to_csv(output_path, index=False)
print(f"\n{'='*70}")
print(f" PREDICTION COMPLETE!")
print(f"{'='*70}")
print(f"Submission saved to: {output_path}")
print(f"\nModel Info:")
print(f"  Strategy: Phase 2 (Phase 1 + Temp/Color Evolution)")
print(f"  CV F1: {model_data['cv_f1_mean']:.4f} ± {model_data['cv_f1_std']:.4f}")
print(f"  Threshold: {threshold:.4f}")
print(f"  Predicted TDEs: {n_tde} ({pct_tde:.2f}%)")
print(f"\nKey Phase 2 Features:")
if model_data['phase2_features']:
    for f in model_data['phase2_features'][:5]:
        print(f"  - {f}")
    if len(model_data['phase2_features']) > 5:
        print(f"  ... and {len(model_data['phase2_features']) - 5} more")
print(f"{'='*70}")
