"""
Generate predictions using the robust model
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from data_loader import DataLoader
from feature_engineer import LightcurveFeatureEngineer  # Use original simpler features

print("="*70)
print(" MALLORN TDE Classification - Robust Prediction Pipeline")
print("="*70)

# Setup paths
data_dir = Path('data/raw')
processed_dir = Path('data/processed')
models_dir = Path('models')

# Initialize data loader and feature engineer (using ORIGINAL version)
loader = DataLoader(data_dir)
engineer = LightcurveFeatureEngineer(apply_deextinction=True)

print("\n[1/5] Loading test log...")
test_log = loader.load_test_log()
print(f"Loaded {len(test_log)} test objects")

# Load or extract features (using ORIGINAL features)
features_path = processed_dir / 'test_features.csv'
if features_path.exists():
    print(f"\n[2/5] Loading existing features from {features_path}...")
    features_df = pd.read_csv(features_path)
    print(f"Loaded features for {len(features_df)} objects")
else:
    print(f"\n[2/5] Extracting ORIGINAL features from test lightcurves...")
    print("This will take a while...")

    features_df = engineer.process_multiple_objects(
        log_df=test_log,
        lightcurve_loader_func=loader.load_lightcurve,
        is_training=False
    )

    # Save features
    features_df.to_csv(features_path, index=False)
    print(f"Features saved to {features_path}")

print(f"\n[3/5] Preparing features...")
print(f"Test features shape: {features_df.shape}")

# Separate object IDs and features
object_ids = features_df['object_id'].values
X_test = features_df.drop(['object_id'], axis=1)

# Handle missing values
X_test = X_test.fillna(X_test.median())

# Load robust model
print(f"\n[4/5] Loading robust model...")
model_path = models_dir / 'tde_classifier_robust.pkl'
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
threshold = model_data['threshold']
model_name = model_data['model_name']
selected_features = model_data['feature_names']

print(f"Model loaded: {model_name}")
print(f"Threshold: {threshold:.4f}")
print(f"CV F1 Score: {model_data['cv_f1_mean']:.4f} ± {model_data['cv_f1_std']:.4f}")
print(f"Selected features: {len(selected_features)}")

# Select only the features used during training
X_test_selected = X_test[selected_features]

print(f"Test set size: {len(X_test_selected)}")
print(f"Number of features: {X_test_selected.shape[1]}")

# Make predictions
print(f"\n[5/5] Making predictions...")

# Get probability scores
y_proba = model.predict_proba(X_test_selected)[:, 1]

# Apply threshold (default 0.5, not optimized)
predictions = (y_proba >= threshold).astype(int)

print(f"\nPrediction confidence statistics:")
print(f"  Mean probability: {np.mean(y_proba):.3f}")
print(f"  Median probability: {np.median(y_proba):.3f}")
print(f"  Min probability: {np.min(y_proba):.3f}")
print(f"  Max probability: {np.max(y_proba):.3f}")
print(f"  Std probability: {np.std(y_proba):.3f}")

# Save detailed predictions
detailed_output = pd.DataFrame({
    'object_id': object_ids,
    'prediction': predictions,
    'probability_TDE': y_proba,
    'probability_Non-TDE': 1 - y_proba
})

detailed_path = f'predictions_robust_with_probabilities.csv'
detailed_output.to_csv(detailed_path, index=False)
print(f"\nDetailed predictions saved to: {detailed_path}")

# Generate submission file
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
submission = pd.DataFrame({
    'object_id': object_ids,
    'prediction': predictions
})

submission_path = f'submission_robust_{timestamp}.csv'
submission.to_csv(submission_path, index=False)

print(f"\nSubmission file saved to: {submission_path}")
print(f"Total predictions: {len(submission)}")
print(f"Predicted TDEs: {predictions.sum()} ({predictions.sum()/len(predictions)*100:.2f}%)")
print(f"Predicted Non-TDEs: {(predictions == 0).sum()} ({(predictions == 0).sum()/len(predictions)*100:.2f}%)")

# Display sample predictions
print(f"\nFirst few predictions:")
print(submission.head(10))

# Save as main submission file
submission.to_csv('submission.csv', index=False)
print(f"\nSubmission also saved as: submission.csv")

# Validate against sample submission
sample_path = data_dir / 'sample_submission.csv'
if sample_path.exists():
    print("\nValidating submission format...")
    sample = pd.read_csv(sample_path)
    print(f"Sample submission has {len(sample)} objects")
    print(f"Your submission has {len(submission)} objects")

    if len(submission) == len(sample):
        print("✓ Submission has correct number of objects!")
    else:
        print("✗ WARNING: Submission has different number of objects!")

print("\n" + "="*70)
print(" PREDICTION COMPLETE!")
print("="*70)
print(f"\nSubmission file ready: submission.csv")
print(f"Strategy used:")
print(f"  - Robust model with strong regularization")
print(f"  - Feature selection ({len(selected_features)} features)")
print(f"  - Default threshold (0.5)")
print(f"  - CV F1: {model_data['cv_f1_mean']:.4f} ± {model_data['cv_f1_std']:.4f}")
print("="*70)
