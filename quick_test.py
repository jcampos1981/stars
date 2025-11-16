"""
Quick test of the training pipeline with a small subset
"""
import sys
sys.path.insert(0, 'src')

from data_loader import DataLoader
from feature_engineer import LightcurveFeatureEngineer
from train import TDEClassifier

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("=" * 70)
print(" MALLORN TDE Classification - Quick Test (200 objects)")
print("=" * 70)

# Initialize
loader = DataLoader(data_dir="data/raw")
engineer = LightcurveFeatureEngineer(apply_deextinction=True)

# Load training log
print("\n[1/5] Loading training log...")
train_log = loader.load_training_log()

# Use only 200 objects for quick test
print(f"\nUsing subset of 200 objects from {len(train_log)} total")
train_log = train_log.head(200)

# Extract features
print("\n[2/5] Extracting features from lightcurves...")
features_df = engineer.process_multiple_objects(
    log_df=train_log,
    lightcurve_loader_func=loader.load_lightcurve,
    is_training=True
)

# Prepare data
print("\n[3/5] Preparing data...")
X = features_df.drop(columns=['object_id', 'target'])
y = features_df['target'].values

print(f"Features shape: {X.shape}")
print(f"Target distribution: Non-TDE={np.sum(y == 0)}, TDE={np.sum(y == 1)}")

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
print("\n[4/5] Training model...")
classifier = TDEClassifier(model_type='xgboost')
classifier.train(X_train, y_train, use_smote=False)

# Evaluate
print("\n[5/5] Evaluating...")
metrics = classifier.evaluate(X_val, y_val)

print("\n" + "=" * 70)
print(f" Quick test complete! F1 Score: {metrics['f1_score']:.4f}")
print("=" * 70)
print("\nIf this looks good, you can run the full pipeline with all objects.")
