"""
Analyze current model performance and identify improvement opportunities
"""
import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path to import modules
sys.path.insert(0, str(Path(__file__).parent))
from train import TDEClassifier

def analyze_feature_importance(model_path='models/tde_classifier.pkl',
                               features_path='data/processed/train_features.csv'):
    """Analyze feature importance from trained model"""

    print("="*70)
    print(" MODEL ANALYSIS")
    print("="*70)

    # Load model
    print("\n[1/4] Loading model...")
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)

    model = classifier.model
    print(f"Model type: {type(model).__name__}")

    # Load features
    print("\n[2/4] Loading training features...")
    train_df = pd.read_csv(features_path)

    # Remove target and ID columns
    feature_cols = [col for col in train_df.columns if col not in ['object_id', 'is_tde']]
    print(f"Total features: {len(feature_cols)}")

    # Get feature importance
    print("\n[3/4] Analyzing feature importance...")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_

        print(f"Number of features in dataset: {len(feature_cols)}")
        print(f"Number of feature importances: {len(importances)}")

        # Match feature names with importances
        # Handle case where there might be a mismatch
        if len(importances) != len(feature_cols):
            print(f"WARNING: Feature count mismatch!")
            min_len = min(len(importances), len(feature_cols))
            feature_cols = feature_cols[:min_len]
            importances = importances[:min_len]

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\nTop 20 most important features:")
        print(importance_df.head(20).to_string(index=False))

        # Group by filter
        print("\n\nFeature importance by filter:")
        for filter_name in ['u', 'g', 'r', 'i', 'z', 'y']:
            filter_features = importance_df[importance_df['feature'].str.contains(f'_{filter_name}_')]
            if len(filter_features) > 0:
                total_importance = filter_features['importance'].sum()
                print(f"{filter_name}: {total_importance:.4f} ({len(filter_features)} features)")

        # Cross-filter features
        cross_features = importance_df[importance_df['feature'].str.contains('cross_')]
        if len(cross_features) > 0:
            print(f"Cross-filter: {cross_features['importance'].sum():.4f} ({len(cross_features)} features)")

        # Metadata features
        meta_features = importance_df[~importance_df['feature'].str.contains('_')]
        if len(meta_features) > 0:
            print(f"Metadata: {meta_features['importance'].sum():.4f} ({len(meta_features)} features)")

        # Skip plotting for now (matplotlib not installed)

        # Save full importance to CSV
        importance_df.to_csv('feature_importance.csv', index=False)
        print("Full feature importance saved to: feature_importance.csv")

    # Analyze class distribution
    print("\n[4/4] Analyzing class distribution...")
    if 'is_tde' in train_df.columns:
        tde_count = train_df['is_tde'].sum()
        total = len(train_df)
        print(f"\nTraining set:")
        print(f"  TDE: {tde_count} ({tde_count/total*100:.2f}%)")
        print(f"  Non-TDE: {total-tde_count} ({(total-tde_count)/total*100:.2f}%)")
        print(f"  Class imbalance ratio: {(total-tde_count)/tde_count:.2f}:1")

    print("\n" + "="*70)
    print(" RECOMMENDATIONS FOR IMPROVEMENT")
    print("="*70)
    print("\n1. FEATURE ENGINEERING:")
    print("   - Add color evolution features (flux ratios between filters over time)")
    print("   - Add periodicity detection (Lomb-Scargle periodogram)")
    print("   - Add rise/decline rate features")
    print("   - Add wavelets or Fourier features")

    print("\n2. MODEL OPTIMIZATION:")
    print("   - Hyperparameter tuning (learning_rate, max_depth, n_estimators)")
    print("   - Adjust scale_pos_weight based on class imbalance")
    print("   - Try ensemble of multiple models")

    print("\n3. THRESHOLD OPTIMIZATION:")
    print("   - Find optimal classification threshold for F1 score")
    print("   - Consider precision-recall tradeoff")

    print("\n4. DATA QUALITY:")
    print("   - Better handling of missing observations per filter")
    print("   - Outlier detection and treatment")
    print("   - Feature selection to remove noisy features")
    print("="*70)

if __name__ == "__main__":
    analyze_feature_importance()
