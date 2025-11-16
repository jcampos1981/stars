"""
Prediction script for MALLORN TDE Classification Challenge

This script:
1. Loads test log file
2. Extracts features from test lightcurves
3. Loads trained model
4. Generates predictions
5. Creates submission.csv file
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from feature_engineer import LightcurveFeatureEngineer
from train import TDEClassifier


def generate_submission(object_ids: np.ndarray, predictions: np.ndarray,
                       output_path: str = "submission.csv"):
    """
    Generate submission file.

    Args:
        object_ids: Array of object IDs
        predictions: Array of predictions (0 or 1)
        output_path: Path to save submission file
    """
    # Create submission DataFrame
    submission = pd.DataFrame({
        'object_id': object_ids,
        'prediction': predictions
    })

    # Save to CSV
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission file saved to: {output_path}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Predicted TDEs: {np.sum(predictions == 1)} ({100 * np.mean(predictions == 1):.2f}%)")
    print(f"Predicted Non-TDEs: {np.sum(predictions == 0)} ({100 * np.mean(predictions == 0):.2f}%)")

    # Show first few rows
    print(f"\nFirst few predictions:")
    print(submission.head(10))

    return submission


def main():
    """Main prediction pipeline."""
    print("=" * 70)
    print(" MALLORN TDE Classification - Prediction Pipeline")
    print("=" * 70)

    # Paths
    model_path = Path("models/tde_classifier.pkl")
    features_path = Path("data/processed/test_features.csv")

    # Initialize components
    loader = DataLoader(data_dir="data/raw")
    engineer = LightcurveFeatureEngineer()

    # Check if model exists
    if not model_path.exists():
        print(f"\nERROR: Model not found at {model_path}")
        print("Please train the model first by running: python src/train.py")
        return

    # Step 1: Load test log
    print("\n[1/5] Loading test log...")
    test_log = loader.load_test_log()

    # For initial testing, you can limit to a subset
    # Uncomment the next line to use only 100 objects for quick testing
    # test_log = test_log.head(100)

    # Step 2: Extract features from lightcurves
    print("\n[2/5] Extracting features from test lightcurves...")
    print("This may take a while...")

    # Check if features already exist
    if features_path.exists():
        print(f"Loading existing features from {features_path}...")
        features_df = pd.read_csv(features_path)
        print(f"Loaded features for {len(features_df)} objects")
    else:
        features_df = engineer.process_multiple_objects(
            log_df=test_log,
            lightcurve_loader_func=loader.load_lightcurve,
            is_training=False
        )

        # Save extracted features
        features_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(features_path, index=False)
        print(f"Features saved to {features_path}")

    # Step 3: Prepare features for prediction
    print("\n[3/5] Preparing features...")

    object_ids = features_df['object_id'].values
    X_test = features_df.drop(columns=['object_id'])

    print(f"Test features shape: {X_test.shape}")

    # Step 4: Load trained model
    print("\n[4/5] Loading trained model...")
    classifier = TDEClassifier.load(str(model_path))
    print(f"Model loaded successfully: {classifier.model_type}")

    # Step 5: Make predictions
    print("\n[5/5] Making predictions...")
    predictions = classifier.predict(X_test)

    # Get prediction probabilities (useful for analysis)
    try:
        probabilities = classifier.predict_proba(X_test)
        print(f"\nPrediction confidence statistics:")
        mean_conf = np.mean([probabilities[i, predictions[i]] for i in range(len(predictions))])
        print(f"  Mean probability for predicted class: {mean_conf:.3f}")

        # Save probabilities for analysis
        prob_df = pd.DataFrame({
            'object_id': object_ids,
            'prob_non_tde': probabilities[:, 0],
            'prob_tde': probabilities[:, 1],
            'prediction': predictions
        })
        prob_df.to_csv('predictions_with_probabilities.csv', index=False)
        print(f"Detailed predictions saved to: predictions_with_probabilities.csv")

    except Exception as e:
        print(f"Could not compute probabilities: {e}")

    # Generate submission file
    print("\nGenerating submission file...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_filename = f"submission_{timestamp}.csv"

    submission = generate_submission(object_ids, predictions, output_path=submission_filename)

    # Also save a copy without timestamp for convenience
    submission.to_csv("submission.csv", index=False)
    print(f"\nSubmission also saved as: submission.csv")

    # Validate submission format
    print("\nValidating submission format...")
    sample_submission_path = Path("data/raw/sample_submission.csv")
    if sample_submission_path.exists():
        sample = pd.read_csv(sample_submission_path)
        print(f"Sample submission has {len(sample)} objects")
        print(f"Your submission has {len(submission)} objects")

        if len(submission) == len(sample):
            print("✓ Submission has correct number of objects!")
        else:
            print("⚠ Warning: Number of objects doesn't match sample submission")

    print("\n" + "=" * 70)
    print(" PREDICTION COMPLETE!")
    print("=" * 70)
    print(f"Submission file ready: {submission_filename}")
    print("You can now submit this file to the competition!")
    print("=" * 70)

    return submission


if __name__ == "__main__":
    main()
