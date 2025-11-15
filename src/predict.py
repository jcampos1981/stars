"""
Prediction script for MALLORN TDE Classification Challenge
Generates submission file with predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime

from data_loader import DataLoader
from feature_engineer import LightcurveFeatureEngineer
from train import TDEClassifier


def generate_submission(test_df: pd.DataFrame, predictions: np.ndarray, output_path: str = "submission.csv"):
    """
    Generate submission file.

    Args:
        test_df: Test DataFrame (to get IDs)
        predictions: Array of predictions (0 or 1)
        output_path: Path to save submission file
    """
    # Find ID column
    id_col = None
    for col in ['id', 'source_id', 'object_id', 'ID', 'Source_ID', 'Object_ID']:
        if col in test_df.columns:
            id_col = col
            break

    if id_col is None:
        # If no ID column found, create one
        print("Warning: No ID column found, using index as ID")
        ids = np.arange(len(predictions))
        id_col = 'id'
    else:
        ids = test_df[id_col].values

    # Create submission DataFrame
    submission = pd.DataFrame({
        id_col: ids,
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
    print("=" * 60)
    print("MALLORN TDE Classification - Prediction Pipeline")
    print("=" * 60)

    # Paths
    model_path = Path("models/tde_classifier.pkl")

    # Check if model exists
    if not model_path.exists():
        print(f"\nError: Model not found at {model_path}")
        print("Please train the model first by running: python src/train.py")
        return

    # Initialize components
    loader = DataLoader(data_dir="data/raw")
    engineer = LightcurveFeatureEngineer()

    # Load test data
    print("\n1. Loading test data...")
    test_df = loader.load_test_data()

    # Prepare data
    print("\n2. Preparing test data...")
    X_test, feature_names = loader.prepare_data(test_df, is_training=False)

    print(f"\nTest dataset shape: {X_test.shape}")
    print(f"Number of features: {len(feature_names)}")

    # Engineer features
    print("\n3. Engineering features...")
    X_test_engineered = engineer.engineer_features(X_test)

    # Load trained model
    print("\n4. Loading trained model...")
    classifier = TDEClassifier.load(str(model_path))
    print(f"Model loaded successfully: {classifier.model_type}")

    # Make predictions
    print("\n5. Making predictions...")
    predictions = classifier.predict(X_test_engineered)

    # Get prediction probabilities (useful for analysis)
    try:
        probabilities = classifier.predict_proba(X_test_engineered)
        print(f"\nPrediction confidence statistics:")
        print(f"  Mean probability for predicted class: {np.mean([probabilities[i, predictions[i]] for i in range(len(predictions))]):.3f}")

        # Save probabilities for analysis
        prob_df = pd.DataFrame({
            'prob_non_tde': probabilities[:, 0],
            'prob_tde': probabilities[:, 1],
            'prediction': predictions
        })
        prob_df.to_csv('predictions_with_probabilities.csv', index=False)
        print(f"Detailed predictions saved to: predictions_with_probabilities.csv")

    except Exception as e:
        print(f"Could not compute probabilities: {e}")

    # Generate submission file
    print("\n6. Generating submission file...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_filename = f"submission_{timestamp}.csv"

    submission = generate_submission(test_df, predictions, output_path=submission_filename)

    # Also save a copy without timestamp for convenience
    submission.to_csv("submission.csv", index=False)
    print(f"\nSubmission also saved as: submission.csv")

    print("\n" + "=" * 60)
    print("Prediction complete!")
    print(f"Submission file ready: {submission_filename}")
    print("=" * 60)

    return submission


if __name__ == "__main__":
    main()
