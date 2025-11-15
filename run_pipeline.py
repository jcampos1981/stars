"""
Complete pipeline runner for MALLORN TDE Classification

This script runs the entire pipeline:
1. Loads training data
2. Trains the model
3. Loads test data
4. Generates predictions
5. Creates submission file

Usage:
    python run_pipeline.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.train import main as train_main
from src.predict import main as predict_main


def main():
    """Run the complete pipeline."""
    print("\n" + "=" * 70)
    print(" MALLORN TDE Classification Challenge - Complete Pipeline")
    print("=" * 70 + "\n")

    # Check if data files exist
    train_file = Path("data/raw/training_log.csv")
    test_file = Path("data/raw/test_log.csv")

    if not train_file.exists():
        print(f"ERROR: Training file not found at {train_file}")
        print("\nPlease place your training_log.csv file in the data/raw/ directory")
        print("You can specify a custom path in src/data_loader.py if needed")
        return

    if not test_file.exists():
        print(f"WARNING: Test file not found at {test_file}")
        print("Will train the model, but cannot generate predictions without test data")
        print()

    # Run training
    print("\nSTEP 1/2: TRAINING THE MODEL")
    print("-" * 70)
    try:
        train_main()
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run prediction if test data exists
    if test_file.exists():
        print("\n\nSTEP 2/2: GENERATING PREDICTIONS")
        print("-" * 70)
        try:
            predict_main()
        except Exception as e:
            print(f"\nERROR during prediction: {e}")
            import traceback
            traceback.print_exc()
            return

        print("\n" + "=" * 70)
        print(" PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nYour submission file is ready: submission.csv")
        print("\nNext steps:")
        print("1. Review the submission file")
        print("2. Check predictions_with_probabilities.csv for detailed analysis")
        print("3. Submit submission.csv to the competition")
        print()
    else:
        print("\n" + "=" * 70)
        print(" TRAINING COMPLETED!")
        print("=" * 70)
        print("\nModel saved successfully.")
        print("Add test_log.csv to data/raw/ and run again to generate predictions.")
        print()


if __name__ == "__main__":
    main()
