"""
Setup script for MALLORN TDE Classification Project

This script helps you set up your environment quickly.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run setup tasks."""
    print("=" * 60)
    print("MALLORN TDE Classification - Setup")
    print("=" * 60)

    # Check Python version
    print(f"\nPython version: {sys.version}")

    if sys.version_info < (3, 8):
        print("WARNING: Python 3.8 or higher is recommended")

    # Install requirements
    print("\nInstalling requirements...")
    requirements_file = Path("requirements.txt")

    if requirements_file.exists():
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            print("\n✓ Requirements installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Error installing requirements: {e}")
            return
    else:
        print("✗ requirements.txt not found")
        return

    # Check directory structure
    print("\nChecking directory structure...")
    required_dirs = [
        "data/raw",
        "data/processed",
        "models",
        "notebooks",
        "src"
    ]

    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (creating...)")
            path.mkdir(parents=True, exist_ok=True)

    # Check for data files
    print("\nChecking for data files...")
    train_file = Path("data/raw/training_log.csv")
    test_file = Path("data/raw/test_log.csv")

    if train_file.exists():
        print(f"  ✓ Training data found")
    else:
        print(f"  ✗ Training data not found at {train_file}")
        print(f"    Please place your training_log.csv in data/raw/")

    if test_file.exists():
        print(f"  ✓ Test data found")
    else:
        print(f"  ✗ Test data not found at {test_file}")
        print(f"    Please place your test_log.csv in data/raw/")

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)

    print("\nNext steps:")
    print("1. Place your data files in data/raw/:")
    print("   - training_log.csv")
    print("   - test_log.csv")
    print("\n2. Run the complete pipeline:")
    print("   python run_pipeline.py")
    print("\n3. Or run steps individually:")
    print("   python src/train.py      # Train the model")
    print("   python src/predict.py    # Generate predictions")
    print()


if __name__ == "__main__":
    main()
