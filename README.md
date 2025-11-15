# MALLORN Astronomical Classification Challenge

## Project Overview

This project aims to identify Tidal Disruption Events (TDEs) from simulated LSST lightcurves using machine learning. TDEs occur when stars are torn apart by supermassive black holes, and detecting them is crucial for astronomical research.

## Challenge Description

The challenge uses the MALLORN (Many Artificial LSST Lightcurves based on Observations of Real Nuclear transients) dataset, which contains simulated lightcurves based on real observations from the Zwicky Transient Facility (ZTF).

**Objective**: Develop a machine learning algorithm to photometrically identify TDEs within a simulated LSST dataset.

**Evaluation Metric**: F1 Score (binary classification - TDE vs non-TDE)

## Project Structure

```
stars/
├── data/
│   ├── raw/              # Place your training_log.csv and test_log.csv here
│   └── processed/        # Processed features will be saved here
├── models/               # Trained models will be saved here
├── notebooks/            # Jupyter notebooks for exploration
├── src/                  # Source code
│   ├── data_loader.py    # Data loading utilities
│   ├── feature_engineer.py  # Feature engineering
│   ├── train.py          # Model training script
│   └── predict.py        # Generate predictions and submission file
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Data
Place your CSV files in the `data/raw/` directory:
- `training_log.csv` - Training data with labels
- `test_log.csv` - Test data for predictions

### 2. Train the Model
```bash
python src/train.py
```

This will:
- Load and preprocess the training data
- Engineer features from lightcurves
- Train a classification model
- Save the trained model to `models/`

### 3. Generate Predictions
```bash
python src/predict.py
```

This will:
- Load the trained model
- Generate predictions for the test set
- Create a submission file: `submission.csv`

## About TDEs (Tidal Disruption Events)

TDEs are astronomical phenomena where a star is ripped apart by the immense gravitational forces near a supermassive black hole. They are:
- Relatively rare (~100 observed)
- Scientifically valuable for studying black holes
- Challenging to identify from lightcurves alone

## Evaluation

The F1 score balances precision and recall:
- **Precision**: Avoid false positives (classifying non-TDEs as TDEs)
- **Recall**: Detect as many true TDEs as possible

This is crucial because the dataset is highly imbalanced (TDEs are rare).

## License

This project is for the MALLORN Astronomical Classification Challenge.
