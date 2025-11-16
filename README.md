# MALLORN Astronomical Classification Challenge

## Project Overview

This project identifies **Tidal Disruption Events (TDEs)** from simulated LSST lightcurves using machine learning. TDEs occur when stars are ripped apart by supermassive black holes, and detecting them is crucial for astronomical research.

## Challenge Description

The MALLORN (Many Artificial LSST Lightcurves based on Observations of Real Nuclear transients) dataset contains simulated lightcurves based on real observations from the Zwicky Transient Facility (ZTF).

**Objective**: Develop a machine learning algorithm to photometrically identify TDEs within a simulated LSST dataset.

**Evaluation Metric**: F1 Score (binary classification - TDE vs non-TDE)

## Dataset Structure

The MALLORN dataset has a complex structure:

```
data/raw/
├── train_log.csv              # Training metadata (object_id, Z, EBV, SpecType, target)
├── test_log.csv               # Test metadata (no target column)
├── sample_submission.csv      # Example submission format
├── split_01/                  # Lightcurve data split 1
│   ├── train_full_lightcurves.csv
│   └── test_full_lightcurves.csv
├── split_02/                  # Lightcurve data split 2
│   ├── train_full_lightcurves.csv
│   └── test_full_lightcurves.csv
...
└── split_20/                  # Lightcurve data split 20
    ├── train_full_lightcurves.csv
    └── test_full_lightcurves.csv
```

### File Descriptions

- **train_log.csv / test_log.csv**: Object metadata
  - `object_id`: Unique identifier (Sindarin words)
  - `Z`: Redshift
  - `Z_err`: Redshift error (only for test data)
  - `EBV`: Extinction coefficient
  - `SpecType`: Spectroscopic type (training only)
  - `split`: Which split folder contains this object's lightcurve
  - `target`: Binary label (1=TDE, 0=non-TDE, training only)

- **Lightcurve files**: Time series data
  - `object_id`: Links to log file
  - `Time (MJD)`: Observation time in Modified Julian Date
  - `Flux`: Flux measurement in microjansky (μJy)
  - `Flux_err`: Flux uncertainty
  - `Filter`: LSST filter (u, g, r, i, z, y)

## Project Structure

```
stars/
├── data/
│   ├── raw/                   # Place MALLORN dataset here
│   │   ├── train_log.csv
│   │   ├── test_log.csv
│   │   ├── sample_submission.csv
│   │   └── split_01/ ... split_20/
│   └── processed/             # Extracted features saved here
│       ├── train_features.csv
│       └── test_features.csv
├── models/                    # Trained models saved here
│   ├── tde_classifier.pkl
│   └── feature_names.txt
├── notebooks/                 # Jupyter notebooks for exploration
│   └── 01_data_exploration.ipynb
├── src/                       # Source code
│   ├── data_loader.py         # Load logs and lightcurves
│   ├── feature_engineer.py    # Extract features from time series
│   ├── train.py               # Train classification model
│   └── predict.py             # Generate predictions
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── QUICKSTART.md              # Quick start guide
└── run_pipeline.py            # Run complete pipeline
```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/jcampos1981/stars.git
cd stars
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Or use the setup script:
```bash
python setup.py
```

3. **Place the MALLORN dataset** in `data/raw/`:
   - Copy all files and folders from the MALLORN dataset
   - Ensure you have train_log.csv, test_log.csv, and all split folders

## Usage

### Quick Start

Run the complete pipeline:
```bash
python run_pipeline.py
```

This will:
1. ✅ Load training data
2. ✅ Extract features from lightcurves (takes time!)
3. ✅ Train a classification model
4. ✅ Generate predictions on test data
5. ✅ Create `submission.csv`

### Step-by-Step

#### 1. Train the Model

```bash
python src/train.py
```

This will:
- Load `train_log.csv`
- Extract features from lightcurves in all split folders
- Train an XGBoost/LightGBM/RandomForest classifier
- Save model to `models/tde_classifier.pkl`
- Save features to `data/processed/train_features.csv`

**Note**: Feature extraction can take time (depends on dataset size). Features are cached for reuse.

#### 2. Generate Predictions

```bash
python src/predict.py
```

This will:
- Load `test_log.csv`
- Extract features from test lightcurves
- Load trained model
- Generate predictions
- Create `submission.csv`

### Testing with Subset

For quick testing, edit `src/train.py` or `src/predict.py` and uncomment:

```python
# Use only 100 objects for testing
train_log = train_log.head(100)
```

## Feature Engineering

The pipeline extracts **~180 features** per object:

### Per-filter features (6 filters × 29 features = 174):
- **Statistical**: mean, std, median, min, max, range, skewness, kurtosis, percentiles
- **Temporal**: duration, observation count, cadence statistics
- **Rise/Fall**: peak flux, rise/fall times and rates
- **Signal quality**: SNR (signal-to-noise ratio), variability

### Cross-filter features:
- Total observations across all filters
- Number of filters with data
- Overall flux statistics

### Metadata features:
- Redshift (Z)
- Redshift error (Z_err)
- Extinction coefficient (EBV)

## Model

Default model: **XGBoost** (falls back to LightGBM or sklearn if unavailable)

Key features:
- Handles imbalanced data (TDEs are rare)
- `scale_pos_weight=10` to boost TDE detection
- Cross-validation for robust evaluation
- F1 score optimization

### Customization

Edit `src/train.py` to change model or hyperparameters:

```python
classifier = TDEClassifier(model_type='xgboost')  # or 'lightgbm', 'balanced_rf'

# Adjust hyperparameters in create_model()
model = xgb.XGBClassifier(
    n_estimators=200,      # Number of trees
    max_depth=6,           # Tree depth
    learning_rate=0.05,    # Learning rate
    ...
)
```

## Output Files

After running the pipeline:

```
submission.csv                          # ← Submit to competition
submission_YYYYMMDD_HHMMSS.csv         # Timestamped copy
predictions_with_probabilities.csv      # Detailed predictions with confidence
models/tde_classifier.pkl              # Trained model
data/processed/train_features.csv       # Extracted training features
data/processed/test_features.csv        # Extracted test features
```

## Submission Format

`submission.csv` format:

```csv
object_id,prediction
Eluwaith_Mithrim_nothrim,0
Eru_heledir_archam,1
Gonhir_anann_fuin,0
...
```

Where:
- `object_id`: Object identifier from test_log.csv
- `prediction`: 0 = Non-TDE, 1 = TDE

## About TDEs (Tidal Disruption Events)

TDEs occur when a star ventures too close to a supermassive black hole:
- Star is torn apart by tidal forces
- Produces a bright flare of electromagnetic radiation
- Relatively rare (~100 observed to date)
- Scientifically valuable for studying black holes
- LSST expected to discover many more

Types in MALLORN dataset:
- TDEs (target class)
- Supernovae (SN Ia, Ib, Ic, II, etc.)
- Superluminous supernovae (SLSN)
- Active Galactic Nuclei (AGN)

## Evaluation Metric

**F1 Score** balances precision and recall:

```
F1 = 2 × (precision × recall) / (precision + recall)
```

Where:
- **Precision** = TP / (TP + FP) - Avoid false TDE identifications
- **Recall** = TP / (TP + FN) - Detect as many true TDEs as possible

F1 is ideal for imbalanced datasets where TDEs are rare.

## Troubleshooting

### "FileNotFoundError: train_log.csv"
- Ensure dataset is in `data/raw/`
- Check file names match exactly

### "Memory Error" during feature extraction
- Process subset first: `train_log = train_log.head(1000)`
- Extract features in batches
- Use a machine with more RAM

### Low F1 Score
- Check class balance in training data
- Try different models (`balanced_rf`, `lightgbm`)
- Tune hyperparameters
- Add more features in `feature_engineer.py`

## Performance Tips

1. **Cache features**: Features are saved to `data/processed/` - reuse them!
2. **Parallel processing**: Models use `n_jobs=-1` for all CPU cores
3. **Start small**: Test with 100-1000 objects first
4. **Monitor progress**: tqdm progress bars show feature extraction status

## Development

Explore data:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

Test individual modules:
```bash
python src/data_loader.py      # Test data loading
python src/feature_engineer.py  # Test feature extraction
```

## Contributing

This is a competition project. Feel free to:
- Experiment with different models
- Add new features
- Improve feature engineering
- Optimize hyperparameters

## License

Created for the MALLORN Astronomical Classification Challenge.

## Acknowledgements

- Zwicky Transient Facility (ZTF) for original observations
- SNCosmo package for data generation
- Rubin Survey Simulator for LSST cadence simulation

---

**Good luck identifying those TDEs!** 🌟🕳️
