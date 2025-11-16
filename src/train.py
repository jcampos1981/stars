"""
Training script for MALLORN TDE Classification Challenge

This script:
1. Loads training log file
2. Extracts features from lightcurves
3. Trains a classification model
4. Saves the trained model and features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced models
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available, using sklearn models")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not available, using sklearn models")

from data_loader import DataLoader
from feature_engineer import LightcurveFeatureEngineer


class TDEClassifier:
    """
    Classifier for identifying Tidal Disruption Events.
    """

    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize the classifier.

        Args:
            model_type: Type of model to use ('xgboost', 'lightgbm', 'random_forest', 'balanced_rf')
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None

    def create_model(self):
        """Create and return a classification model."""

        if self.model_type == 'xgboost' and HAS_XGB:
            print("Using XGBoost classifier")
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=10,  # Handle imbalanced data
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )

        elif self.model_type == 'lightgbm' and HAS_LGB:
            print("Using LightGBM classifier")
            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                is_unbalance=True,  # Handle imbalanced data
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )

        elif self.model_type == 'balanced_rf':
            print("Using Balanced Random Forest classifier")
            model = BalancedRandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )

        else:
            print("Using Gradient Boosting classifier")
            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )

        return model

    def train(self, X_train, y_train, use_smote: bool = False):
        """
        Train the classifier.

        Args:
            X_train: Training features
            y_train: Training labels
            use_smote: Whether to use SMOTE for handling imbalanced data
        """
        print(f"\nTraining {self.model_type} model...")
        print(f"Training set size: {len(X_train)}")
        print(f"Class distribution: Non-TDE={np.sum(y_train == 0)}, TDE={np.sum(y_train == 1)}")

        # Apply SMOTE if requested and not using balanced methods
        if use_smote and self.model_type not in ['balanced_rf']:
            print("Applying SMOTE to balance classes...")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE - Training set size: {len(X_train)}")
            print(f"Class distribution: Non-TDE={np.sum(y_train == 0)}, TDE={np.sum(y_train == 1)}")

        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        # Create and train model
        self.model = self.create_model()
        self.model.fit(X_train, y_train)

        print("Training completed!")

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Features to predict on

        Returns:
            Array of predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: Features to predict on

        Returns:
            Array of class probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)

        f1 = f1_score(y_test, y_pred)
        print(f"\nF1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-TDE', 'TDE']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        return {
            'f1_score': f1,
            'predictions': y_pred
        }

    def save(self, filepath: str):
        """Save the trained model."""
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath: str):
        """Load a trained model."""
        return joblib.load(filepath)


def main():
    """Main training pipeline."""
    print("=" * 70)
    print(" MALLORN TDE Classification - Training Pipeline")
    print("=" * 70)

    # Initialize components
    loader = DataLoader(data_dir="data/raw")
    engineer = LightcurveFeatureEngineer()

    # Step 1: Load training log
    print("\n[1/6] Loading training log...")
    train_log = loader.load_training_log()

    # For initial testing, you can limit to a subset
    # Uncomment the next line to use only 100 objects for quick testing
    # train_log = train_log.head(100)

    # Step 2: Extract features from lightcurves
    print("\n[2/6] Extracting features from lightcurves...")
    print("This may take a while...")

    features_df = engineer.process_multiple_objects(
        log_df=train_log,
        lightcurve_loader_func=loader.load_lightcurve,
        is_training=True
    )

    # Save extracted features for future use
    features_path = Path("data/processed/train_features.csv")
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(features_path, index=False)
    print(f"Features saved to {features_path}")

    # Step 3: Prepare data for modeling
    print("\n[3/6] Preparing data for modeling...")

    # Separate features and target
    X = features_df.drop(columns=['object_id', 'target'])
    y = features_df['target'].values

    print(f"Features shape: {X.shape}")
    print(f"Target distribution: Non-TDE={np.sum(y == 0)}, TDE={np.sum(y == 1)}")

    # Step 4: Split data
    print("\n[4/6] Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # Step 5: Train model
    print("\n[5/6] Training model...")

    # Try XGBoost first, fall back to other models if not available
    model_preference = ['xgboost', 'lightgbm', 'balanced_rf', 'gradient_boosting']
    classifier = None

    for model_type in model_preference:
        try:
            classifier = TDEClassifier(model_type=model_type)
            classifier.train(X_train, y_train, use_smote=False)
            break
        except Exception as e:
            print(f"Could not use {model_type}: {e}")
            continue

    if classifier is None:
        raise ValueError("Could not create any classifier!")

    # Step 6: Evaluate
    print("\n[6/6] Evaluating model...")
    metrics = classifier.evaluate(X_val, y_val)

    # Cross-validation for more robust estimate
    print("\nPerforming cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        classifier.model, X, y, cv=cv, scoring='f1', n_jobs=-1
    )
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Save model
    print("\nSaving model...")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "tde_classifier.pkl"
    classifier.save(str(model_path))

    # Save feature names for use in prediction
    feature_names_path = model_dir / "feature_names.txt"
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(X.columns.tolist()))
    print(f"Feature names saved to {feature_names_path}")

    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final F1 Score: {metrics['f1_score']:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Features saved to: {features_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
