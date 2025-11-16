"""
Deep Error Analysis - Understanding Model Failures (using CV predictions)
Goal: Identify patterns in misclassifications to guide feature engineering
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import xgboost as xgb
from data_loader import DataLoader

print("="*70)
print(" DEEP ERROR ANALYSIS - Cross-Validation Errors")
print("="*70)
print("Note: Analyzing CV errors (not training errors) for realistic insights")
print("="*70)

# Setup paths
data_dir = Path('data/raw')
processed_dir = Path('data/processed')
models_dir = Path('models')

# Load fine-tuned model (best model with score 0.5040)
model_path = models_dir / 'tde_classifier_finetuned.pkl'
print(f"\n[1/7] Loading fine-tuned model from {model_path}...")
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

threshold = model_data['threshold']
feature_names = model_data['feature_names']

print(f"Model: {model_data['model_name']}")
print(f"CV F1: {model_data['cv_f1_mean']:.4f} ± {model_data['cv_f1_std']:.4f}")
print(f"Threshold: {threshold:.4f}")
print(f"Features: {len(feature_names)}")

# Load training data with features
features_path = processed_dir / 'train_features.csv'
print(f"\n[2/7] Loading training features from {features_path}...")
features_df = pd.read_csv(features_path)
print(f"Loaded {len(features_df)} objects")

# Load training log for metadata
loader = DataLoader(data_dir)
train_log = loader.load_training_log()
if 'target' in train_log.columns:
    train_log['is_tde'] = train_log['target']

print(f"Class distribution: TDE={train_log['is_tde'].sum()}, Non-TDE={(~train_log['is_tde']).sum()}")

# Prepare features
target_col = 'is_tde' if 'is_tde' in features_df.columns else 'target'
X = features_df.drop(['object_id', target_col], axis=1)
y = features_df[target_col].astype(int)
object_ids = features_df['object_id'].values

# Ensure same features as model
X = X[feature_names]
X = X.fillna(X.median())

# Calculate scale_pos_weight
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

# Get CV predictions (this shows realistic errors)
print(f"\n[3/7] Generating out-of-fold CV predictions...")
print("This may take a few minutes...")

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Recreate the same model used in training
cv_model = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.05,
    n_estimators=300,
    min_child_weight=5,
    gamma=0.2,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

# Get out-of-fold predictions
y_proba = cross_val_predict(cv_model, X, y, cv=cv, method='predict_proba')[:, 1]
y_pred = (y_proba >= threshold).astype(int)

# Calculate metrics
print(f"\n[4/7] Analyzing predictions...")
cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(f"               Predicted")
print(f"              Non-TDE  TDE")
print(f"Actual Non-TDE  {tn:4d}   {fp:4d}")
print(f"       TDE      {fn:4d}   {tp:4d}")

print(f"\nError Analysis:")
print(f"True Positives (TP):  {tp:4d} - TDEs correctly identified")
print(f"True Negatives (TN):  {tn:4d} - Non-TDEs correctly rejected")
print(f"False Positives (FP): {fp:4d} - Non-TDEs misclassified as TDEs")
print(f"False Negatives (FN): {fn:4d} - TDEs missed")

print(f"\nError Rates:")
print(f"False Positive Rate: {fp/(fp+tn)*100:.2f}% - {fp}/{fp+tn} Non-TDEs misclassified")
print(f"False Negative Rate: {fn/(fn+tp)*100:.2f}% - {fn}/{fn+tp} TDEs missed")

print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=['Non-TDE', 'TDE']))

# Identify error groups
tp_mask = (y == 1) & (y_pred == 1)
tn_mask = (y == 0) & (y_pred == 0)
fp_mask = (y == 0) & (y_pred == 1)
fn_mask = (y == 1) & (y_pred == 0)

print(f"\n[5/7] Analyzing error patterns...")

# Create analysis dataframe
analysis_df = pd.DataFrame({
    'object_id': object_ids,
    'true_label': y.values,
    'predicted_label': y_pred,
    'probability': y_proba,
    'error_type': 'TN'
})

analysis_df.loc[tp_mask, 'error_type'] = 'TP'
analysis_df.loc[fp_mask, 'error_type'] = 'FP'
analysis_df.loc[fn_mask, 'error_type'] = 'FN'

# Add features for detailed analysis (Z and EBV are already in features!)
X_with_id = pd.DataFrame(X, columns=feature_names)
X_with_id['object_id'] = object_ids
analysis_df = pd.merge(analysis_df, X_with_id, on='object_id', how='left')

# Add SpecType separately (not in features)
spectype_df = train_log[['object_id', 'SpecType']].copy()
analysis_df = pd.merge(analysis_df, spectype_df, on='object_id', how='left')

# Analyze probability distributions by error type
print("\n" + "="*70)
print(" PROBABILITY DISTRIBUTIONS BY ERROR TYPE")
print("="*70)

for error_type in ['TP', 'TN', 'FP', 'FN']:
    subset = analysis_df[analysis_df['error_type'] == error_type]
    if len(subset) > 0:
        print(f"\n{error_type} ({len(subset)} objects):")
        print(f"  Probability: mean={subset['probability'].mean():.4f}, "
              f"median={subset['probability'].median():.4f}, "
              f"std={subset['probability'].std():.4f}")
        print(f"  Min={subset['probability'].min():.4f}, "
              f"Max={subset['probability'].max():.4f}")

# Analyze metadata patterns (if available)
print("\n" + "="*70)
print(" METADATA ANALYSIS BY ERROR TYPE")
print("="*70)

if 'Z' in analysis_df.columns and 'EBV' in analysis_df.columns:
    for error_type in ['TP', 'FN', 'TN', 'FP']:
        subset = analysis_df[analysis_df['error_type'] == error_type]
        if len(subset) > 0:
            print(f"\n{error_type} ({len(subset)} objects):")
            print(f"  Redshift (Z): mean={subset['Z'].mean():.4f}, median={subset['Z'].median():.4f}")
            print(f"  EBV: mean={subset['EBV'].mean():.4f}, median={subset['EBV'].median():.4f}")

            # SpecType distribution
            if 'SpecType' in subset.columns:
                spectype_counts = subset['SpecType'].value_counts().head(3)
                print(f"  Top SpecTypes: {dict(spectype_counts)}")
else:
    print("Metadata not available in analysis dataframe (merge issue)")
    print("Continuing with feature analysis...")

# Analyze feature differences
print("\n" + "="*70)
print(" FEATURE ANALYSIS - FALSE NEGATIVES vs TRUE POSITIVES")
print("="*70)
print("(TDEs we MISSED vs TDEs we FOUND)")

if fn > 0 and tp > 0:
    fn_features = analysis_df[fn_mask][feature_names].mean()
    tp_features = analysis_df[tp_mask][feature_names].mean()

    # Calculate relative difference
    feature_diff = ((fn_features - tp_features) / (tp_features.abs() + 1e-10)).abs()
    top_diff_fn_tp = feature_diff.nlargest(20)

    print(f"\nTop 20 features most different between FN and TP:")
    for i, (feat, diff) in enumerate(top_diff_fn_tp.items(), 1):
        fn_val = fn_features[feat]
        tp_val = tp_features[feat]
        print(f"{i:2d}. {feat:40s} | FN={fn_val:8.4f} vs TP={tp_val:8.4f} (diff={diff:.2f}x)")

print("\n" + "="*70)
print(" FEATURE ANALYSIS - FALSE POSITIVES vs TRUE NEGATIVES")
print("="*70)
print("(Non-TDEs we MISCLASSIFIED vs Non-TDEs we correctly REJECTED)")

if fp > 0 and tn > 0:
    fp_features = analysis_df[fp_mask][feature_names].mean()
    tn_features = analysis_df[tn_mask][feature_names].mean()

    # Calculate relative difference
    feature_diff = ((fp_features - tn_features) / (tn_features.abs() + 1e-10)).abs()
    top_diff_fp_tn = feature_diff.nlargest(20)

    print(f"\nTop 20 features most different between FP and TN:")
    for i, (feat, diff) in enumerate(top_diff_fp_tn.items(), 1):
        fp_val = fp_features[feat]
        tn_val = tn_features[feat]
        print(f"{i:2d}. {feat:40s} | FP={fp_val:8.4f} vs TN={tn_val:8.4f} (diff={diff:.2f}x)")

# Save detailed error analysis
print(f"\n[6/7] Saving detailed error analysis...")
error_analysis_path = processed_dir / 'error_analysis_cv.csv'
analysis_df.to_csv(error_analysis_path, index=False)
print(f"Detailed error analysis saved to: {error_analysis_path}")

# Save specific error lists
print(f"\n[7/7] Saving specific error lists...")

# Save with available columns
cols_to_save = ['object_id', 'probability']
if 'Z' in analysis_df.columns:
    cols_to_save.extend(['Z', 'EBV', 'SpecType'])

fp_objects = analysis_df[fp_mask][cols_to_save].copy()
fn_objects = analysis_df[fn_mask][cols_to_save].copy()

fp_path = processed_dir / 'false_positives_cv.csv'
fn_path = processed_dir / 'false_negatives_cv.csv'

fp_objects.to_csv(fp_path, index=False)
fn_objects.to_csv(fn_path, index=False)

print(f"False Positives saved to: {fp_path} ({len(fp_objects)} objects)")
print(f"False Negatives saved to: {fn_path} ({len(fn_objects)} objects)")

# Summary insights
print("\n" + "="*70)
print(" KEY INSIGHTS FOR IMPROVEMENT")
print("="*70)

print(f"\n1. ERROR DISTRIBUTION:")
print(f"   - Missing {fn}/{fn+tp} TDEs ({fn/(fn+tp)*100:.1f}%) - Need better recall")
print(f"   - Misclassifying {fp}/{fp+tn} Non-TDEs ({fp/(fp+tn)*100:.1f}%) - Need better precision")

if fn > fp:
    print(f"   → PRIMARY ISSUE: False Negatives (missing real TDEs)")
    print(f"   → FOCUS: Improve recall - find more subtle TDE signatures")
else:
    print(f"   → PRIMARY ISSUE: False Positives (misclassifying Non-TDEs)")
    print(f"   → FOCUS: Improve precision - better distinguish TDEs from other transients")

print(f"\n2. PROBABILITY DISTRIBUTIONS:")
if fn > 0:
    fn_probs = analysis_df[fn_mask]['probability']
    print(f"   - False Negatives: mean prob = {fn_probs.mean():.3f}")
    print(f"     → These TDEs have LOW probabilities (model not confident)")
    print(f"     → Need features that distinguish them better")

if fp > 0:
    fp_probs = analysis_df[fp_mask]['probability']
    print(f"   - False Positives: mean prob = {fp_probs.mean():.3f}")
    print(f"     → These Non-TDEs look like TDEs to the model")
    print(f"     → Need features to separate them")

print(f"\n3. NEXT STEPS:")
print(f"   1. Review False Negatives list - what TDE characteristics are we missing?")
print(f"   2. Review False Positives list - what Non-TDEs look like TDEs?")
print(f"   3. Design new features based on the feature differences above")
print(f"   4. Focus on features with highest difference between errors and correct predictions")

print("\n" + "="*70)
print(" ANALYSIS COMPLETE!")
print("="*70)
print(f"\nFiles created:")
print(f"  - {error_analysis_path} (full analysis)")
print(f"  - {fp_path} ({len(fp_objects)} False Positives)")
print(f"  - {fn_path} ({len(fn_objects)} False Negatives)")
print("\nReview these files to understand what the model is getting wrong.")
print("="*70)
