"""
Analyze color features from V2 to identify the most important ones
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb

print("="*70)
print(" Color Features Analysis")
print("="*70)

# Load V2 features
features_path = Path('data/processed/train_features_v2.csv')
if not features_path.exists():
    print(f"ERROR: {features_path} not found!")
    print("Please run train_optimized.py first to generate V2 features")
    exit(1)

features_df = pd.read_csv(features_path)
print(f"Loaded V2 features: {features_df.shape}")

# Determine target column
target_col = 'is_tde' if 'is_tde' in features_df.columns else 'target'

# Separate features and target
X = features_df.drop(['object_id', target_col], axis=1)
y = features_df[target_col].astype(int)
X = X.fillna(X.median())

# Identify color features
color_features = [col for col in X.columns if 'color_' in col]
print(f"\nTotal color features: {len(color_features)}")
print("Color features found:")
for cf in sorted(color_features):
    print(f"  - {cf}")

# Train a simple model to get feature importance
print(f"\nTraining simple model to analyze color feature importance...")
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

model = xgb.XGBClassifier(
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X, y)

# Get feature importances
importances_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Filter color features
color_importances = importances_df[importances_df['feature'].str.contains('color_')]

print(f"\n{'='*70}")
print(" TOP COLOR FEATURES BY IMPORTANCE")
print('='*70)
print(f"\nTop 20 color features:")
print(color_importances.head(20).to_string(index=False))

print(f"\n{'='*70}")
print(" SUMMARY")
print('='*70)
print(f"\nTotal features in V2: {len(X.columns)}")
print(f"Total color features: {len(color_features)}")
print(f"Color features in top 50 overall: {len([f for f in importances_df.head(50)['feature'] if 'color_' in f])}")
print(f"Color features in top 100 overall: {len([f for f in importances_df.head(100)['feature'] if 'color_' in f])}")

# Save color importances
color_importances.to_csv('color_feature_importance.csv', index=False)
print(f"\nColor feature importances saved to: color_feature_importance.csv")

# Recommend top features
print(f"\n{'='*70}")
print(" RECOMMENDATION FOR HYBRID MODEL")
print('='*70)
top_n = 10
print(f"\nTop {top_n} color features to add to base model:")
for i, row in color_importances.head(top_n).iterrows():
    print(f"  {i+1}. {row['feature']:<40} (importance: {row['importance']:.4f})")

print(f"\n{'='*70}")
