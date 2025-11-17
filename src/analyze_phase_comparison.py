#!/usr/bin/env python3
"""
Deep Error Analysis - Compare Phase 1 vs Phase 2 predictions
Understand why Phase 2 (0.4927) performed worse than Phase 1 (0.5105)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def load_model_info(model_path):
    """Load model metadata"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def analyze_prediction_differences(pred1_path, pred2_path, output_file='analysis/phase_comparison.txt'):
    """
    Compare predictions between Phase 1 and Phase 2
    Identify which objects changed predictions and analyze patterns
    """

    # Load predictions
    print("Loading predictions...")
    phase1 = pd.read_csv(pred1_path)
    phase2 = pd.read_csv(pred2_path)

    # Merge predictions
    comparison = phase1.merge(phase2, on='object_id', suffixes=('_phase1', '_phase2'))

    # Identify changes
    comparison['changed'] = comparison['target_phase1'] != comparison['target_phase2']
    comparison['tp1_to_fp2'] = (comparison['target_phase1'] == 1) & (comparison['target_phase2'] == 0)  # Phase 1 predicted TDE, Phase 2 didn't
    comparison['fp1_to_tp2'] = (comparison['target_phase1'] == 0) & (comparison['target_phase2'] == 1)  # Phase 1 didn't, Phase 2 did

    # Load model info
    print("\nLoading model information...")
    model1_info = load_model_info('models/tde_classifier_phase1.pkl')
    model2_info = load_model_info('models/tde_classifier_phase2.pkl')

    # Create output directory
    Path('analysis').mkdir(exist_ok=True)

    # Generate report
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(" PHASE 1 vs PHASE 2 COMPARISON ANALYSIS\n")
        f.write("="*80 + "\n\n")

        # Model comparison
        f.write("MODEL COMPARISON:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Phase 1:\n")
        f.write(f"  Test Score: 0.5105\n")
        f.write(f"  CV F1: {model1_info['cv_f1_mean']:.4f} ± {model1_info['cv_f1_std']:.4f}\n")
        f.write(f"  Features: {len(model1_info['feature_names'])} total\n")
        f.write(f"  Threshold: {model1_info['threshold']:.4f}\n\n")

        f.write(f"Phase 2:\n")
        f.write(f"  Test Score: 0.4927\n")
        f.write(f"  CV F1: {model2_info['cv_f1_mean']:.4f} ± {model2_info['cv_f1_std']:.4f}\n")
        f.write(f"  Features: {len(model2_info['feature_names'])} total ({len(model2_info.get('phase2_features', []))} Phase 2)\n")
        f.write(f"  Threshold: {model2_info['threshold']:.4f}\n")
        f.write(f"  Score Difference: {0.4927 - 0.5105:.4f} ({((0.4927/0.5105 - 1)*100):.2f}%)\n\n")

        # Prediction statistics
        f.write("\nPREDICTION STATISTICS:\n")
        f.write("-" * 80 + "\n")
        total = len(comparison)
        phase1_tdes = comparison['target_phase1'].sum()
        phase2_tdes = comparison['target_phase2'].sum()

        f.write(f"Total test objects: {total}\n\n")
        f.write(f"Phase 1 predictions:\n")
        f.write(f"  TDEs predicted: {phase1_tdes} ({phase1_tdes/total*100:.2f}%)\n")
        f.write(f"  Non-TDEs predicted: {total - phase1_tdes} ({(total-phase1_tdes)/total*100:.2f}%)\n\n")

        f.write(f"Phase 2 predictions:\n")
        f.write(f"  TDEs predicted: {phase2_tdes} ({phase2_tdes/total*100:.2f}%)\n")
        f.write(f"  Non-TDEs predicted: {total - phase2_tdes} ({(total-phase2_tdes)/total*100:.2f}%)\n\n")

        # Changes analysis
        f.write("\nPREDICTION CHANGES:\n")
        f.write("-" * 80 + "\n")
        n_changed = comparison['changed'].sum()
        n_tp1_to_fp2 = comparison['tp1_to_fp2'].sum()
        n_fp1_to_tp2 = comparison['fp1_to_tp2'].sum()

        f.write(f"Total predictions changed: {n_changed} ({n_changed/total*100:.2f}%)\n")
        f.write(f"  Phase 1 TDE → Phase 2 Non-TDE: {n_tp1_to_fp2} ({n_tp1_to_fp2/total*100:.2f}%)\n")
        f.write(f"  Phase 1 Non-TDE → Phase 2 TDE: {n_fp1_to_tp2} ({n_fp1_to_tp2/total*100:.2f}%)\n")
        f.write(f"  Net change in TDEs: {phase2_tdes - phase1_tdes} ({(phase2_tdes - phase1_tdes)/phase1_tdes*100:.2f}%)\n\n")

        # Hypothesis about what went wrong
        f.write("\nHYPOTHESIS - WHY PHASE 2 PERFORMED WORSE:\n")
        f.write("-" * 80 + "\n")

        if phase2_tdes < phase1_tdes:
            f.write(f"✗ Phase 2 became MORE CONSERVATIVE (fewer TDEs predicted)\n")
            f.write(f"  - Lost {phase1_tdes - phase2_tdes} TDE predictions\n")
            f.write(f"  - This suggests Phase 2 features may be filtering out true TDEs\n")
            f.write(f"  - Color evolution features might be too restrictive\n")
        else:
            f.write(f"✗ Phase 2 became MORE AGGRESSIVE (more TDEs predicted)\n")
            f.write(f"  - Added {phase2_tdes - phase1_tdes} TDE predictions\n")
            f.write(f"  - This suggests Phase 2 features may be causing false positives\n")
            f.write(f"  - Color evolution features might not be discriminative enough\n")

        f.write(f"\n")

        # CV vs Test comparison
        cv_test_gap1 = 0.5105 - model1_info['cv_f1_mean']
        cv_test_gap2 = 0.4927 - model2_info['cv_f1_mean']

        f.write(f"CV-Test Gap Analysis:\n")
        f.write(f"  Phase 1: CV {model1_info['cv_f1_mean']:.4f} vs Test 0.5105 (gap: {cv_test_gap1:+.4f})\n")
        f.write(f"  Phase 2: CV {model2_info['cv_f1_mean']:.4f} vs Test 0.4927 (gap: {cv_test_gap2:+.4f})\n")

        if abs(cv_test_gap2) > abs(cv_test_gap1):
            f.write(f"\n  ⚠ Phase 2 has WORSE generalization gap\n")

        # Phase 2 features analysis
        f.write(f"\n\nPHASE 2 FEATURES ANALYSIS:\n")
        f.write("-" * 80 + "\n")
        phase2_features = model2_info.get('phase2_features', [])
        f.write(f"Phase 2 features selected: {len(phase2_features)} out of 44 total\n")
        f.write(f"Selection rate: {len(phase2_features)/44*100:.1f}%\n\n")

        if len(phase2_features) > 0:
            f.write("Selected Phase 2 features:\n")
            for feat in sorted(phase2_features):
                f.write(f"  - {feat}\n")

        # Recommendations
        f.write("\n\nRECOMMENDATIONS:\n")
        f.write("-" * 80 + "\n")
        f.write("Based on this analysis, consider:\n\n")

        f.write("Option A - ROLLBACK TO PHASE 1:\n")
        f.write("  ✓ Phase 1 achieved better test score (0.5105)\n")
        f.write("  ✓ More stable model with better generalization\n")
        f.write("  → Focus on improving Phase 1 with hyperparameter tuning\n")
        f.write("  → Try different thresholds or ensemble methods\n\n")

        f.write("Option B - FIX PHASE 2:\n")
        f.write("  → Investigate which Phase 2 features are problematic\n")
        f.write("  → Remove low-impact Phase 2 features\n")
        f.write("  → Adjust feature engineering (less strict color evolution)\n")
        f.write("  → Retrain with different regularization\n\n")

        f.write("Option C - SKIP TO PHASE 3:\n")
        f.write("  → Try Multi-band coherence features instead\n")
        f.write("  → Color features might not be the right approach\n")
        f.write("  → Test if Phase 3 works better on top of Phase 1\n\n")

    print(f"\nAnalysis saved to {output_file}")

    # Save changed predictions for further investigation
    changed_objects = comparison[comparison['changed']][['object_id', 'target_phase1', 'target_phase2', 'tp1_to_fp2', 'fp1_to_tp2']]
    changed_objects.to_csv('analysis/changed_predictions.csv', index=False)
    print(f"Changed predictions saved to analysis/changed_predictions.csv ({len(changed_objects)} objects)")

    # Summary statistics
    print("\n" + "="*80)
    print(" QUICK SUMMARY")
    print("="*80)
    print(f"Phase 1 score: 0.5105 | Phase 2 score: 0.4927 | Difference: -0.0178 (-3.49%)")
    print(f"Phase 1 TDEs: {phase1_tdes} ({phase1_tdes/total*100:.2f}%) | Phase 2 TDEs: {phase2_tdes} ({phase2_tdes/total*100:.2f}%)")
    print(f"Predictions changed: {n_changed} ({n_changed/total*100:.2f}%)")
    print(f"  Phase 1 TDE → Phase 2 Non-TDE: {n_tp1_to_fp2}")
    print(f"  Phase 1 Non-TDE → Phase 2 TDE: {n_fp1_to_tp2}")
    print("="*80)

if __name__ == "__main__":
    print("="*80)
    print(" PHASE COMPARISON ANALYSIS")
    print("="*80)

    # Wait for Phase 1 predictions to be ready
    phase1_pred = 'submission_phase1.csv'
    phase2_pred = 'submission_phase2.csv'

    if not Path(phase1_pred).exists():
        print(f"\n⚠ Waiting for Phase 1 predictions: {phase1_pred}")
        print("Please run: python src/predict_phase1.py")
        exit(1)

    if not Path(phase2_pred).exists():
        print(f"\n⚠ Phase 2 predictions not found: {phase2_pred}")
        print("Please ensure submission_phase2.csv exists")
        exit(1)

    analyze_prediction_differences(phase1_pred, phase2_pred)
    print("\n✓ Analysis complete!")
