#!/usr/bin/env python3
"""
Feature Analysis Example

This example demonstrates how to:
1. Analyze feature importance
2. Compare different feature sets
3. Find optimal features
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from f1sim.features.builder import build_features_for_races
from f1sim.models.feature_selection import FeatureSelector, find_optimal_feature_set


def main():
    """Run feature analysis example."""
    print("🔍 F1 Strategy Simulator - Feature Analysis Example")
    print("=" * 55)
    
    # Load data for multiple races
    race_ids = ["2024-1", "2024-2"]
    data_root = "data"
    
    print(f"📊 Loading data for races: {race_ids}")
    X, y, meta, groups = build_features_for_races(data_root, race_ids)
    
    print(f"✅ Loaded {len(X)} samples from {groups.nunique()} races")
    print(f"📈 Total features: {len(X.columns)}")
    
    # Initialize feature selector
    print("\n🔍 Analyzing feature importance...")
    selector = FeatureSelector(model_type="xgboost")
    
    # Run comprehensive analysis
    results = selector.comprehensive_analysis(X, y, groups)
    
    # Display feature importance table
    print("\n📊 Top 10 Most Important Features:")
    print("-" * 50)
    importance_table = selector.get_feature_importance_table(top_n=10)
    print(importance_table)
    
    # Find optimal feature set
    optimal_set, optimal_results = find_optimal_feature_set(results['evaluation_results'])
    
    print(f"\n🎯 Optimal Feature Set: {optimal_set}")
    print(f"📈 CV MAE: {optimal_results.get('cv_mae_mean', 'N/A'):.3f} ± {optimal_results.get('cv_mae_std', 'N/A'):.3f}")
    print(f"🔢 Number of features: {optimal_results.get('n_features', 'N/A')}")
    
    # Show all feature set comparisons
    print(f"\n📊 Feature Set Performance Comparison:")
    print("-" * 50)
    for set_name, eval_results in results['evaluation_results'].items():
        if 'cv_mae_mean' in eval_results:
            mae = eval_results['cv_mae_mean']
            std = eval_results['cv_mae_std']
            n_features = eval_results['n_features']
            print(f"{set_name:20s}: MAE = {mae:.3f} ± {std:.3f} ({n_features:2d} features)")
    
    print("\n🎯 Feature analysis completed successfully!")


if __name__ == "__main__":
    main()
