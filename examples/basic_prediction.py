#!/usr/bin/env python3
"""
Basic F1 Race Prediction Example

This example demonstrates how to:
1. Load race data
2. Train a model
3. Make predictions
4. Display results
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from f1sim.features.builder import build_features_from_dir
from f1sim.models.advanced import XGBoostRegressor
from f1sim.models.baseline import predict_order, estimate_finish_gaps


def main():
    """Run basic prediction example."""
    print("🏎️ F1 Strategy Simulator - Basic Prediction Example")
    print("=" * 50)
    
    # Load data for 2024 Bahrain Grand Prix
    race_dir = Path("data/2024_1_R")
    
    if not race_dir.exists():
        print(f"❌ Race data not found at {race_dir}")
        print("Please run: f1sim ingest --season 2024 --race-round 1 --session R")
        return
    
    print(f"📊 Loading data from: {race_dir}")
    X, y, meta = build_features_from_dir(race_dir)
    
    print(f"✅ Loaded {len(X)} drivers with {len(X.columns)} features")
    print(f"📈 Features: {list(X.columns)[:5]}...")  # Show first 5 features
    
    # Train XGBoost model
    print("\n🤖 Training XGBoost model...")
    model = XGBoostRegressor()
    model.fit(X, y)
    print("✅ Model trained successfully!")
    
    # Make predictions
    print("\n🔮 Making predictions...")
    predictions = model.predict(X)
    
    # Convert to finishing order
    order_df = predict_order(predictions, meta)
    order_df = estimate_finish_gaps(order_df, X, race_dir)
    
    # Display results
    print("\n🏁 Predicted Finishing Order:")
    print("-" * 60)
    
    if "Abbreviation" in order_df.columns:
        display_cols = ["pred_final_pos", "Abbreviation", "TeamName", "est_gap_to_winner_s"]
        for _, row in order_df.iterrows():
            pos = int(row["pred_final_pos"])
            driver = row["Abbreviation"]
            team = row["TeamName"]
            gap = row["est_gap_to_winner_s"]
            print(f"{pos:2d}. {driver:3s} ({team:15s}) +{gap:.3f}s")
    else:
        display_cols = ["pred_final_pos", "DriverNumber", "est_gap_to_winner_s"]
        for _, row in order_df.iterrows():
            pos = int(row["pred_final_pos"])
            driver = row["DriverNumber"]
            gap = row["est_gap_to_winner_s"]
            print(f"{pos:2d}. Driver {driver:2s} +{gap:.3f}s")
    
    print("\n🎯 Example completed successfully!")


if __name__ == "__main__":
    main()
