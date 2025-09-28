#!/usr/bin/env python3
"""
Bulk Season Ingestion Example

This example demonstrates how to:
1. Download an entire F1 season
2. Check available data
3. Train models on bulk data
4. Make predictions across multiple races
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd: str, description: str):
    """Run a CLI command and display the result."""
    print(f"\n🔧 {description}")
    print(f"Command: {cmd}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"Failed to run command: {e}")


def main():
    """Run bulk ingestion example."""
    print("🏎️ F1 Strategy Simulator - Bulk Ingestion Example")
    print("=" * 60)
    
    # Check current data
    run_command("f1sim list-seasons", "Checking current data")
    
    # Download a few races from 2024 season
    print("\n📥 Downloading 2024 races 6-8 (race sessions only)...")
    run_command(
        "f1sim ingest-season --season 2024 --start-round 6 --end-round 8 --sessions R",
        "Bulk downloading race sessions"
    )
    
    # Download complete weekend for one race
    print("\n📥 Downloading complete weekend for race 6...")
    run_command(
        "f1sim ingest-season --season 2024 --start-round 6 --end-round 6 --sessions 'FP1,FP2,FP3,Q,R'",
        "Downloading complete race weekend"
    )
    
    # Check updated data
    run_command("f1sim list-seasons", "Checking updated data")
    
    # Train a model on multiple races
    print("\n🤖 Training model on multiple races...")
    run_command(
        "f1sim train-multi --races '2024-1,2024-2,2024-3,2024-4,2024-5,2024-6'",
        "Training model on 6 races"
    )
    
    # Make predictions
    print("\n🔮 Making predictions...")
    run_command(
        "f1sim predict --season 2024 --race-round 6",
        "Predicting race 6 results"
    )
    
    print("\n🎯 Bulk ingestion example completed!")
    print("\n💡 Tips:")
    print("  - Use 'f1sim list-seasons' to check your data")
    print("  - Use '--skip-existing true' to avoid re-downloading")
    print("  - Download specific sessions: '--sessions FP1,Q,R'")
    print("  - Download entire season: '--start-round 1 --end-round 24'")


if __name__ == "__main__":
    main()
