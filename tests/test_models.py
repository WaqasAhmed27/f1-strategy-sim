"""Tests for model implementations."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from f1sim.models.advanced import XGBoostRegressor, CatBoostRegressorWrapper
from f1sim.models.baseline import PositionRegressor


class TestModels:
    """Test model functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 20
        
        X = pd.DataFrame({
            'GridPosition': np.random.randint(1, 21, n_samples),
            'mean_lap_time': np.random.normal(95, 2, n_samples),
            'best_lap_time': np.random.normal(94, 1.5, n_samples),
            'std_lap_time': np.random.normal(1, 0.5, n_samples),
            'n_laps': np.random.randint(50, 60, n_samples),
            'race_pace_delta': np.random.normal(0, 1, n_samples),
        })
        
        y = pd.Series(np.arange(1, n_samples + 1))  # Perfect ranking
        
        return X, y
    
    def test_xgboost_basic(self, sample_data):
        """Test XGBoost model basic functionality."""
        X, y = sample_data
        
        model = XGBoostRegressor()
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert all(pred > 0 for pred in predictions)
    
    def test_catboost_basic(self, sample_data):
        """Test CatBoost model basic functionality."""
        X, y = sample_data
        
        model = CatBoostRegressorWrapper()
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert all(pred > 0 for pred in predictions)
    
    def test_baseline_model(self, sample_data):
        """Test baseline model functionality."""
        X, y = sample_data
        
        model = PositionRegressor()
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert all(pred > 0 for pred in predictions)
    
    def test_feature_alignment(self, sample_data):
        """Test feature alignment functionality."""
        X, y = sample_data
        
        # Train with full features
        model = XGBoostRegressor()
        model.fit(X, y)
        
        # Predict with subset of features
        X_subset = X[['GridPosition', 'mean_lap_time']]
        predictions = model.predict(X_subset)
        
        assert len(predictions) == len(y)
        assert all(pred > 0 for pred in predictions)
    
    def test_model_save_load(self, sample_data, tmp_path):
        """Test model save/load functionality."""
        X, y = sample_data
        
        # Train and save
        model = XGBoostRegressor()
        model.fit(X, y)
        model_path = tmp_path / "test_model.joblib"
        model.save(model_path)
        
        # Load and predict
        loaded_model = XGBoostRegressor.load(model_path)
        predictions = loaded_model.predict(X)
        
        assert len(predictions) == len(y)
        assert all(pred > 0 for pred in predictions)


if __name__ == "__main__":
    pytest.main([__file__])
