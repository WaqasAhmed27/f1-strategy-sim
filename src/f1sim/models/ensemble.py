"""
Ensemble Model Implementation for F1 Predictions
Combines multiple models for improved accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import joblib
import logging
from datetime import datetime

from .advanced import XGBoostRegressor, CatBoostRegressorWrapper
from .baseline import PositionRegressor
from .feature_selection import FeatureSelector

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble predictor that combines multiple models for improved accuracy.
    
    Supports:
    - Weighted averaging
    - Stacking
    - Voting
    - Dynamic weight adjustment
    """
    
    def __init__(self, models: Optional[Dict[str, Any]] = None):
        """
        Initialize ensemble predictor.
        
        Args:
            models: Dictionary of model names to model instances
        """
        self.models = models or {}
        self.weights = {}
        self.feature_selector = None
        self.selected_features = None
        self.is_fitted = False
        
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models[name] = model
        self.weights[name] = weight
        logger.info(f"Added model '{name}' with weight {weight}")
    
    def remove_model(self, name: str):
        """Remove a model from the ensemble."""
        if name in self.models:
            del self.models[name]
            del self.weights[name]
            logger.info(f"Removed model '{name}'")
    
    def set_weights(self, weights: Dict[str, float]):
        """Set weights for ensemble models."""
        self.weights.update(weights)
        logger.info(f"Updated weights: {weights}")
    
    def _normalize_weights(self) -> Dict[str, float]:
        """Normalize weights to sum to 1."""
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            # Equal weights if all weights are 0
            return {name: 1.0 / len(self.models) for name in self.models.keys()}
        return {name: weight / total_weight for name, weight in self.weights.items()}
    
    def fit(self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None):
        """
        Fit all models in the ensemble.
        
        Args:
            X: Feature matrix
            y: Target values
            groups: Group labels for cross-validation
        """
        if not self.models:
            raise ValueError("No models added to ensemble")
        
        logger.info(f"Fitting ensemble with {len(self.models)} models")
        
        # Fit each model
        for name, model in self.models.items():
            try:
                logger.info(f"Fitting model: {name}")
                model.fit(X, y)
                logger.info(f"Successfully fitted model: {name}")
            except Exception as e:
                logger.error(f"Failed to fit model {name}: {e}")
                raise
        
        self.is_fitted = True
        logger.info("Ensemble fitting completed")
    
    def predict(self, X: pd.DataFrame, meta: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Make predictions using ensemble.
        
        Args:
            X: Feature matrix
            meta: Metadata DataFrame
            
        Returns:
            DataFrame with predictions and metadata
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        if not self.models:
            raise ValueError("No models in ensemble")
        
        logger.info(f"Making ensemble predictions with {len(self.models)} models")
        
        # Get predictions from each model
        predictions = {}
        normalized_weights = self._normalize_weights()
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
                logger.debug(f"Model {name} prediction range: {pred.min():.3f} - {pred.max():.3f}")
            except Exception as e:
                logger.error(f"Prediction failed for model {name}: {e}")
                raise
        
        # Combine predictions using weighted average
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            weight = normalized_weights[name]
            ensemble_pred += weight * pred
            logger.debug(f"Model {name} weight: {weight:.3f}")
        
        # Create result DataFrame
        result = pd.DataFrame({
            'ensemble_prediction': ensemble_pred
        })
        
        # Add individual model predictions
        for name, pred in predictions.items():
            result[f'{name}_prediction'] = pred
        
        # Add metadata if provided
        if meta is not None:
            for col in meta.columns:
                result[col] = meta[col].values
        
        # Add confidence score (inverse of prediction variance)
        pred_variance = np.var(list(predictions.values()), axis=0)
        result['confidence'] = 1.0 / (1.0 + pred_variance)
        
        logger.info(f"Ensemble prediction completed. Mean confidence: {result['confidence'].mean():.3f}")
        
        return result
    
    def predict_with_uncertainty(self, X: pd.DataFrame, meta: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Make predictions with uncertainty estimates.
        
        Returns:
            Tuple of (predictions, uncertainty)
        """
        predictions = self.predict(X, meta)
        
        # Calculate uncertainty metrics
        uncertainty = pd.DataFrame(index=predictions.index)
        
        # Prediction variance across models
        model_preds = [predictions[f'{name}_prediction'] for name in self.models.keys()]
        uncertainty['prediction_variance'] = np.var(model_preds, axis=0)
        uncertainty['prediction_std'] = np.std(model_preds, axis=0)
        
        # Confidence intervals (assuming normal distribution)
        uncertainty['confidence_interval_95'] = 1.96 * uncertainty['prediction_std']
        uncertainty['confidence_interval_99'] = 2.58 * uncertainty['prediction_std']
        
        return predictions, uncertainty
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about ensemble models."""
        info = {
            'model_count': len(self.models),
            'models': list(self.models.keys()),
            'weights': self.weights.copy(),
            'normalized_weights': self._normalize_weights(),
            'is_fitted': self.is_fitted
        }
        return info
    
    def get_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self._normalize_weights()
    
    def optimize_weights(self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None):
        """
        Optimize ensemble weights using validation data.
        
        Args:
            X: Feature matrix
            y: Target values
            groups: Group labels for cross-validation
        """
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import mean_absolute_error
        
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before optimizing weights")
        
        logger.info("Optimizing ensemble weights")
        
        # Get individual model predictions
        model_preds = {}
        for name, model in self.models.items():
            model_preds[name] = model.predict(X)
        
        # Use cross-validation to find optimal weights
        if groups is not None:
            cv = GroupKFold(n_splits=min(5, groups.nunique()))
            splits = cv.split(X, y, groups)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            splits = cv.split(X, y)
        
        best_weights = None
        best_score = float('inf')
        
        # Try different weight combinations
        weight_candidates = [
            {name: 1.0 for name in self.models.keys()},  # Equal weights
            {name: 1.0 if 'xgboost' in name.lower() else 0.5 for name in self.models.keys()},  # XGBoost preference
            {name: 1.0 if 'catboost' in name.lower() else 0.5 for name in self.models.keys()},  # CatBoost preference
        ]
        
        for weights in weight_candidates:
            # Normalize weights
            total_weight = sum(weights.values())
            normalized_weights = {name: weight / total_weight for name, weight in weights.items()}
            
            cv_scores = []
            for train_idx, val_idx in splits:
                # Calculate ensemble prediction for validation set
                ensemble_pred = np.zeros(len(val_idx))
                for name, pred in model_preds.items():
                    ensemble_pred += normalized_weights[name] * pred[val_idx]
                
                # Calculate MAE
                mae = mean_absolute_error(y.iloc[val_idx], ensemble_pred)
                cv_scores.append(mae)
            
            avg_score = np.mean(cv_scores)
            logger.info(f"Weight combination {normalized_weights}: MAE = {avg_score:.3f}")
            
            if avg_score < best_score:
                best_score = avg_score
                best_weights = normalized_weights
        
        # Update weights
        self.weights = best_weights
        logger.info(f"Optimized weights: {best_weights} (MAE: {best_score:.3f})")
    
    def save(self, filepath: str | Path):
        """Save ensemble predictor."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save models separately
        models_dir = filepath.parent / f"{filepath.stem}_models"
        models_dir.mkdir(exist_ok=True)
        
        saved_models = {}
        for name, model in self.models.items():
            model_path = models_dir / f"{name}.joblib"
            model.save(model_path)
            saved_models[name] = str(model_path)
        
        # Save ensemble metadata
        ensemble_data = {
            'weights': self.weights,
            'models': saved_models,
            'is_fitted': self.is_fitted,
            'selected_features': self.selected_features,
            'saved_at': datetime.now().isoformat()
        }
        
        joblib.dump(ensemble_data, filepath)
        logger.info(f"Ensemble saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str | Path) -> 'EnsemblePredictor':
        """Load ensemble predictor."""
        filepath = Path(filepath)
        
        # Load ensemble metadata
        ensemble_data = joblib.load(filepath)
        
        # Create ensemble instance
        ensemble = cls()
        ensemble.weights = ensemble_data['weights']
        ensemble.is_fitted = ensemble_data['is_fitted']
        ensemble.selected_features = ensemble_data.get('selected_features')
        
        # Load models
        models_dir = filepath.parent / f"{filepath.stem}_models"
        for name, model_path in ensemble_data['models'].items():
            if 'xgboost' in name.lower():
                model = XGBoostRegressor.load(model_path)
            elif 'catboost' in name.lower():
                model = CatBoostRegressorWrapper.load(model_path)
            elif 'baseline' in name.lower():
                model = PositionRegressor.load(model_path)
            else:
                model = joblib.load(model_path)
            
            ensemble.models[name] = model
        
        logger.info(f"Ensemble loaded from {filepath}")
        return ensemble
    
    async def load_models(self):
        """Async method to load models for web API."""
        try:
            # Try to load existing ensemble
            ensemble_path = Path("models/ensemble_predictor.joblib")
            if ensemble_path.exists():
                loaded_ensemble = self.load(ensemble_path)
                self.models = loaded_ensemble.models
                self.weights = loaded_ensemble.weights
                self.is_fitted = loaded_ensemble.is_fitted
                logger.info("Loaded existing ensemble from disk")
            else:
                # Create default ensemble
                self.add_model("xgboost", XGBoostRegressor(), weight=0.4)
                self.add_model("catboost", CatBoostRegressorWrapper(), weight=0.4)
                self.add_model("baseline", PositionRegressor(), weight=0.2)
                logger.info("Created default ensemble")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise


def create_default_ensemble() -> EnsemblePredictor:
    """Create a default ensemble with optimized models."""
    ensemble = EnsemblePredictor()
    
    # Add models with initial weights
    ensemble.add_model("xgboost", XGBoostRegressor(), weight=0.5)
    ensemble.add_model("catboost", CatBoostRegressorWrapper(), weight=0.3)
    ensemble.add_model("baseline", PositionRegressor(), weight=0.2)
    
    return ensemble

