from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class XGBoostRegressor:
    """XGBoost-based position regressor with feature alignment."""
    
    def __init__(self, **kwargs) -> None:
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        
        # Default XGBoost parameters optimized for F1 prediction
        default_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        default_params.update(kwargs)
        
        self.model = XGBRegressor(**default_params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if hasattr(self.model, "feature_names_in_"):
            required = list(self.model.feature_names_in_)
            Xc = X.copy()
            # Add missing with sensible defaults
            for col in required:
                if col not in Xc.columns:
                    Xc[col] = 0.0
            # Extra columns are ignored by selecting required order
            Xc = Xc[required]
            return Xc
        return X

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_aligned = self._align_features(X)
        return self.model.predict(X_aligned)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @staticmethod
    def load(path: str | Path) -> "XGBoostRegressor":
        mdl = XGBoostRegressor()
        mdl.model = joblib.load(path)
        return mdl


class CatBoostRegressorWrapper:
    """CatBoost-based position regressor with feature alignment."""
    
    def __init__(self, **kwargs) -> None:
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not available. Install with: pip install catboost")
        
        # Default CatBoost parameters optimized for F1 prediction
        default_params = {
            'iterations': 200,
            'depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bylevel': 0.8,
            'random_seed': 42,
            'thread_count': -1,
            'verbose': False
        }
        default_params.update(kwargs)
        
        self.model = CatBoostRegressor(**default_params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if hasattr(self.model, "feature_names_"):
            required = list(self.model.feature_names_)
            Xc = X.copy()
            # Add missing with sensible defaults
            for col in required:
                if col not in Xc.columns:
                    Xc[col] = 0.0
            # Extra columns are ignored by selecting required order
            Xc = Xc[required]
            return Xc
        return X

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_aligned = self._align_features(X)
        return self.model.predict(X_aligned)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @staticmethod
    def load(path: str | Path) -> "CatBoostRegressorWrapper":
        mdl = CatBoostRegressorWrapper()
        mdl.model = joblib.load(path)
        return mdl


def cross_val_mae_advanced(X: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None, 
                          n_splits: int = 5, model_type: str = "xgboost") -> float:
    """Cross-validation MAE for advanced models."""
    gkf = GroupKFold(n_splits=n_splits) if groups is not None else None
    maes: list[float] = []
    
    if gkf is None:
        for _ in range(n_splits):
            if model_type == "xgboost":
                mdl = XGBoostRegressor()
            elif model_type == "catboost":
                mdl = CatBoostRegressorWrapper()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            mdl.fit(X, y)
            y_pred = mdl.predict(X)
            maes.append(mean_absolute_error(y, y_pred))
    else:
        for tr_idx, te_idx in gkf.split(X, y, groups):
            if model_type == "xgboost":
                mdl = XGBoostRegressor()
            elif model_type == "catboost":
                mdl = CatBoostRegressorWrapper()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            mdl.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            y_pred = mdl.predict(X.iloc[te_idx])
            maes.append(mean_absolute_error(y.iloc[te_idx], y_pred))
    
    return float(np.mean(maes))


def compare_models(X: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None, 
                  n_splits: int = 5) -> dict[str, float]:
    """Compare performance of different models."""
    results = {}
    
    # Gradient Boosting (baseline)
    from .baseline import cross_val_mae
    results["gradient_boosting"] = cross_val_mae(X, y, groups, n_splits)
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        results["xgboost"] = cross_val_mae_advanced(X, y, groups, n_splits, "xgboost")
    else:
        results["xgboost"] = None
    
    # CatBoost
    if CATBOOST_AVAILABLE:
        results["catboost"] = cross_val_mae_advanced(X, y, groups, n_splits, "catboost")
    else:
        results["catboost"] = None
    
    return results
