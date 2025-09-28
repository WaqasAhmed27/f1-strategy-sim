from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.metrics import mean_absolute_error
from rich import print
from rich.progress import Progress, TaskID

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


class HyperparameterOptimizer:
    """Hyperparameter optimization for F1 prediction models."""
    
    def __init__(self, model_type: str = "xgboost", cv_splits: int = 5):
        self.model_type = model_type.lower()
        self.cv_splits = cv_splits
        
        if self.model_type == "xgboost" and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        elif self.model_type == "catboost" and not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not available. Install with: pip install catboost")
    
    def get_default_param_grid(self) -> Dict[str, list]:
        """Get default parameter grid for the specified model type."""
        if self.model_type == "xgboost":
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [1, 1.5, 2],
                'random_state': [42]
            }
        elif self.model_type == "catboost":
            return {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bylevel': [0.8, 0.9, 1.0],
                'l2_leaf_reg': [1, 3, 5],
                'random_seed': [42]
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def get_fast_param_grid(self) -> Dict[str, list]:
        """Get a smaller parameter grid for quick testing."""
        if self.model_type == "xgboost":
            return {
                'n_estimators': [200, 300],
                'max_depth': [6, 8],
                'learning_rate': [0.1, 0.15],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [1, 1.5],
                'random_state': [42]
            }
        elif self.model_type == "catboost":
            return {
                'iterations': [200, 300],
                'depth': [6, 8],
                'learning_rate': [0.1, 0.15],
                'subsample': [0.8, 0.9],
                'colsample_bylevel': [0.8, 0.9],
                'l2_leaf_reg': [1, 3],
                'random_seed': [42]
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _create_model(self, params: Dict[str, Any]):
        """Create a model instance with given parameters."""
        if self.model_type == "xgboost":
            return XGBRegressor(**params)
        elif self.model_type == "catboost":
            return CatBoostRegressor(**params, verbose=False)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def optimize(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series, 
                 param_grid: Optional[Dict[str, list]] = None, 
                 n_jobs: int = -1, verbose: bool = True) -> Tuple[Dict[str, Any], float]:
        """
        Optimize hyperparameters using grid search with GroupKFold CV.
        
        Returns:
            Tuple of (best_params, best_score)
        """
        if param_grid is None:
            param_grid = self.get_default_param_grid()
        
        # Create parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        n_combinations = len(param_combinations)
        
        if verbose:
            print(f"[bold green]Starting hyperparameter optimization[/bold green]")
            print(f"Model: {self.model_type}")
            print(f"CV splits: {self.cv_splits}")
            print(f"Parameter combinations: {n_combinations}")
            print(f"Total evaluations: {n_combinations * self.cv_splits}")
        
        best_score = float('inf')
        best_params = None
        results = []
        
        # GroupKFold for proper race-based validation
        gkf = GroupKFold(n_splits=self.cv_splits)
        
        with Progress() as progress:
            task = progress.add_task("[green]Optimizing...", total=n_combinations)
            
            for i, params in enumerate(param_combinations):
                cv_scores = []
                
                for train_idx, val_idx in gkf.split(X, y, groups):
                    # Create model with current parameters
                    model = self._create_model(params)
                    
                    # Train on fold
                    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
                    
                    model.fit(X_train, y_train)
                    
                    # Predict and evaluate
                    y_pred = model.predict(X_val)
                    mae = mean_absolute_error(y_val, y_pred)
                    cv_scores.append(mae)
                
                # Average CV score for this parameter combination
                mean_cv_score = np.mean(cv_scores)
                std_cv_score = np.std(cv_scores)
                
                results.append({
                    'params': params,
                    'mean_cv_score': mean_cv_score,
                    'std_cv_score': std_cv_score,
                    'cv_scores': cv_scores
                })
                
                # Update best if this is better
                if mean_cv_score < best_score:
                    best_score = mean_cv_score
                    best_params = params.copy()
                
                progress.update(task, advance=1)
                
                if verbose and (i + 1) % max(1, n_combinations // 10) == 0:
                    print(f"Progress: {i+1}/{n_combinations} - Best CV MAE: {best_score:.4f}")
        
        if verbose:
            print(f"[bold green]Optimization complete![/bold green]")
            print(f"Best CV MAE: {best_score:.4f}")
            print(f"Best parameters: {best_params}")
        
        return best_params, best_score, results
    
    def optimize_fast(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> Tuple[Dict[str, Any], float]:
        """Quick optimization using a smaller parameter grid."""
        return self.optimize(X, y, groups, self.get_fast_param_grid(), verbose=True)
    
    def save_results(self, results: list, filepath: str | Path):
        """Save optimization results to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        serializable_results = []
        for result in results:
            serializable_result = {
                'params': {k: v for k, v in result['params'].items()},
                'mean_cv_score': float(result['mean_cv_score']),
                'std_cv_score': float(result['std_cv_score']),
                'cv_scores': [float(score) for score in result['cv_scores']]
            }
            serializable_results.append(serializable_result)
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def load_results(self, filepath: str | Path) -> list:
        """Load optimization results from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)


def find_best_params(results: list, top_k: int = 5) -> list:
    """Find the top-k parameter combinations from optimization results."""
    sorted_results = sorted(results, key=lambda x: x['mean_cv_score'])
    return sorted_results[:top_k]


def analyze_parameter_importance(results: list) -> Dict[str, Dict[str, float]]:
    """Analyze which parameters have the most impact on performance."""
    param_importance = {}
    
    # Get all unique parameter names
    all_params = set()
    for result in results:
        all_params.update(result['params'].keys())
    
    for param_name in all_params:
        param_values = []
        scores = []
        
        for result in results:
            if param_name in result['params']:
                param_values.append(result['params'][param_name])
                scores.append(result['mean_cv_score'])
        
        if param_values and scores:
            # Calculate correlation between parameter value and score
            param_df = pd.DataFrame({'value': param_values, 'score': scores})
            
            # Group by parameter value and calculate mean score
            grouped = param_df.groupby('value')['score'].agg(['mean', 'std', 'count'])
            
            param_importance[param_name] = {
                'best_value': grouped['mean'].idxmin(),
                'worst_value': grouped['mean'].idxmax(),
                'score_range': grouped['mean'].max() - grouped['mean'].min(),
                'impact': grouped['mean'].std()
            }
    
    return param_importance
