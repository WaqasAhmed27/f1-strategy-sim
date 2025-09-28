from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_absolute_error
from rich import print
from rich.table import Table
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


class FeatureSelector:
    """Comprehensive feature selection for F1 prediction models."""
    
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type.lower()
        self.feature_importance_scores = {}
        self.selection_results = {}
        
        if self.model_type == "xgboost" and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        elif self.model_type == "catboost" and not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not available. Install with: pip install catboost")
    
    def _create_model(self, **kwargs):
        """Create a model instance for feature selection."""
        if self.model_type == "xgboost":
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbosity': 0
            }
            default_params.update(kwargs)
            return XGBRegressor(**default_params)
        elif self.model_type == "catboost":
            default_params = {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1,
                'random_seed': 42,
                'verbose': False
            }
            default_params.update(kwargs)
            return CatBoostRegressor(**default_params)
        else:
            # Fallback to Random Forest
            return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def analyze_feature_importance(self, X: pd.DataFrame, y: pd.Series, 
                                 groups: pd.Series | None = None) -> Dict[str, float]:
        """Analyze feature importance using tree-based models."""
        print(f"[bold green]Analyzing feature importance[/bold green] with {self.model_type}")
        
        # Train model and get feature importance
        model = self._create_model()
        model.fit(X, y)
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
            feature_names = X.columns
        elif hasattr(model, 'feature_names_in_'):
            importance_scores = model.feature_importances_
            feature_names = model.feature_names_in_
        else:
            raise ValueError("Model does not support feature importance")
        
        # Create importance dictionary
        importance_dict = dict(zip(feature_names, importance_scores))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        self.feature_importance_scores = sorted_importance
        
        return sorted_importance
    
    def univariate_selection(self, X: pd.DataFrame, y: pd.Series, 
                           k: int = 15, method: str = "f_regression") -> List[str]:
        """Select top k features using univariate statistical tests."""
        print(f"[bold green]Univariate selection[/bold green]: top {k} features using {method}")
        
        if method == "f_regression":
            selector = SelectKBest(score_func=f_regression, k=k)
        elif method == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Get scores for selected features
        scores = selector.scores_
        feature_scores = dict(zip(X.columns, scores))
        
        self.selection_results[f"univariate_{method}"] = {
            'selected_features': selected_features,
            'scores': feature_scores
        }
        
        return selected_features
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series,
                                    n_features: int = 15, 
                                    groups: pd.Series | None = None) -> List[str]:
        """Select features using Recursive Feature Elimination."""
        print(f"[bold green]Recursive Feature Elimination[/bold green]: selecting {n_features} features")
        
        # Create base estimator
        estimator = self._create_model()
        
        if groups is not None:
            # Use RFECV for cross-validation
            cv = GroupKFold(n_splits=min(5, len(groups.unique())))
            selector = RFECV(estimator, cv=cv, scoring='neg_mean_absolute_error')
            selector.fit(X, y, groups=groups)
            selected_features = X.columns[selector.support_].tolist()
        else:
            # Use RFE
            selector = RFE(estimator, n_features_to_select=n_features)
            selector.fit(X, y)
            selected_features = X.columns[selector.support_].tolist()
        
        self.selection_results["rfe"] = {
            'selected_features': selected_features,
            'ranking': dict(zip(X.columns, selector.ranking_))
        }
        
        return selected_features
    
    def correlation_analysis(self, X: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """Remove highly correlated features."""
        print(f"[bold green]Correlation analysis[/bold green]: removing features with correlation > {threshold}")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for feat1, feat2 in high_corr_pairs:
            # Keep the feature with higher variance (more information)
            var1 = X[feat1].var()
            var2 = X[feat2].var()
            if var1 > var2:
                features_to_remove.add(feat2)
            else:
                features_to_remove.add(feat1)
        
        selected_features = [f for f in X.columns if f not in features_to_remove]
        
        self.selection_results["correlation"] = {
            'selected_features': selected_features,
            'removed_features': list(features_to_remove),
            'high_corr_pairs': high_corr_pairs
        }
        
        return selected_features
    
    def evaluate_feature_sets(self, X: pd.DataFrame, y: pd.Series, 
                            groups: pd.Series | None = None,
                            feature_sets: Dict[str, List[str]] | None = None) -> Dict[str, float]:
        """Evaluate different feature sets using cross-validation."""
        print(f"[bold green]Evaluating feature sets[/bold green]")
        
        if feature_sets is None:
            # Create default feature sets
            feature_sets = {
                'all_features': X.columns.tolist(),
                'top_10_importance': list(self.feature_importance_scores.keys())[:10],
                'top_15_importance': list(self.feature_importance_scores.keys())[:15],
                'top_20_importance': list(self.feature_importance_scores.keys())[:20],
            }
        
        results = {}
        
        # Cross-validation setup
        if groups is not None:
            cv = GroupKFold(n_splits=min(5, len(groups.unique())))
        else:
            cv = 5
        
        for set_name, features in feature_sets.items():
            if not features:
                continue
                
            # Select features
            X_subset = X[features]
            
            # Create model
            model = self._create_model()
            
            # Cross-validation
            scores = cross_val_score(model, X_subset, y, cv=cv, 
                                   scoring='neg_mean_absolute_error', groups=groups)
            mae_scores = -scores  # Convert to positive MAE
            
            results[set_name] = {
                'cv_mae_mean': float(np.mean(mae_scores)),
                'cv_mae_std': float(np.std(mae_scores)),
                'n_features': len(features),
                'features': features
            }
        
        return results
    
    def comprehensive_analysis(self, X: pd.DataFrame, y: pd.Series, 
                            groups: pd.Series | None = None) -> Dict:
        """Run comprehensive feature selection analysis."""
        print(f"[bold green]Comprehensive Feature Selection Analysis[/bold green]")
        
        results = {}
        
        # 1. Feature importance analysis
        importance_scores = self.analyze_feature_importance(X, y, groups)
        results['importance_analysis'] = importance_scores
        
        # 2. Univariate selection
        univariate_f = self.univariate_selection(X, y, k=15, method="f_regression")
        univariate_mi = self.univariate_selection(X, y, k=15, method="mutual_info")
        results['univariate_selection'] = {
            'f_regression': univariate_f,
            'mutual_info': univariate_mi
        }
        
        # 3. Recursive feature elimination
        rfe_features = self.recursive_feature_elimination(X, y, n_features=15, groups=groups)
        results['rfe_selection'] = rfe_features
        
        # 4. Correlation analysis
        corr_features = self.correlation_analysis(X, threshold=0.95)
        results['correlation_analysis'] = corr_features
        
        # 5. Evaluate different feature sets
        feature_sets = {
            'all_features': X.columns.tolist(),
            'top_10_importance': list(importance_scores.keys())[:10],
            'top_15_importance': list(importance_scores.keys())[:15],
            'top_20_importance': list(importance_scores.keys())[:20],
            'univariate_f': univariate_f,
            'univariate_mi': univariate_mi,
            'rfe': rfe_features,
            'correlation_filtered': corr_features,
        }
        
        evaluation_results = self.evaluate_feature_sets(X, y, groups, feature_sets)
        results['evaluation_results'] = evaluation_results
        
        return results
    
    def get_feature_importance_table(self, top_n: int = 20) -> Table:
        """Create a rich table showing feature importance."""
        if not self.feature_importance_scores:
            return Table(title="No feature importance data available")
        
        table = Table(title=f"Top {top_n} Feature Importance ({self.model_type})")
        table.add_column("Rank", style="cyan", no_wrap=True)
        table.add_column("Feature", style="magenta")
        table.add_column("Importance", style="green")
        table.add_column("Category", style="yellow")
        
        # Feature categories
        feature_categories = {
            'GridPosition': 'Starting Position',
            'mean_lap_time': 'Race Pace',
            'best_lap_time': 'Race Pace',
            'std_lap_time': 'Race Pace',
            'n_laps': 'Race Completion',
            'n_pit_laps': 'Strategy',
            'AirTemp': 'Weather',
            'TrackTemp': 'Weather',
            'Humidity': 'Weather',
            'WindSpeed': 'Weather',
            'grid_drop': 'Penalties',
            'time_penalty_seconds': 'Penalties',
            'dnf_risk': 'Penalties',
            'penalty_severity': 'Penalties',
            'avg_pit_time': 'Pit Strategy',
            'pit_stops': 'Pit Strategy',
            'pit_efficiency': 'Pit Strategy',
            'pit_strategy_risk': 'Pit Strategy',
            'qual_vs_race_pace': 'Session Pace',
            'practice_vs_race_pace': 'Session Pace',
            'pace_consistency': 'Session Pace',
            'session_adaptation': 'Session Pace',
            'team_form_trend': 'Form',
            'driver_form_trend': 'Form',
            'team_consistency': 'Form',
            'driver_consistency': 'Form',
            'race_pace_delta': 'Race Pace',
            'driver_form_meanlap_prev': 'Form'
        }
        
        for i, (feature, importance) in enumerate(list(self.feature_importance_scores.items())[:top_n], 1):
            category = feature_categories.get(feature, 'Other')
            table.add_row(
                str(i),
                feature,
                f"{importance:.4f}",
                category
            )
        
        return table
    
    def save_results(self, results: Dict, filepath: str | Path):
        """Save feature selection results to JSON file."""
        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python types."""
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Convert all numpy types to Python types
        serializable_results = convert_numpy_types(results)
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def load_results(self, filepath: str | Path) -> Dict:
        """Load feature selection results from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)


def find_optimal_feature_set(evaluation_results: Dict[str, Dict]) -> Tuple[str, Dict]:
    """Find the optimal feature set based on evaluation results."""
    best_set = None
    best_score = float('inf')
    
    for set_name, results in evaluation_results.items():
        if 'cv_mae_mean' in results:
            score = results['cv_mae_mean']
            if score < best_score:
                best_score = score
                best_set = set_name
    
    return best_set, evaluation_results.get(best_set, {})
