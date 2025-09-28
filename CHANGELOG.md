# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-28

### Added
- **Core Features**
  - Multi-session data analysis (Practice, Qualifying, Race)
  - Advanced feature engineering with 29+ features
  - XGBoost and CatBoost model implementations
  - Hyperparameter optimization with grid search
  - Comprehensive feature selection analysis
  
- **CLI Commands**
  - `f1sim ingest` - Download race data from FastF1
  - `f1sim predict` - Predict race results
  - `f1sim train-multi` - Train on multiple races
  - `f1sim analyze-features` - Feature importance analysis
  - `f1sim tune-hyperparameters` - Hyperparameter optimization
  - `f1sim compare-models-cmd` - Model comparison
  - `f1sim predict-advanced` - Advanced model predictions
  - `f1sim train-with-selected-features` - Optimal feature training
  - `f1sim predict-with-selected-features` - Selected feature predictions

- **Feature Engineering**
  - Race pace analysis (mean, best, std lap times)
  - Session adaptation (practice to qualifying to race)
  - Penalty features (grid drops, time penalties, DNF risk)
  - Pit strategy analysis (stop count, efficiency, risk)
  - Driver/team form trends from historical data
  - Tire degradation analysis
  - Weather impact assessment

- **Models**
  - XGBoost with optimized default parameters
  - CatBoost with categorical feature handling
  - Gradient Boosting baseline
  - Feature alignment for consistent predictions
  - Model persistence with joblib

- **Analysis Tools**
  - Feature importance ranking
  - Univariate feature selection
  - Recursive feature elimination
  - Correlation analysis
  - Cross-validation with GroupKFold
  - Performance comparison across models

- **Documentation**
  - Comprehensive README with examples
  - API documentation
  - Usage examples
  - Contributing guidelines
  - MIT License

- **Testing & CI/CD**
  - Test suite with pytest
  - GitHub Actions CI/CD pipeline
  - Code quality tools (black, ruff, mypy)
  - Pre-commit hooks
  - Coverage reporting

### Performance
- **Best Model**: XGBoost with selected features
- **CV MAE**: 3.008 (42.4% improvement over baseline)
- **Feature Reduction**: 29 → 10 features (65% reduction)
- **Key Insight**: Race pace delta dominates with 80% importance

### Data Support
- **Seasons**: 2021-2024
- **Sessions**: FP1, FP2, FP3, Q, R
- **Races**: 13 races with multi-session data
- **Features**: 29 engineered features
- **Storage**: Efficient Parquet format

## [0.0.1] - 2025-01-25

### Added
- Initial project scaffold
- Basic CLI structure
- FastF1 data ingestion
- Simple feature engineering
- Baseline Gradient Boosting model
- Basic prediction functionality

---

## Version History

- **1.0.0**: Production-ready release with advanced ML capabilities
- **0.0.1**: Initial MVP with basic functionality
