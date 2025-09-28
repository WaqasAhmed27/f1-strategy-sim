# 🏎️ F1 Strategy Simulator

A comprehensive Formula 1 race prediction system using machine learning to predict finishing positions, lap times, and strategic scenarios.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Features

- **Multi-Session Data Analysis**: Practice, Qualifying, and Race sessions
- **Advanced Feature Engineering**: 29+ features including pace analysis, penalties, pit strategy, and driver form
- **Multiple ML Models**: XGBoost, CatBoost, and Gradient Boosting
- **Hyperparameter Optimization**: Automated grid search for optimal performance
- **Feature Selection**: Intelligent feature importance analysis and optimization
- **Real-time Predictions**: Predict finishing order with estimated time gaps
- **Historical Data**: Support for multiple seasons (2021-2024)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/WaqasAhmed27/f1-strategy-sim.git
cd f1-strategy-sim

# Install dependencies
pip install -e .

# Download sample data
f1sim ingest --season 2024 --race-round 1 --session R
```

### Basic Usage

```bash
# Predict race results
f1sim predict --season 2024 --race-round 1

# Train a model on multiple races
f1sim train-multi --races "2024-1,2024-2,2024-3"

# Analyze feature importance
f1sim analyze-features --races "2024-1,2024-2" --model-type xgboost

# Optimize hyperparameters
f1sim tune-hyperparameters --races "2024-1,2024-2" --model-type xgboost
```

## 📊 Performance

My best model achieves:
- **CV MAE**: 3.008 (Mean Absolute Error in positions)
- **Feature Count**: 10 optimized features (65% reduction from 29)
- **Key Features**: Race pace delta (80% importance), grid position, session adaptation

## 🏗️ Architecture

```
f1-strategy-sim/
├── 📁 configs/          # Configuration files
├── 📁 data/            # Race data storage
├── 📁 models/          # Trained models and results
├── 📁 src/f1sim/       # Core package
│   ├── 📁 features/    # Feature engineering
│   ├── 📁 ingest/      # Data ingestion
│   ├── 📁 models/      # ML model implementations
│   └── 📁 storage/     # File management
├── 📁 examples/        # Usage examples
├── 📁 tests/           # Test suite
└── 📁 docs/            # Documentation
```

## 🔧 Advanced Features

### Bulk Data Ingestion
```bash
# Download entire season (all races, all sessions)
f1sim ingest-season --season 2024 --sessions "FP1,FP2,FP3,Q,R"

# Download specific race range
f1sim ingest-season --season 2024 --start-round 1 --end-round 10 --sessions "R"

# Download single race with all sessions
f1sim ingest-season --season 2024 --start-round 1 --end-round 1 --sessions "FP1,FP2,FP3,Q,R"

# Check what data you have
f1sim list-seasons
```

### Single Session Ingestion
```bash
# Download individual sessions
f1sim ingest --season 2024 --race-round 1 --session FP1
f1sim ingest --season 2024 --race-round 1 --session FP2
f1sim ingest --season 2024 --race-round 1 --session FP3
f1sim ingest --season 2024 --race-round 1 --session Q
f1sim ingest --season 2024 --race-round 1 --session R
```

### Feature Engineering
- **Race Pace Analysis**: Mean, best, and standard deviation of lap times
- **Session Adaptation**: Practice to qualifying to race pace evolution
- **Penalty Features**: Grid drops, time penalties, DNF risk
- **Pit Strategy**: Stop count, efficiency, and risk assessment
- **Driver/Team Form**: Historical performance trends and consistency

### Model Comparison
```bash
# Compare different models
f1sim compare-models-cmd --races "2024-1,2024-2" --cv-splits 3

# Train with selected features
f1sim train-with-selected-features --races "2024-1,2024-2" --feature-set top_10_importance
```

## 📈 Model Performance

| Model | CV MAE | Features | Notes |
|-------|--------|----------|-------|
| Gradient Boosting | 3.578 | 29 | Baseline |
| XGBoost | 2.416 | 29 | Advanced |
| XGBoost (Optimized) | 2.312 | 29 | Hyperparameter tuned |
| XGBoost (Selected) | **3.008** | **10** | **Best performing** |

## 🎮 CLI Commands

### Data Management
- `f1sim ingest` - Download single race session from FastF1
- `f1sim ingest-season` - Download entire season with multiple sessions
- `f1sim list-seasons` - List available data in your local directory
- `f1sim train` - Train a single-race model
- `f1sim train-multi` - Train on multiple races

### Prediction
- `f1sim predict` - Predict race results
- `f1sim predict-advanced` - Use advanced models
- `f1sim predict-optimized` - Use hyperparameter-tuned models

### Analysis
- `f1sim analyze-features` - Feature importance analysis
- `f1sim compare-models-cmd` - Model comparison
- `f1sim tune-hyperparameters` - Hyperparameter optimization

## 🔬 Technical Details

### Feature Categories
1. **Race Pace** (40.1% importance): Mean lap time, pace delta, consistency
2. **Starting Position** (5.5%): Grid position and penalties
3. **Race Completion** (5.4%): Lap count and pit stops
4. **Session Analysis** (4.1%): Multi-session pace adaptation
5. **Form Trends** (2.0%): Historical driver/team performance

### Model Architecture
- **XGBoost**: Gradient boosting with optimized parameters
- **Feature Selection**: Recursive feature elimination + importance analysis
- **Cross-Validation**: Group K-Fold by race to prevent data leakage
- **Hyperparameter Tuning**: Grid search with 5-fold CV

## 📚 Examples

### Basic Prediction
```python
from f1sim.features.builder import build_features_from_dir
from f1sim.models.advanced import XGBoostRegressor
from pathlib import Path

# Load data
X, y, meta = build_features_from_dir(Path("data/2024_1_R"))

# Train model
model = XGBoostRegressor()
model.fit(X, y)

# Predict
predictions = model.predict(X)
```

### Feature Analysis
```python
from f1sim.models.feature_selection import FeatureSelector

# Analyze features
selector = FeatureSelector("xgboost")
results = selector.comprehensive_analysis(X, y, groups)

# Get top features
top_features = list(results['importance_analysis'].keys())[:10]
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/

# Type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastF1](https://github.com/theOehrly/Fast-F1) for Formula 1 data access
- [XGBoost](https://xgboost.readthedocs.io/) for gradient boosting
- [CatBoost](https://catboost.ai/) for categorical boosting
- [scikit-learn](https://scikit-learn.org/) for machine learning utilities

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/WaqasAhmed27/f1-strategy-sim/issues)
- **Discussions**: [GitHub Discussions](https://github.com/WaqasAhmed27/f1-strategy-sim/discussions)
- **Email**: vvaqasahmed27@gmail.com

---

**Made with ❤️ for Formula 1 fans and data scientists**