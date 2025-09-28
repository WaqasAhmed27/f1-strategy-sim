# Contributing to F1 Strategy Simulator

Thank you for your interest in contributing to the F1 Strategy Simulator! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/f1-strategy-sim.git
   cd f1-strategy-sim
   ```

2. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/f1sim --cov-report=html

# Run specific test file
pytest tests/test_models.py
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## 📝 Contribution Guidelines

### Code Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all public functions and classes
- Keep line length under 100 characters

### Commit Messages

Use clear, descriptive commit messages:

```
feat: Add new feature
fix: Fix bug
docs: Update documentation
test: Add tests
refactor: Refactor code
```

### Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write tests for new functionality
   - Update documentation if needed
   - Ensure all tests pass

3. **Submit a pull request**
   - Provide a clear description of changes
   - Reference any related issues
   - Ensure CI passes

## 🧪 Testing

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies when appropriate

Example test structure:
```python
def test_feature_functionality():
    """Test that feature works correctly."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = feature_function(input_data)
    
    # Assert
    assert result == expected_output
```

### Test Data

- Use synthetic data for unit tests
- Don't commit large data files
- Use fixtures for common test data

## 📚 Documentation

### Docstrings

Use Google-style docstrings:

```python
def predict_race_results(data: pd.DataFrame) -> pd.DataFrame:
    """Predict race results from input data.
    
    Args:
        data: DataFrame containing race data
        
    Returns:
        DataFrame with predicted results
        
    Raises:
        ValueError: If data is invalid
    """
```

### README Updates

- Update README.md for new features
- Include usage examples
- Update installation instructions if needed

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment information**
   - Python version
   - Operating system
   - Package versions

2. **Steps to reproduce**
   - Clear, minimal steps
   - Expected vs actual behavior

3. **Error messages**
   - Full traceback if available
   - Screenshots if helpful

## 💡 Feature Requests

For feature requests, please:

1. **Check existing issues** first
2. **Describe the feature** clearly
3. **Explain the use case** and benefits
4. **Provide examples** if possible

## 🏷️ Release Process

Releases are managed through GitHub releases:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a release tag
4. GitHub Actions will build and publish

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: For private or sensitive matters

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to F1 Strategy Simulator! 🏎️
