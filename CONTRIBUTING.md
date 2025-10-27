# Contributing to H. pylori RL Simulator

Thank you for your interest in contributing! This is a research prototype project.

## Important Disclaimer

⚠️ **This is a research prototype. NOT a medical device.**

All contributions must maintain this fundamental principle:
- Never claim clinical validity
- Always include research disclaimers
- Never process real patient data
- Follow ethical research practices

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System information (OS, Python version, GPU, etc.)
   - Relevant logs or error messages

### Code Contributions

#### Setting Up Development Environment

```bash
# Clone repository
git clone <repo-url>
cd rl_model_cdss

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r backend/requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Run tests
pytest backend/tests/
```

#### Making Changes

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**:
   - Follow existing code style
   - Add docstrings to new functions/classes
   - Include type hints
   - Update tests
   - Update documentation

4. **Test your changes**:
   ```bash
   # Run tests
   pytest backend/tests/ -v
   
   # Check code style
   black backend/ --check
   flake8 backend/
   
   # Type checking
   mypy backend/ --ignore-missing-imports
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add feature description"
   ```
   
   Use conventional commits:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `test:` for tests
   - `refactor:` for refactoring

6. **Push and create pull request**:
   ```bash
   git push origin feature/your-feature-name
   ```

#### Pull Request Guidelines

- Provide clear description of changes
- Reference related issues
- Include test coverage for new code
- Update documentation
- Ensure all tests pass
- Follow code style guidelines

### Code Style

#### Python

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use docstrings (Google style)
- Format with `black`

Example:
```python
def train_model(
    data_path: str,
    epochs: int = 10,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Train the model on provided data.

    Args:
        data_path: Path to training data
        epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        Dictionary with training metrics

    Raises:
        FileNotFoundError: If data_path doesn't exist
    """
    # Implementation
    pass
```

#### JavaScript

- Use ES6+ features
- Clear variable names
- Comments for complex logic
- Consistent formatting

### Testing

#### Writing Tests

Add tests for new features:

```python
# backend/tests/test_feature.py
import pytest

def test_new_feature():
    """Test description."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = new_feature(input_data)
    
    # Assert
    assert result == expected_output
```

#### Running Tests

```bash
# All tests
pytest backend/tests/

# Specific test file
pytest backend/tests/test_env.py

# With coverage
pytest backend/tests/ --cov=backend --cov-report=html
```

### Documentation

#### Docstrings

All functions/classes should have docstrings:

```python
def complex_function(
    param1: str,
    param2: int = 10,
    param3: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """Brief description of function.

    Longer description if needed, explaining the purpose,
    algorithm, or important details.

    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)
        param3: Description of param3 (optional)

    Returns:
        Tuple of (success status, message)

    Raises:
        ValueError: If param1 is empty
        TypeError: If param2 is negative

    Examples:
        >>> result, msg = complex_function("test", param2=5)
        >>> print(result)
        True
    """
```

#### README Updates

If your changes affect usage:
- Update README.md
- Update QUICKSTART.md if needed
- Add examples

### Research Ethics

When contributing:

1. **Never use real patient data**
   - Only synthetic or publicly available test data
   - No PHI (Protected Health Information)

2. **Maintain disclaimers**
   - Keep research warnings visible
   - Don't remove disclaimer banners
   - Don't make clinical claims

3. **Proper attribution**
   - Credit original authors
   - Respect licenses
   - Document data sources

4. **Reproducibility**
   - Use random seeds for experiments
   - Document hyperparameters
   - Include configuration files

### Areas for Contribution

We welcome contributions in:

#### Simulation
- More realistic GI tract models
- Better lesion synthesis
- Improved physics/collision detection
- Domain randomization techniques

#### Machine Learning
- Better CNN architectures
- Alternative RL algorithms
- Transfer learning approaches
- Multi-task learning

#### Visualization
- Enhanced 3D viewer
- Better metrics visualization
- Real-time plotting
- Export to medical formats

#### Infrastructure
- Performance optimization
- Better error handling
- Monitoring and logging
- Cloud deployment guides

#### Documentation
- Tutorials
- Video guides
- Architecture diagrams
- API examples

#### Testing
- More test coverage
- Integration tests
- Performance benchmarks
- Stress tests

### Review Process

1. **Automated checks**:
   - Tests must pass
   - Linting must pass
   - No security issues

2. **Code review**:
   - Maintainer reviews code
   - May request changes
   - Discussion in PR comments

3. **Merge**:
   - After approval
   - Squash and merge
   - Update changelog

### Getting Help

Questions? Create an issue with the "question" label.

### License

By contributing, you agree that your contributions will be licensed under the MIT License.

### Code of Conduct

- Be respectful and professional
- Welcome newcomers
- Give constructive feedback
- Focus on the research goals
- Remember: this is for research, not clinical use

### Acknowledgments

Contributors will be acknowledged in:
- CONTRIBUTORS.md (if we create one)
- Release notes
- Research publications (if applicable)

---

Thank you for contributing to research in medical AI simulation!

**Remember**: This is a research tool. Our goal is to advance understanding, not to create a clinical product.

