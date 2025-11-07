# Contributing to Tocantins Framework

Thank you for your interest in contributing to the Tocantins Framework! This document provides guidelines for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/EcoAcao-Brasil/tocantins-framework/issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Detailed description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (Python version, OS, package versions)
   - Minimal code example if possible

### Suggesting Enhancements

1. Check existing issues and discussions
2. Create a new issue with:
   - Clear description of the enhancement
   - Use cases and benefits
   - Possible implementation approach
   - Any references to scientific literature if applicable

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes following our coding standards
4. Add/update tests as appropriate
5. Update documentation
6. Ensure all tests pass
7. Commit with clear, descriptive messages
8. Push to your fork
9. Submit a pull request

## Development Setup

```bash
# Clone repository
git clone https://github.com/EcoAcao-Brasil/tocantins-framework
cd tocantins-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Coding Standards

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use `black` for formatting
- Use `isort` for import sorting

### Documentation

- All public functions/classes must have docstrings
- Use NumPy-style docstrings
- Include:
  - Brief description
  - Parameters with types
  - Returns with types
  - Examples where helpful
  - References for scientific methods

Example:
```python
def calculate_metric(data: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculate a specific metric from input data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array.
    threshold : float, default=0.5
        Threshold value for calculation.
    
    Returns
    -------
    metric : float
        Calculated metric value.
    
    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> metric = calculate_metric(data, threshold=0.6)
    >>> print(f"Metric: {metric:.2f}")
    """
    pass
```

### Testing

- Write tests for new features
- Maintain or improve code coverage
- Use pytest for testing
- Test edge cases and error conditions

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=tocantins_framework --cov-report=html
```

### Commit Messages

Use clear, descriptive commit messages:

```
type(scope): brief description

Detailed explanation if needed.

Fixes #issue_number
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style/formatting changes
- `refactor`: Code refactoring
- `test`: Test additions/modifications
- `chore`: Maintenance tasks

## Scientific Contributions

### Adding New Methods

When adding new detection methods or metrics:

1. Provide scientific justification and references
2. Document the methodology thoroughly
3. Include validation against existing methods
4. Provide example use cases
5. Consider computational efficiency

### Validation

- Compare with published results where possible
- Test on diverse datasets
- Document assumptions and limitations
- Include sensitivity analysis if appropriate

## Review Process

1. All contributions require review
2. Maintainers will provide feedback
3. Address review comments
4. Once approved, changes will be merged

## Questions?

- Open a [Discussion](https://github.com/EcoAcao-Brasil/tocantins-framework/discussions)
- Email: isaque@ecoacaobrasil.org

## License
By contributing, you agree that your contributions will be licensed under the MIT License.