# Contributing

Thank you for your interest in contributing to DesignGenAI!

## Development Setup

```bash
git clone https://github.com/Lumi-node/design-genai.git
cd design-genai
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/
```

## Code Style

- Follow PEP 8 conventions
- Add type hints to all function signatures
- Write docstrings (Google style) for public functions
- Include unit tests for new functionality

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with tests
4. Run the test suite: `pytest tests/`
5. Commit with a descriptive message
6. Push and open a Pull Request

## Reporting Issues

Please include:
- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
