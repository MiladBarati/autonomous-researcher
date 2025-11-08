# Contributing to Autonomous Research Assistant

Thank you for your interest in contributing to the Autonomous Research Assistant project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Project Structure](#project-structure)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful, inclusive, and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/autonomous-research-assistant.git
   cd autonomous-research-assistant
   ```
3. **Set up a remote** for the upstream repository:
   ```bash
   git remote add upstream https://github.com/original-owner/autonomous-research-assistant.git
   ```

## Development Setup

1. **Install the project in development mode** with all dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. **Set up pre-commit hooks** (recommended):
   ```bash
   pre-commit install
   ```
   This will automatically run code quality checks before each commit.

3. **Set up environment variables**:
   Create a `.env` file with your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key
   TAVILY_API_KEY=your_tavily_api_key
   LANGCHAIN_API_KEY=your_langchain_api_key  # Optional
   ```

4. **Install Playwright browsers** (if needed for web scraping):
   ```bash
   playwright install
   ```

## Making Changes

1. **Create a new branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes** following the code style guidelines below

3. **Test your changes** to ensure everything works correctly

4. **Update documentation** if your changes affect user-facing functionality

## Code Style

This project uses several tools to maintain consistent code style:

### Formatting

- **black**: Code formatter (line length: 100)
- **isort**: Import sorter (compatible with black)
- **ruff**: Fast linter and formatter

### Type Checking

- **mypy**: Static type checker (Python 3.11+)

### Running Code Quality Checks

**With pre-commit hooks (recommended):**

If you've installed pre-commit hooks, they will run automatically on each commit. You can also run them manually:

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run hooks on staged files only
pre-commit run
```

**Manual checks:**

If you prefer to run checks manually:

```bash
# Format code
black .
isort .
ruff format .

# Lint code
ruff check .

# Type check
mypy .
```

Or run all checks at once:
```bash
black . && isort . && ruff check . && ruff format . && mypy .
```

### Code Style Guidelines

- **Line length**: Maximum 100 characters
- **Python version**: Target Python 3.11+
- **Type hints**: Use type hints for all function parameters and return values
- **Docstrings**: Use docstrings for all public functions and classes
- **Imports**: Sort imports using isort (automatically handled)

## Testing

### Running Tests

Run the test suite:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=agent --cov-report=html
```

### Writing Tests

- Place test files in the `tests/` directory
- Test files should be named `test_*.py`
- Use pytest fixtures from `tests/conftest.py` when available
- Aim for good test coverage, especially for new features

### Test Structure

- Unit tests for individual functions and classes
- Integration tests for component interactions
- Mock external API calls to avoid rate limits and costs

## Submitting Changes

1. **Commit your changes** with clear, descriptive commit messages:
   ```bash
   git commit -m "Add feature: description of what you added"
   ```

2. **Keep your branch up to date** with the main branch:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

3. **Push your changes** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request** on GitHub:
   - Provide a clear title and description
   - Reference any related issues
   - Describe what changes you made and why

### Pull Request Guidelines

- **Title**: Clear and descriptive
- **Description**: Explain what and why, not just how
- **Scope**: Keep PRs focused on a single feature or fix
- **Tests**: Include tests for new features
- **Documentation**: Update README or other docs if needed
- **CI**: Ensure all CI checks pass

## Project Structure

```
autonomous-research-assistant/
├── agent/              # Core agent modules
│   ├── graph.py        # LangGraph workflow
│   ├── state.py        # State definitions
│   ├── tools.py        # Research tools
│   └── rag.py          # RAG pipeline
├── tests/              # Test suite
├── config.py           # Configuration
├── app.py              # Streamlit web interface
├── main.py             # CLI entry point
└── pyproject.toml      # Project configuration
```

## Areas for Contribution

We welcome contributions in the following areas:

- **Bug fixes**: Report and fix bugs
- **New features**: See the "Future Enhancements" section in README.md
- **Documentation**: Improve docs, add examples, fix typos
- **Tests**: Increase test coverage
- **Performance**: Optimize code and reduce API calls
- **UI/UX**: Improve the Streamlit interface
- **Tool integrations**: Add new research tools or data sources

## Questions?

If you have questions or need help, please:
- Open an issue on GitHub
- Check existing issues and discussions
- Review the README.md for project documentation

Thank you for contributing! 🎉
