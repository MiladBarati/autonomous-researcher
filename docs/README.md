# Documentation

This directory contains the Sphinx documentation for the Autonomous Research Assistant.

## Building the Documentation

### Prerequisites

Install the development dependencies:

```bash
pip install -e ".[dev]"
```

This installs Sphinx and the Read the Docs theme.

### Build Commands

**Using Make (Linux/macOS):**
```bash
cd docs
make html
```

**Using Sphinx directly (Windows/Cross-platform):**
```bash
cd docs
sphinx-build -b html . _build/html
```

### Viewing the Documentation

After building, open `_build/html/index.html` in your web browser.

## Documentation Structure

- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation index
- `installation.rst` - Installation guide
- `quickstart.rst` - Quick start guide
- `api/` - API reference documentation
  - `modules.rst` - All modules overview
  - `agent.rst` - Agent and graph documentation
  - `config.rst` - Configuration documentation
- `examples.rst` - Usage examples
- `_static/` - Static files (CSS, images, etc.)
- `Makefile` - Makefile for building docs (Linux/macOS)

## Auto-documentation

The documentation uses Sphinx autodoc to automatically extract docstrings from the Python code. When you update docstrings in the code, rebuild the documentation to see the changes.

## Continuous Updates

To keep the documentation up to date:

1. Write clear docstrings in your Python code
2. Rebuild the documentation after making changes
3. Review the generated HTML to ensure everything looks correct
