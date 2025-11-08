"""Sphinx configuration for Autonomous Research Assistant API documentation."""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Project information
project = "Autonomous Research Assistant"
copyright = "2025, Autonomous Research Assistant Contributors"
author = "Autonomous Research Assistant Contributors"
release = "0.1.0"
version = "0.1.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # For Google/NumPy style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
    "exclude-members": "__weakref__",
}

# Autosummary settings
autosummary_generate = True

# Napoleon settings (for docstring parsing)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# HTML output options
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "logo_only": False,
    "navigation_depth": 4,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#2980B9",
    "titles_only": False,
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Intersphinx mapping
# Note: If intersphinx causes issues, you can disable it by commenting out the extension
# or setting intersphinx_disabled_domains = []
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    # Temporarily disabled due to URL changes and Sphinx 8.2.3 compatibility issues
    # "langchain": ("https://api.python.langchain.com/en/stable/", None),
    # "langgraph": ("https://langchain-ai.github.io/langgraph/", None),
}

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source file extensions
source_suffix = ".rst"

# Master document
master_doc = "index"

# Language
language = "en"

# Output file base name
htmlhelp_basename = "AutonomousResearchAssistantDoc"
