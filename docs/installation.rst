Installation
============

Prerequisites
-------------

- Python 3.11 or higher
- pip (Python package manager)

Installing the Package
----------------------

Install the package and its dependencies:

.. code-block:: bash

   pip install -e .

Or install manually with all dependencies:

.. code-block:: bash

   pip install langgraph langchain langchain-groq langchain-community \
       langchain-text-splitters chromadb sentence-transformers tavily-python \
       arxiv beautifulsoup4 playwright pypdf2 pdfplumber streamlit \
       python-dotenv requests tiktoken

Development Dependencies
------------------------

For development, including documentation generation:

.. code-block:: bash

   pip install -e ".[dev]"

This installs:
- Testing tools (pytest, pytest-cov)
- Code quality tools (black, isort, ruff, mypy)
- Documentation tools (sphinx, sphinx-rtd-theme)
- Pre-commit hooks

Environment Variables
----------------------

Create a ``.env`` file in the project root with your API keys:

.. code-block:: text

   GROQ_API_KEY=your_groq_api_key
   TAVILY_API_KEY=your_tavily_api_key

   # Optional: LangSmith tracing
   LANGCHAIN_API_KEY=your_langchain_api_key
   LANGSMITH_TRACING=true
   LANGSMITH_ENDPOINT=https://api.smith.langchain.com
   LANGSMITH_PROJECT=your_project_name

Get API Keys:
- Groq: https://console.groq.com/
- Tavily: https://tavily.com/
- LangSmith (optional): https://smith.langchain.com/
