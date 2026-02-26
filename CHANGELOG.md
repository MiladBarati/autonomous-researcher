# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project setup with LangGraph workflow
- Web search integration via Tavily API
- Web scraping capabilities with BeautifulSoup and Playwright
- ArXiv academic paper search and retrieval
- PDF processing with PyPDF2 and pdfplumber
- RAG pipeline with ChromaDB vector storage
- Sentence transformers for document embeddings
- Streamlit web interface
- CLI interface
- Real-time progress tracking
- Comprehensive test suite
- Code quality tools (black, isort, ruff, mypy)
- CI/CD pipeline configuration

### Changed
- **Refactored graph node methods**: Extracted cross-cutting concerns (logging, performance monitoring, validation) into a reusable `_node_decorator` function, significantly reducing code duplication and improving maintainability. All 7 node methods now use the decorator pattern for consistent behavior.

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [0.1.0] - 2024-XX-XX

### Added
- Initial release of Autonomous Research Assistant
- Core research agent with LangGraph workflow
- Multi-source intelligence gathering (web search, scraping, ArXiv)
- RAG pipeline for context-aware synthesis
- Dual interfaces: Streamlit web UI and CLI
- Support for Groq LLM inference
- Vector storage with ChromaDB
- Document chunking and embedding
- Research report generation
- Configuration management
- Environment variable support for API keys
- Comprehensive logging system

[Unreleased]: https://github.com/your-username/autonomous-research-assistant/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-username/autonomous-research-assistant/releases/tag/v0.1.0
