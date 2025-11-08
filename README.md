# Autonomous Research Assistant

[![Coverage](https://img.shields.io/badge/coverage-80%25-brightgreen)](https://github.com/miladbarati/autonomous-researcher)
[![CodeQL](https://github.com/miladbarati/autonomous-research-assistant/actions/workflows/codeql.yml/badge.svg)](https://github.com/miladbarati/autonomous-researcher/actions/workflows/codeql.yml)

An event-driven AI research agent that autonomously retrieves, summarizes, and synthesizes web content using LangGraph, LangChain, and Retrieval-Augmented Generation (RAG).

## Features

🤖 **Autonomous Research**: Self-directed topic research with minimal human intervention

🔍 **Multi-Source Intelligence**: Integrates web search (Tavily), web scraping, ArXiv academic papers, and PDF processing

💾 **RAG Pipeline**: Vector storage with ChromaDB and semantic retrieval for context-aware synthesis

⚡ **Fast Inference**: Powered by Groq's high-speed LLM inference (Llama 3.3)

🌐 **Dual Interfaces**: Both Streamlit web UI and CLI support

📊 **Real-time Tracking**: Visual progress monitoring and detailed result exploration

🏥 **Health API**: FastAPI-based health check and monitoring endpoints

📈 **Metrics & Monitoring**: Built-in metrics collection and system monitoring

## Architecture

```
┌─────────────┐
│   User      │
│  (Topic)    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│          LangGraph Workflow                     │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Plan     │→ │ Search   │→ │ Scrape   │     │
│  │ Research │  │ Web      │  │ Content  │     │
│  └──────────┘  └──────────┘  └──────────┘     │
│       │               │              │          │
│       ▼               ▼              ▼          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Search   │→ │ Process  │→ │ Embed &  │     │
│  │ ArXiv    │  │ Docs     │  │ Store    │     │
│  └──────────┘  └──────────┘  └──────────┘     │
│                                     │           │
│                                     ▼           │
│                              ┌──────────┐      │
│                              │ Retrieve │      │
│                              │ & Synth. │      │
│                              └──────────┘      │
└─────────────────────────────────────────────────┘
                      │
                      ▼
              ┌──────────────┐
              │ ChromaDB     │
              │ Vector Store │
              └──────────────┘
                      │
                      ▼
              ┌──────────────┐
              │  Research    │
              │  Report      │
              └──────────────┘
```

## Installation

1. **Clone or create the project directory**

2. **Install dependencies** (Python 3.11+ required):

```bash
pip install -e .
```

Or install manually:

```bash
pip install langgraph langchain langchain-groq langchain-community \
    langchain-text-splitters chromadb sentence-transformers tavily-python \
    arxiv beautifulsoup4 playwright pypdf2 pdfplumber streamlit \
    python-dotenv requests tiktoken fastapi uvicorn psutil
```

3. **Set up environment variables**

Create a `.env` file with your API keys:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key

# Optional: LangSmith tracing
LANGCHAIN_API_KEY=your_langchain_api_key
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=your_project_name
```

**Get API Keys:**
- Groq: https://console.groq.com/
- Tavily: https://tavily.com/
- LangSmith (optional): https://smith.langchain.com/

## Docker Installation (Alternative)

### Prerequisites
- Docker and Docker Compose installed on your system
- `.env` file with your API keys (see above)

### Quick Start with Docker Compose

1. **Ensure your `.env` file is configured** with your API keys

2. **Build and run the container:**
   ```bash
   docker-compose up -d
   ```

3. **Access the web interface:**
   Open your browser to `http://localhost:8501`

4. **View logs:**
   ```bash
   docker-compose logs -f
   ```

5. **Stop the container:**
   ```bash
   docker-compose down
   ```

### Using Docker Directly

1. **Build the Docker image:**
   ```bash
   docker build -t autonomous-research-assistant .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     --name research-assistant \
     -p 8501:8501 \
     -v $(pwd)/chroma_db:/app/chroma_db \
     -v $(pwd)/logs:/app/logs \
     --env-file .env \
     autonomous-research-assistant
   ```

3. **Access the web interface:**
   Open your browser to `http://localhost:8501`

### Docker Features

- **Persistent Storage**: ChromaDB data and logs are persisted in volumes
- **Environment Variables**: Automatically loaded from `.env` file
- **Health Checks**: Built-in health monitoring
- **Auto-restart**: Container automatically restarts on failure

### Docker Compose Configuration

The `docker-compose.yml` file includes:
- Port mapping (8501 for Streamlit)
- Volume mounts for data persistence
- Environment variable configuration
- Health checks and auto-restart policies

## Usage

### Web Interface (Recommended)

Launch the Streamlit web UI:

```bash
streamlit run app.py
```

Or:

```bash
python main.py --web
```

Then open your browser to `http://localhost:8501`

### Command Line Interface

Run research from the terminal:

```bash
python main.py "quantum computing applications"
```

The report will be displayed and saved to a markdown file.

### Health Check API

The project includes a FastAPI-based health check API for monitoring and observability:

```bash
python health_api.py
```

Or using uvicorn directly:

```bash
uvicorn health_api:app --host 0.0.0.0 --port 8080
```

**Available Endpoints:**
- `GET /` - Service information
- `GET /health` - Basic health check
- `GET /health/live` - Liveness probe
- `GET /health/ready` - Readiness probe (checks dependencies and vector store)
- `GET /metrics` - All collected metrics
- `GET /metrics/summary` - Metrics summary
- `GET /status` - Comprehensive service status

**Environment Variables:**
- `HEALTH_API_PORT` - Port for health API (default: 8080)
- `HEALTH_API_HOST` - Host for health API (default: 0.0.0.0)

The health API can run alongside the Streamlit application for production monitoring.

## Project Structure

```
autonomous-research-assistant/
├── agent/
│   ├── __init__.py          # Module exports
│   ├── state.py             # LangGraph state definition
│   ├── tools.py             # Research tools (search, scrape, etc.)
│   ├── graph.py             # LangGraph workflow
│   ├── rag.py               # RAG pipeline (chunking, embedding, retrieval)
│   ├── logger.py            # Logging configuration
│   ├── metrics.py           # Metrics collection
│   ├── monitoring.py        # System monitoring
│   └── validation.py        # Input validation
├── config.py                # Configuration and API clients
├── app.py                   # Streamlit web interface
├── main.py                  # CLI entry point
├── health_api.py            # FastAPI health check and monitoring API
├── pyproject.toml           # Dependencies and project configuration
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile               # Docker image definition
├── .env                     # Environment variables (API keys)
├── docs/                    # Sphinx documentation
├── tests/                   # Test suite
└── README.md                # This file
```

## How It Works

1. **Planning**: Agent analyzes the topic and creates a research strategy with search queries

2. **Web Search**: Executes multiple web searches using Tavily API

3. **Content Scraping**: Extracts full content from relevant web pages

4. **Academic Search**: Queries ArXiv for relevant academic papers

5. **Document Processing**: Processes PDFs and combines all gathered content

6. **Embedding & Storage**: Chunks documents, generates embeddings (sentence-transformers), and stores in ChromaDB

7. **Retrieval & Synthesis**: Retrieves relevant context and uses Groq LLM to synthesize a comprehensive research report

## Configuration

Edit `config.py` or use environment variables to customize:

- **Model**: Default is `llama-3.3-70b-versatile` (Groq)
  - Alternative models: `llama-3.1-8b-instant`, `mixtral-8x7b-32768`, `gemma2-9b-it`
  - See https://console.groq.com/docs/models for current models
- **Chunk Size**: Default 1000 characters with 200 overlap
- **Embedding Model**: Default `all-MiniLM-L6-v2`
- **Top-K Results**: Default 5 chunks for retrieval
- **Max Search Results**: Default 5 per query
- **Max ArXiv Papers**: Default 3

## Technologies Used

- **LangGraph**: State machine workflow orchestration
- **LangChain**: LLM framework and tool integration
- **Groq**: Fast LLM inference
- **ChromaDB**: Vector database for RAG
- **Sentence Transformers**: Document embeddings
- **Tavily**: Web search API
- **BeautifulSoup**: Web scraping
- **ArXiv**: Academic paper search
- **Streamlit**: Web interface
- **PyPDF2**: PDF processing
- **FastAPI**: Health check API
- **Uvicorn**: ASGI server for health API
- **psutil**: System monitoring

## Features in Detail

### Autonomous Operation
- Agent decides what to research and when to stop
- No human intervention required during research process
- Self-directed query generation

### Multi-Tool Integration
- **Tavily**: Real-time web search with relevance scoring
- **Web Scraper**: Full content extraction from URLs
- **ArXiv**: Academic paper discovery
- **PDF Processor**: Extract text from research papers

### RAG Pipeline
- Recursive text chunking for optimal context windows
- Semantic embeddings for similarity search
- Persistent vector storage with ChromaDB
- Context-aware synthesis using retrieved information

### User Interfaces
- **Web UI**: Rich, interactive Streamlit interface with real-time progress
- **CLI**: Simple command-line interface for scripting and automation
- **Health API**: RESTful API for monitoring, health checks, and metrics

### Monitoring & Observability
- **Metrics Collection**: Tracks research requests, API calls, durations, and errors
- **System Monitoring**: Memory usage tracking and system resource monitoring
- **Health Checks**: Liveness and readiness probes for containerized deployments
- **Logging**: Comprehensive logging with context tracking

## Troubleshooting

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -e .
```

### API Key Errors
Verify your `.env` file has valid API keys:
```bash
cat .env
```

### ChromaDB Issues
Delete the vector database to reset:
```bash
rm -rf chroma_db/
```

### Model Not Found
Ensure you have internet connection for downloading embedding models on first run.

## Code Quality

This project uses several code quality tools to maintain consistent code style and catch potential issues:

- **black**: Code formatter
- **isort**: Import sorter
- **ruff**: Fast linter and formatter
- **mypy**: Static type checker
- **pre-commit**: Git hooks for automated checks

### Installation

Install the development dependencies:

```bash
pip install -e ".[dev]"
```

### Pre-commit Hooks (Recommended)

Set up pre-commit hooks to automatically run code quality checks before each commit:

```bash
pre-commit install
```

The hooks will run automatically on commit. You can also run them manually:

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run hooks on staged files only
pre-commit run
```

### Manual Usage

**Format code with black:**
```bash
black .
```

**Sort imports with isort:**
```bash
isort .
```

**Lint and format with ruff:**
```bash
ruff check .
ruff format .
```

**Type check with mypy:**
```bash
mypy .
```

**Run all checks:**
```bash
black . ; isort . ; ruff check . ; ruff format . ; mypy .
```

All tools are configured in `pyproject.toml` with sensible defaults for Python 3.11+.

## Testing

The project includes a comprehensive test suite using pytest.

### Running Tests

Run all tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=agent --cov-report=html
```

Run tests with verbose output:
```bash
pytest -v
```

### Test Structure

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test component interactions
- **Test fixtures**: Shared test utilities in `tests/conftest.py`

The test suite maintains 80% code coverage threshold. Coverage reports are generated in HTML format for detailed analysis.

## Documentation

API documentation is generated using Sphinx with autodoc. The documentation includes detailed API references for all modules, classes, and functions.

### Building Documentation

First, ensure you have the development dependencies installed:

```bash
pip install -e ".[dev]"
```

Then, build the documentation:

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

The generated HTML documentation will be in `docs/_build/html/`. Open `index.html` in your browser to view it.

### Documentation Structure

- `docs/index.rst` - Main documentation index
- `docs/installation.rst` - Installation instructions
- `docs/quickstart.rst` - Quick start guide
- `docs/api/` - API reference documentation
  - `modules.rst` - All modules overview
  - `agent.rst` - Agent and graph documentation
  - `config.rst` - Configuration documentation
- `docs/examples.rst` - Usage examples

### Viewing Documentation Locally

After building, you can view the documentation by opening `docs/_build/html/index.html` in your web browser.

## Additional Features

### Metrics & Monitoring

The project includes comprehensive metrics collection and monitoring:

- **Counters**: Research requests, completions, failures, API calls, errors
- **Gauges**: Active research requests, total documents stored
- **Histograms**: Research duration, API call duration
- **System Metrics**: Memory usage tracking

Access metrics via the health API or programmatically through the `agent.metrics` module.

### Logging

Structured logging with context tracking:
- Research topic tracking
- Request ID correlation
- Error tracking with stack traces
- Configurable log levels

Logs are written to `logs/research_assistant.log` and `logs/research_assistant_errors.log`.

## Future Enhancements

- [ ] Support for more LLM providers (OpenAI, Anthropic, etc.)
- [ ] Multi-language support
- [ ] Citation extraction and formatting
- [ ] Custom tool addition via plugins
- [ ] Conversational refinement of reports
- [ ] Export to multiple formats (PDF, DOCX)
- [ ] Scheduled/batch research jobs
- [ ] Prometheus metrics export
- [ ] Grafana dashboards

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

Built with LangGraph, LangChain, Groq, and the amazing open-source community.
