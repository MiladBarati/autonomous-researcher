# Autonomous Research Assistant

An event-driven AI research agent that autonomously retrieves, summarizes, and synthesizes web content using LangGraph, LangChain, and Retrieval-Augmented Generation (RAG).

## Features

🤖 **Autonomous Research**: Self-directed topic research with minimal human intervention

🔍 **Multi-Source Intelligence**: Integrates web search (Tavily), web scraping, ArXiv academic papers, and PDF processing

💾 **RAG Pipeline**: Vector storage with ChromaDB and semantic retrieval for context-aware synthesis

⚡ **Fast Inference**: Powered by Groq's high-speed LLM inference (Llama 3.1)

🌐 **Dual Interfaces**: Both Streamlit web UI and CLI support

📊 **Real-time Tracking**: Visual progress monitoring and detailed result exploration

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
    python-dotenv requests tiktoken
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

## Project Structure

```
autonomous-research-assistant/
├── agent/
│   ├── __init__.py          # Module exports
│   ├── state.py             # LangGraph state definition
│   ├── tools.py             # Research tools (search, scrape, etc.)
│   ├── graph.py             # LangGraph workflow
│   └── rag.py               # RAG pipeline (chunking, embedding, retrieval)
├── config.py                # Configuration and API clients
├── app.py                   # Streamlit web interface
├── main.py                  # CLI entry point
├── pyproject.toml           # Dependencies
├── .env                     # Environment variables (API keys)
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

### Installation

Install the development dependencies:

```bash
pip install -e ".[dev]"
```

### Usage

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
black . && isort . && ruff check . && ruff format . && mypy .
```

All tools are configured in `pyproject.toml` with sensible defaults for Python 3.11+.

## Future Enhancements

- [ ] Support for more LLM providers (OpenAI, Anthropic, etc.)
- [ ] Multi-language support
- [ ] Citation extraction and formatting
- [ ] Custom tool addition via plugins
- [ ] Conversational refinement of reports
- [ ] Export to multiple formats (PDF, DOCX)
- [ ] Scheduled/batch research jobs

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

Built with LangGraph, LangChain, Groq, and the amazing open-source community.

