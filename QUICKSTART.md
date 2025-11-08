# Quick Start Guide

## Installation Steps

### 1. Install Dependencies

The project requires Python 3.11 or higher. Install all dependencies:

```bash
pip install langgraph langchain langchain-groq langchain-community langchain-text-splitters chromadb sentence-transformers tavily-python arxiv beautifulsoup4 playwright pypdf2 pdfplumber streamlit python-dotenv requests tiktoken
```

Or use the pyproject.toml:

```bash
pip install -e .
```

**Note on Windows**: If you get an "externally-managed-environment" error, either:
- Use a virtual environment: `python -m venv venv` then activate it
- Or use the `--break-system-packages` flag (not recommended)

### 2. Verify API Keys

Make sure your `.env` file has valid API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Get free API keys:
- **Groq**: https://console.groq.com/ (Free tier available)
- **Tavily**: https://tavily.com/ (Free tier: 1000 searches/month)

### 3. Run the Application

#### Web Interface (Recommended)

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

#### Command Line

```bash
python main.py "your research topic here"
```

Example:
```bash
python main.py "latest developments in quantum computing"
```

## First Research Example

### Using Web UI

1. Launch: `streamlit run app.py`
2. Enter a topic like "AI ethics in healthcare"
3. Click "Start Research"
4. Watch the progress as the agent:
   - Plans the research
   - Searches the web
   - Scrapes content
   - Finds academic papers
   - Stores in vector database
   - Synthesizes a comprehensive report
5. Download the report when complete

### Using CLI

```bash
python main.py "climate change solutions"
```

The report will be:
- Displayed in the terminal
- Saved to `research_report_climate_change_solutions.md`

## What Happens During Research

The autonomous agent follows this workflow:

1. **Planning** (~5 seconds)
   - Analyzes your topic
   - Creates research strategy
   - Generates 3-5 search queries

2. **Web Search** (~10 seconds)
   - Executes searches via Tavily
   - Finds 15+ relevant sources
   - Ranks by relevance

3. **Content Scraping** (~15 seconds)
   - Extracts full content from top 5 URLs
   - Cleans and formats text
   - Removes ads and navigation

4. **Academic Search** (~5 seconds)
   - Queries ArXiv for papers
   - Retrieves abstracts and metadata
   - Finds 3 most relevant papers

5. **Document Processing** (~2 seconds)
   - Combines all sources
   - Organizes by type

6. **Embedding & Storage** (~10 seconds)
   - Chunks documents (1000 chars each)
   - Generates embeddings
   - Stores in ChromaDB vector database

7. **Synthesis** (~15 seconds)
   - Retrieves most relevant chunks
   - Uses Groq LLM to synthesize
   - Creates comprehensive report (800-1500 words)

**Total time**: ~60 seconds on average

## Troubleshooting

### "No module named 'langchain_groq'"

Dependencies not installed. Run:
```bash
pip install langgraph langchain langchain-groq langchain-community
```

### "Missing required API keys"

Check your `.env` file:
```bash
# On Windows
type .env

# On Linux/Mac
cat .env
```

Make sure `GROQ_API_KEY` and `TAVILY_API_KEY` are set.

### Slow First Run

The first run downloads the embedding model (~80MB). Subsequent runs are much faster.

### ChromaDB Errors

Reset the vector database:
```bash
# Windows
rmdir /s chroma_db

# Linux/Mac
rm -rf chroma_db/
```

### Streamlit Port Already in Use

Run on a different port:
```bash
streamlit run app.py --server.port 8502
```

## Tips for Best Results

### Good Topics

✅ "recent advances in renewable energy storage"
✅ "impact of social media on mental health"
✅ "quantum computing applications in cryptography"
✅ "CRISPR gene editing ethics and regulations"

### Topics to Avoid

❌ Too broad: "science" or "technology"
❌ Too specific: "equation 7 in paper XYZ"
❌ Very recent: events from last 24 hours (web search may not have indexed yet)

### Research Quality

- More specific topics get better results
- Academic topics benefit from ArXiv integration
- Current events topics leverage web search more
- Technical topics get both academic and practical sources

## Configuration

Edit `config.py` to customize:

```python
# Model settings
MODEL_NAME = "llama-3.3-70b-versatile"  # Fast and capable (updated Nov 2024)
# Alternative models: "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"
MODEL_TEMPERATURE = 0.7                  # Creativity vs accuracy

# RAG settings
CHUNK_SIZE = 1000           # Chunk size for documents
TOP_K_RESULTS = 5          # Context chunks for synthesis

# Search settings
MAX_SEARCH_RESULTS = 5     # Results per query
MAX_ARXIV_RESULTS = 3      # Academic papers to find
MAX_SCRAPE_URLS = 5        # URLs to scrape fully
```

**Note**: If you get a "model_decommissioned" error, update `MODEL_NAME` in `config.py` with a current model from https://console.groq.com/docs/models

## Next Steps

1. Try researching your favorite topic
2. Explore the expandable sections in the web UI to see sources
3. Check the `chroma_db/` directory to see persisted embeddings
4. Try the CLI for automation and scripting
5. Experiment with different topics and compare reports

Enjoy your autonomous research assistant! 🚀
