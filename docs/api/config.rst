Configuration Module
====================

The configuration module provides settings and API client initialization.

Config Class
------------

.. autoclass:: config.Config
   :members:
   :undoc-members:
   :show-inheritance:

   Configuration class for the research assistant.

   Attributes:
      GROQ_API_KEY: Groq API key for LLM inference
      TAVILY_API_KEY: Tavily API key for web search
      MODEL_NAME: LLM model name (default: "llama-3.3-70b-versatile")
      MODEL_TEMPERATURE: LLM temperature (default: 0.7)
      MAX_TOKENS: Maximum tokens for LLM responses (default: 8192)
      CHUNK_SIZE: Document chunk size for RAG (default: 1000)
      CHUNK_OVERLAP: Chunk overlap size (default: 200)
      EMBEDDING_MODEL: Embedding model name (default: "all-MiniLM-L6-v2")
      TOP_K_RESULTS: Number of top results for retrieval (default: 5)
      MAX_SEARCH_RESULTS: Maximum search results per query (default: 5)
      MAX_ARXIV_RESULTS: Maximum ArXiv papers to fetch (default: 3)
      MAX_ITERATIONS: Maximum workflow iterations (default: 10)

   Methods:
      validate: Validate that required API keys are present

LLM Factory
-----------

.. autofunction:: config.get_llm
   :noindex:

   Get configured Groq LLM instance.

   Args:
      temperature: Override default temperature
      max_tokens: Override default max tokens

   Returns:
      Configured ChatGroq instance
