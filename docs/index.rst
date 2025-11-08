Autonomous Research Assistant Documentation
============================================

Welcome to the API documentation for the Autonomous Research Assistant, an event-driven AI research agent that autonomously retrieves, summarizes, and synthesizes web content using LangGraph, LangChain, and Retrieval-Augmented Generation (RAG).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/modules
   api/agent
   api/config
   examples

Overview
--------

The Autonomous Research Assistant is a sophisticated research tool that combines:

- **Autonomous Research**: Self-directed topic research with minimal human intervention
- **Multi-Source Intelligence**: Integrates web search (Tavily), web scraping, ArXiv academic papers, and PDF processing
- **RAG Pipeline**: Vector storage with ChromaDB and semantic retrieval for context-aware synthesis
- **Fast Inference**: Powered by Groq's high-speed LLM inference (Llama 3.1)
- **Dual Interfaces**: Both Streamlit web UI and CLI support

Architecture
------------

The system uses a LangGraph-based workflow that orchestrates multiple research tools:

1. **Planning**: Agent analyzes the topic and creates a research strategy
2. **Web Search**: Executes multiple web searches using Tavily API
3. **Content Scraping**: Extracts full content from relevant web pages
4. **Academic Search**: Queries ArXiv for relevant academic papers
5. **Document Processing**: Processes PDFs and combines all gathered content
6. **Embedding & Storage**: Chunks documents, generates embeddings, and stores in ChromaDB
7. **Retrieval & Synthesis**: Retrieves relevant context and synthesizes a comprehensive research report

API Reference
-------------

The API is organized into the following modules:

.. toctree::
   :maxdepth: 1

   api/modules
   api/agent
   api/config

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
