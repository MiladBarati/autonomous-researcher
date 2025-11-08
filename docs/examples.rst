Examples
========

Basic Usage
-----------

Simple research query:

.. code-block:: python

   from agent import create_research_graph
   from agent.state import create_initial_state

   graph = create_research_graph()
   state = create_initial_state("machine learning")
   result = graph.invoke(state)
   print(result["synthesis"])

Custom Configuration
--------------------

Using custom model settings:

.. code-block:: python

   from config import get_llm, Config
   from agent.graph import ResearchAgent

   # Override model configuration
   llm = get_llm(temperature=0.5, max_tokens=4096)
   agent = ResearchAgent()
   agent.llm = llm

   # Run research
   state = create_initial_state("quantum computing")
   result = agent.graph.invoke(state)

Direct Tool Usage
-----------------

Using tools directly:

.. code-block:: python

   from agent.tools import ToolManager

   tools = ToolManager()

   # Web search
   results = tools.web_search("Python async programming")
   print(results)

   # Scrape content
   content = tools.scrape_url("https://example.com/article")
   print(content)

RAG Pipeline Usage
------------------

Using RAG pipeline directly:

.. code-block:: python

   from agent.rag import RAGPipeline

   rag = RAGPipeline()

   # Add documents
   documents = [
       {"content": "Document 1 content...", "metadata": {"source": "web"}},
       {"content": "Document 2 content...", "metadata": {"source": "arxiv"}},
   ]
   rag.add_documents(documents)

   # Retrieve relevant chunks
   query = "What is machine learning?"
   chunks = rag.retrieve(query, top_k=5)
   print(chunks)
