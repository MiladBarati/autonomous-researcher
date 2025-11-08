Quick Start
===========

This guide will help you get started with the Autonomous Research Assistant.

Web Interface
-------------

Launch the Streamlit web UI:

.. code-block:: bash

   streamlit run app.py

Or:

.. code-block:: bash

   python main.py --web

Then open your browser to ``http://localhost:8501``

Command Line Interface
----------------------

Run research from the terminal:

.. code-block:: bash

   python main.py "quantum computing applications"

The report will be displayed and saved to a markdown file.

Programmatic Usage
------------------

You can also use the agent programmatically:

.. code-block:: python

   from agent import ResearchState, create_research_graph
   from agent.state import create_initial_state

   # Create the research graph
   graph = create_research_graph()

   # Initialize state with a research topic
   initial_state = create_initial_state("quantum computing applications")

   # Run the research workflow
   result = graph.invoke(initial_state)

   # Access the final synthesis
   print(result["synthesis"])

Configuration
-------------

Edit ``config.py`` or use environment variables to customize:

- **Model**: Default is ``llama-3.3-70b-versatile`` (Groq)
- **Chunk Size**: Default 1000 characters with 200 overlap
- **Embedding Model**: Default ``all-MiniLM-L6-v2``
- **Top-K Results**: Default 5 chunks for retrieval
- **Max Search Results**: Default 5 per query
- **Max ArXiv Papers**: Default 3

See :doc:`api/config` for detailed configuration options.
