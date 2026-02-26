Agent Module
============

The agent module provides the core research agent functionality using LangGraph.

The module uses a decorator pattern to handle cross-cutting concerns (logging,
performance monitoring, and validation) for all graph nodes, reducing code
duplication and ensuring consistent behavior.

Research Agent
--------------

.. autoclass:: agent.graph.ResearchAgent
   :members:
   :undoc-members:
   :show-inheritance:

Node Decorator
--------------

.. autofunction:: agent.graph._node_decorator
   :noindex:

Research Graph
--------------

.. autofunction:: agent.graph.create_research_graph
   :noindex:

State Management
----------------

.. autoclass:: agent.state.ResearchState
   :members:
   :undoc-members:
   :no-index:

.. autofunction:: agent.state.create_initial_state
   :noindex:
