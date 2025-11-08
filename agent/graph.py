"""
LangGraph Workflow for Autonomous Research Agent

Defines the state graph with nodes for:
- Research planning
- Web search
- Content scraping
- ArXiv search
- Document processing
- Vector storage
- Retrieval and synthesis
"""

from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from agent.logger import get_logger, set_logging_context
from agent.rag import RAGPipeline
from agent.state import ResearchState
from agent.tools import ToolManager
from agent.validation import ValidationError, validate_query, validate_topic
from config import Config, get_llm

logger = get_logger("graph")


class ResearchAgent:
    """Autonomous Research Agent using LangGraph"""

    def __init__(self) -> None:
        self.llm: ChatGroq = get_llm()
        self.tools: ToolManager = ToolManager()
        self.rag: RAGPipeline = RAGPipeline()
        # Note: Using Any because langgraph doesn't have complete type stubs for CompiledGraph
        self.graph: Any = self._build_graph()

    def _build_graph(self) -> Any:
        # Note: Returns CompiledGraph, but langgraph doesn't have complete type stubs
        """Build the LangGraph state machine"""

        # Create state graph
        workflow = StateGraph(ResearchState)

        # Add nodes
        workflow.add_node("plan_research", self.plan_research)
        workflow.add_node("search_web", self.search_web)
        workflow.add_node("scrape_content", self.scrape_content)
        workflow.add_node("search_arxiv", self.search_arxiv)
        workflow.add_node("process_documents", self.process_documents)
        workflow.add_node("embed_and_store", self.embed_and_store)
        workflow.add_node("retrieve_and_synthesize", self.retrieve_and_synthesize)

        # Set entry point
        workflow.set_entry_point("plan_research")

        # Add edges
        workflow.add_edge("plan_research", "search_web")
        workflow.add_edge("search_web", "scrape_content")
        workflow.add_edge("scrape_content", "search_arxiv")
        workflow.add_edge("search_arxiv", "process_documents")
        workflow.add_edge("process_documents", "embed_and_store")
        workflow.add_edge("embed_and_store", "retrieve_and_synthesize")
        workflow.add_edge("retrieve_and_synthesize", END)

        return workflow.compile()

    def plan_research(self, state: ResearchState) -> dict[str, Any]:
        """
        Node: Create research plan and generate search queries.

        Args:
            state: Current research state

        Returns:
            Updated state with research plan and queries
        """
        set_logging_context(request_id=state.get("request_id"), topic=state.get("topic"))
        logger.info("=== PLANNING RESEARCH ===")

        # Validate topic
        topic: str = state["topic"]
        try:
            topic = validate_topic(topic)
        except ValidationError as e:
            logger.error(f"Invalid topic in state: {e}")
            raise

        # Create planning prompt
        planning_prompt: str = f"""You are an autonomous research agent. Your task is to create a comprehensive research plan for the following topic:

Topic: {topic}

Please:
1. Analyze the topic and identify key areas to research
2. Generate 3-5 specific search queries that will help gather comprehensive information
3. Explain your research strategy

Format your response as:
RESEARCH PLAN:
[Your analysis and strategy]

SEARCH QUERIES:
1. [Query 1]
2. [Query 2]
3. [Query 3]
...
"""

        # Get LLM response
        messages: list[SystemMessage | HumanMessage] = [
            SystemMessage(content="You are a research planning expert."),
            HumanMessage(content=planning_prompt),
        ]

        response = self.llm.invoke(messages)
        # Handle union type: content can be str | list[str | dict[Any, Any]]
        content = response.content
        plan_text: str = content if isinstance(content, str) else str(content)

        # Extract search queries from response
        queries: list[str] = []
        lines: list[str] = plan_text.split("\n")
        in_queries_section: bool = False

        for line in lines:
            line = line.strip()
            if "SEARCH QUERIES:" in line:
                in_queries_section = True
                continue
            if in_queries_section and line and (line[0].isdigit() or line.startswith("-")):
                # Extract query text, removing numbering
                query: str = line.split(".", 1)[-1].strip() if "." in line else line.lstrip("- ")
                if query:
                    # Validate and sanitize query
                    try:
                        query = validate_query(query)
                        queries.append(query)
                    except ValidationError as e:
                        logger.warning(f"Skipping invalid query: {e}")
                        continue

        # Fallback: if no queries extracted, use the topic
        if not queries:
            try:
                queries = [
                    validate_query(topic),
                    validate_query(f"{topic} overview"),
                    validate_query(f"{topic} recent developments"),
                ]
            except ValidationError:
                # If topic itself is invalid, use a safe fallback
                queries = [validate_query("research")] if topic else []

        logger.info("Research Plan Created")
        logger.info(f"Generated {len(queries)} search queries")

        return {
            "research_plan": plan_text,
            "search_queries": queries,
            "messages": state["messages"]
            + [HumanMessage(content=planning_prompt), AIMessage(content=plan_text)],
            "step_count": state["step_count"] + 1,
            "status": "planned",
        }

    def search_web(self, state: ResearchState) -> dict[str, Any]:
        """
        Node: Execute web searches using Tavily.

        Args:
            state: Current research state

        Returns:
            Updated state with search results
        """
        set_logging_context(request_id=state.get("request_id"), topic=state.get("topic"))
        logger.info("=== SEARCHING WEB ===")

        queries: list[str] = state["search_queries"]
        all_results: list[dict[str, Any]] = []

        # Validate queries before processing
        valid_queries: list[str] = []
        for query in queries[:5]:  # Limit to 5 queries
            try:
                validated_query = validate_query(query)
                valid_queries.append(validated_query)
            except ValidationError as e:
                logger.warning(f"Skipping invalid query '{query}': {e}")
                continue

        for i, query in enumerate(valid_queries, 1):
            logger.debug(f"Query {i}/{len(valid_queries)}: {query}")
            results = self.tools.tavily.search(query, max_results=3)
            all_results.extend(results)
            logger.debug(f"  Found {len(results)} results")

        logger.info(f"Total search results: {len(all_results)}")

        return {
            "search_results": all_results,
            "step_count": state["step_count"] + 1,
            "status": "searched",
        }

    def scrape_content(self, state: ResearchState) -> dict[str, Any]:
        """
        Node: Scrape full content from search result URLs.

        Args:
            state: Current research state

        Returns:
            Updated state with scraped content
        """
        set_logging_context(request_id=state.get("request_id"), topic=state.get("topic"))
        logger.info("=== SCRAPING CONTENT ===")

        search_results: list[dict[str, Any]] = state["search_results"]
        urls: list[str] = [result["url"] for result in search_results if result.get("url")]

        logger.info(f"Scraping {min(len(urls), Config.MAX_SCRAPE_URLS)} URLs...")
        scraped: list[dict[str, Any]] = self.tools.scraper.scrape_multiple(
            urls, max_urls=Config.MAX_SCRAPE_URLS
        )

        logger.info(f"Successfully scraped {len(scraped)} pages")

        return {
            "scraped_content": scraped,
            "step_count": state["step_count"] + 1,
            "status": "scraped",
        }

    def search_arxiv(self, state: ResearchState) -> dict[str, Any]:
        """
        Node: Search ArXiv for academic papers.

        Args:
            state: Current research state

        Returns:
            Updated state with arxiv papers
        """
        set_logging_context(request_id=state.get("request_id"), topic=state.get("topic"))
        logger.info("=== SEARCHING ARXIV ===")

        # Validate topic
        topic: str = state["topic"]
        try:
            topic = validate_topic(topic)
        except ValidationError as e:
            logger.error(f"Invalid topic for ArXiv search: {e}")
            return {
                "arxiv_papers": [],
                "step_count": state["step_count"] + 1,
                "status": "arxiv_searched",
            }

        papers: list[dict[str, Any]] = self.tools.arxiv.search(
            topic, max_results=Config.MAX_ARXIV_RESULTS
        )

        logger.info(f"Found {len(papers)} ArXiv papers")

        return {
            "arxiv_papers": papers,
            "step_count": state["step_count"] + 1,
            "status": "arxiv_searched",
        }

    def process_documents(self, state: ResearchState) -> dict[str, Any]:
        """
        Node: Process PDFs and combine all documents.

        Args:
            state: Current research state

        Returns:
            Updated state with processed documents
        """
        set_logging_context(request_id=state.get("request_id"), topic=state.get("topic"))
        logger.info("=== PROCESSING DOCUMENTS ===")

        all_docs: list[dict[str, Any]] = []

        # Add scraped web content
        all_docs.extend(state["scraped_content"])
        logger.debug(f"Added {len(state['scraped_content'])} web documents")

        # Add arxiv papers (summaries)
        for paper in state["arxiv_papers"]:
            all_docs.append(
                {
                    "title": paper["title"],
                    "content": f"{paper['title']}\n\nAuthors: {', '.join(paper['authors'])}\n\n{paper['summary']}",
                    "url": paper["url"],
                    "source": "arxiv",
                    "authors": paper["authors"],
                }
            )
        logger.debug(f"Added {len(state['arxiv_papers'])} ArXiv paper summaries")

        # Optional: Process ArXiv PDFs (commented out for speed, can be enabled)
        # pdf_content = self.tools.pdf_processor.extract_from_arxiv_papers(state["arxiv_papers"][:2])
        # all_docs.extend(pdf_content)
        # logger.debug(f"Processed {len(pdf_content)} PDFs")

        logger.info(f"Total documents: {len(all_docs)}")

        return {
            "all_documents": all_docs,
            "step_count": state["step_count"] + 1,
            "status": "documents_processed",
        }

    def embed_and_store(self, state: ResearchState) -> dict[str, Any]:
        """
        Node: Embed documents and store in vector database.

        Args:
            state: Current research state

        Returns:
            Updated state with vector store info
        """
        set_logging_context(request_id=state.get("request_id"), topic=state.get("topic"))
        logger.info("=== EMBEDDING AND STORING ===")

        documents: list[dict[str, Any]] = state["all_documents"]

        if not documents:
            logger.warning("No documents to store")
            return {"step_count": state["step_count"] + 1, "status": "no_documents"}

        # Store documents in vector DB
        self.rag.store_documents(documents)

        stats: dict[str, Any] = self.rag.get_collection_stats()
        logger.info(f"Vector store stats: {stats}")

        return {
            "vector_store_id": self.rag.collection_name,
            "step_count": state["step_count"] + 1,
            "status": "stored",
        }

    def retrieve_and_synthesize(self, state: ResearchState) -> dict[str, Any]:
        """
        Node: Retrieve relevant context and synthesize final report.

        Args:
            state: Current research state

        Returns:
            Updated state with synthesis
        """
        set_logging_context(request_id=state.get("request_id"), topic=state.get("topic"))
        logger.info("=== RETRIEVING AND SYNTHESIZING ===")

        # Validate topic
        topic: str = state["topic"]
        try:
            topic = validate_topic(topic)
        except ValidationError as e:
            logger.error(f"Invalid topic for synthesis: {e}")
            raise

        # Retrieve relevant chunks
        logger.debug("Retrieving relevant context...")
        retrieved_chunks: list[dict[str, Any]] = self.rag.retrieve(topic, top_k=10)
        logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks")

        # Format context
        context: str = self.rag.format_retrieved_context(retrieved_chunks)

        # Create synthesis prompt
        synthesis_prompt: str = f"""You are a research analyst tasked with synthesizing information on the following topic:

Topic: {topic}

Based on the research gathered (see context below), create a comprehensive research report that:
1. Provides a clear overview of the topic
2. Summarizes key findings and insights
3. Highlights important facts, statistics, and developments
4. Cites sources appropriately
5. Is well-structured with sections

CONTEXT FROM RESEARCH:
{context}

Please write a comprehensive, well-organized research report (aim for 800-1500 words):
"""

        # Generate synthesis
        logger.debug("Generating synthesis...")
        messages: list[SystemMessage | HumanMessage] = [
            SystemMessage(
                content="You are an expert research analyst who synthesizes information into clear, comprehensive reports."
            ),
            HumanMessage(content=synthesis_prompt),
        ]

        response = self.llm.invoke(messages)
        # Handle union type: content can be str | list[str | dict[Any, Any]]
        content = response.content
        synthesis: str = content if isinstance(content, str) else str(content)

        logger.info(f"Synthesis completed ({len(synthesis)} characters)")

        return {
            "retrieved_chunks": retrieved_chunks,
            "synthesis": synthesis,
            "messages": state["messages"]
            + [HumanMessage(content=synthesis_prompt), AIMessage(content=synthesis)],
            "step_count": state["step_count"] + 1,
            "status": "completed",
        }

    def research(self, topic: str) -> dict[str, Any]:
        """
        Execute full research workflow.

        Args:
            topic: Research topic

        Returns:
            Final state with synthesis
        """
        from agent.state import create_initial_state

        # Validate and sanitize topic before creating state
        try:
            topic = validate_topic(topic)
        except ValidationError as e:
            logger.error(f"Invalid research topic: {e}")
            raise

        # Create initial state
        initial_state: ResearchState = create_initial_state(topic)

        # Set logging context for the entire research session
        set_logging_context(request_id=initial_state["request_id"], topic=topic)

        logger.info("=" * 60)
        logger.info("STARTING AUTONOMOUS RESEARCH")
        logger.info(f"Topic: {topic}")
        logger.info(f"Request ID: {initial_state['request_id']}")
        logger.info("=" * 60)

        # Run the graph
        final_state: ResearchState = self.graph.invoke(initial_state)

        logger.info("=" * 60)
        logger.info("RESEARCH COMPLETED")
        logger.info(f"Status: {final_state['status']}")
        logger.info(f"Steps taken: {final_state['step_count']}")
        logger.info("=" * 60)

        # ResearchState is a TypedDict, which is compatible with dict[str, Any]
        return cast(dict[str, Any], final_state)


def create_research_graph() -> ResearchAgent:
    """
    Factory function to create research agent.

    Returns:
        Configured ResearchAgent instance
    """
    return ResearchAgent()
