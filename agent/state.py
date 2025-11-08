"""
LangGraph State Definition for Autonomous Research Agent
"""

import uuid
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage

from agent.validation import ValidationError, validate_topic


class ResearchState(TypedDict):
    """
    State structure for the research agent workflow.

    Attributes:
        topic: The research topic provided by the user
        request_id: Unique identifier for this research request
        research_plan: The agent's planned research strategy
        search_queries: List of search queries to execute
        search_results: Results from web searches
        scraped_content: Content extracted from web pages
        arxiv_papers: Academic papers found on arxiv
        pdf_content: Text extracted from PDF documents
        all_documents: Combined list of all collected documents
        vector_store_id: ID of the ChromaDB collection
        retrieved_chunks: Context chunks retrieved from vector store
        synthesis: Final research report
        messages: Conversation/reasoning history
        step_count: Number of steps taken (to prevent infinite loops)
        status: Current status of the research process
    """

    topic: str
    request_id: str
    research_plan: str | None
    search_queries: list[str]
    search_results: list[dict[str, Any]]
    scraped_content: list[dict[str, Any]]
    arxiv_papers: list[dict[str, Any]]
    pdf_content: list[dict[str, Any]]
    all_documents: list[dict[str, Any]]
    vector_store_id: str | None
    retrieved_chunks: list[dict[str, Any]]
    synthesis: str | None
    messages: list[BaseMessage]
    step_count: int
    status: str


def create_initial_state(topic: str, request_id: str | None = None) -> ResearchState:
    """
    Create initial state for a new research task.

    Args:
        topic: The research topic
        request_id: Optional request ID (generated if not provided)

    Returns:
        Initial ResearchState

    Raises:
        ValidationError: If topic is invalid
    """
    # Validate and sanitize topic
    try:
        topic = validate_topic(topic)
    except ValidationError as e:
        raise ValidationError(f"Invalid topic for initial state: {e}") from e

    if request_id is None:
        request_id = str(uuid.uuid4())

    # Ensure request_id is a string (not None) for TypedDict
    request_id_str: str = request_id

    return ResearchState(
        topic=topic,
        research_plan=None,
        search_queries=[],
        search_results=[],
        scraped_content=[],
        arxiv_papers=[],
        pdf_content=[],
        all_documents=[],
        vector_store_id=None,
        retrieved_chunks=[],
        synthesis=None,
        messages=[],
        step_count=0,
        status="initialized",
        request_id=request_id_str,
    )
