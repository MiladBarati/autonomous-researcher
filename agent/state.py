"""
LangGraph State Definition for Autonomous Research Agent
"""

from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage


class ResearchState(TypedDict):
    """
    State structure for the research agent workflow.
    
    Attributes:
        topic: The research topic provided by the user
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
    research_plan: Optional[str]
    search_queries: List[str]
    search_results: List[Dict[str, Any]]
    scraped_content: List[Dict[str, Any]]
    arxiv_papers: List[Dict[str, Any]]
    pdf_content: List[Dict[str, Any]]
    all_documents: List[Dict[str, Any]]
    vector_store_id: Optional[str]
    retrieved_chunks: List[Dict[str, Any]]
    synthesis: Optional[str]
    messages: List[BaseMessage]
    step_count: int
    status: str


def create_initial_state(topic: str) -> ResearchState:
    """
    Create initial state for a new research task.
    
    Args:
        topic: The research topic
        
    Returns:
        Initial ResearchState
    """
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
        status="initialized"
    )

