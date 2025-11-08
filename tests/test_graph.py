from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.graph import ResearchAgent


class DummyLLM:
    def __init__(self, content: str):
        self._content: str = content

    def invoke(self, _messages: list[Any]) -> Any:
        return type("Resp", (), {"content": self._content})


@pytest.fixture()
def agent_with_stubs(monkeypatch):
    # Stub get_llm to avoid API validation
    monkeypatch.setattr(
        "agent.graph.get_llm",
        lambda: DummyLLM("RESEARCH PLAN:\nPlan\n\nSEARCH QUERIES:\n1. foo\n2. bar\n"),
        raising=False,
    )

    # Stub RAGPipeline used inside ResearchAgent
    with patch("agent.graph.RAGPipeline") as RAG:
        rag_instance = MagicMock()
        rag_instance.store_documents.return_value = 3
        rag_instance.get_collection_stats.return_value = {"document_count": 3}
        rag_instance.retrieve.return_value = [
            {"text": "t1", "metadata": {"title": "S1"}, "similarity_score": 0.9, "rank": 1}
        ]
        rag_instance.format_retrieved_context.return_value = "[Source 1: S1]\n t1"
        RAG.return_value = rag_instance

        # Build agent
        agent = ResearchAgent()

        # Replace llm for synthesis step too
        agent.llm = DummyLLM("SYNTHESIS TEXT")  # type: ignore[assignment]

        # Stub tools to deterministic outputs
        agent.tools.tavily.search = MagicMock(return_value=[{"title": "A", "url": "http://a", "content": "c", "score": 1.0}])
        agent.tools.scraper.scrape_multiple = MagicMock(return_value=[{"title": "A", "url": "http://a", "content": "content", "source": "web_scrape", "success": True}])
        agent.tools.arxiv.search = MagicMock(return_value=[{"title": "Paper", "authors": ["X"], "summary": "sum", "url": "u", "pdf_url": "pu"}])

    return agent


def test_plan_research_generates_queries(agent_with_stubs):
    from agent.state import create_initial_state

    state = create_initial_state("Test Topic")
    out = agent_with_stubs.plan_research(state)
    assert out["search_queries"]
    assert out["status"] == "planned"


def test_full_graph_flow_completes(agent_with_stubs):
    from agent.state import create_initial_state

    initial_state = create_initial_state("AI safety")
    final_state = agent_with_stubs.graph.invoke(initial_state)

    assert final_state["status"] == "completed"
    assert "SYNTHESIS TEXT" in final_state["synthesis"]
    # Steps: plan, search_web, scrape, arxiv, process, store, retrieve = 7
    assert final_state["step_count"] >= 7
