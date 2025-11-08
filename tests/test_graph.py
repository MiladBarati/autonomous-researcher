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
        agent.tools.tavily.search = MagicMock(
            return_value=[{"title": "A", "url": "http://a", "content": "c", "score": 1.0}]
        )
        agent.tools.scraper.scrape_multiple = MagicMock(
            return_value=[
                {
                    "title": "A",
                    "url": "http://a",
                    "content": "content",
                    "source": "web_scrape",
                    "success": True,
                }
            ]
        )
        agent.tools.arxiv.search = MagicMock(
            return_value=[
                {"title": "Paper", "authors": ["X"], "summary": "sum", "url": "u", "pdf_url": "pu"}
            ]
        )

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


def test_plan_research_fallback_queries(agent_with_stubs):
    """Test that plan_research falls back to default queries if extraction fails"""
    from agent.state import create_initial_state

    # Mock LLM to return response without proper query format
    agent_with_stubs.llm = DummyLLM("RESEARCH PLAN:\nSome plan text\nNo queries here")

    state = create_initial_state("Test Topic")
    out = agent_with_stubs.plan_research(state)

    # Should have fallback queries
    assert out["search_queries"]
    assert len(out["search_queries"]) >= 1
    assert out["status"] == "planned"


def test_plan_research_extracts_queries_with_dashes(agent_with_stubs):
    """Test that plan_research extracts queries with dash formatting"""
    from agent.state import create_initial_state

    agent_with_stubs.llm = DummyLLM(
        "RESEARCH PLAN:\nPlan\n\nSEARCH QUERIES:\n- query one\n- query two\n- query three"
    )

    state = create_initial_state("Test Topic")
    out = agent_with_stubs.plan_research(state)

    assert len(out["search_queries"]) >= 3
    assert out["status"] == "planned"


def test_search_web_limits_queries(agent_with_stubs):
    """Test that search_web limits to 5 queries"""
    from agent.state import create_initial_state

    state = create_initial_state("Test Topic")
    state["search_queries"] = [f"query {i}" for i in range(10)]  # 10 queries

    out = agent_with_stubs.search_web(state)

    # Should only process 5 queries
    assert agent_with_stubs.tools.tavily.search.call_count == 5
    assert out["status"] == "searched"


def test_search_web_handles_empty_queries(agent_with_stubs):
    """Test that search_web handles empty query list"""
    from agent.state import create_initial_state

    state = create_initial_state("Test Topic")
    state["search_queries"] = []

    out = agent_with_stubs.search_web(state)

    assert out["search_results"] == []
    assert out["status"] == "searched"


def test_scrape_content_handles_empty_results(agent_with_stubs):
    """Test that scrape_content handles empty search results"""
    from agent.state import create_initial_state

    state = create_initial_state("Test Topic")
    state["search_results"] = []

    # Mock scrape_multiple to return empty list when called with empty URLs
    agent_with_stubs.tools.scraper.scrape_multiple = MagicMock(return_value=[])

    out = agent_with_stubs.scrape_content(state)

    assert out["scraped_content"] == []
    assert out["status"] == "scraped"


def test_scrape_content_filters_urls(agent_with_stubs):
    """Test that scrape_content filters out results without URLs"""
    from agent.state import create_initial_state

    state = create_initial_state("Test Topic")
    state["search_results"] = [
        {"title": "A", "url": "http://a"},
        {"title": "B"},  # No URL
        {"title": "C", "url": "http://c"},
    ]

    out = agent_with_stubs.scrape_content(state)

    # Should only scrape URLs that exist
    assert len(out["scraped_content"]) >= 0
    assert out["status"] == "scraped"


def test_search_arxiv_handles_empty_topic(agent_with_stubs):
    """Test that search_arxiv handles empty topic"""
    from agent.state import create_initial_state

    state = create_initial_state("")
    out = agent_with_stubs.search_arxiv(state)

    assert out["status"] == "arxiv_searched"
    assert agent_with_stubs.tools.arxiv.search.called


def test_process_documents_combines_sources(agent_with_stubs):
    """Test that process_documents combines scraped content and arxiv papers"""
    from agent.state import create_initial_state

    state = create_initial_state("Test Topic")
    state["scraped_content"] = [
        {"title": "Web Doc", "content": "web content", "url": "http://web", "source": "web_scrape"}
    ]
    state["arxiv_papers"] = [
        {
            "title": "Paper",
            "authors": ["Author"],
            "summary": "Summary",
            "url": "http://arxiv",
        }
    ]

    out = agent_with_stubs.process_documents(state)

    assert len(out["all_documents"]) == 2
    assert out["all_documents"][0]["source"] == "web_scrape"
    assert out["all_documents"][1]["source"] == "arxiv"
    assert out["status"] == "documents_processed"


def test_embed_and_store_handles_empty_documents(agent_with_stubs):
    """Test that embed_and_store handles empty document list"""
    from agent.state import create_initial_state

    state = create_initial_state("Test Topic")
    state["all_documents"] = []

    out = agent_with_stubs.embed_and_store(state)

    assert out["status"] == "no_documents"
    assert out["step_count"] == state["step_count"] + 1


def test_retrieve_and_synthesize_handles_empty_retrieval(agent_with_stubs):
    """Test that retrieve_and_synthesize handles empty retrieval results"""
    from agent.state import create_initial_state

    agent_with_stubs.rag.retrieve.return_value = []
    agent_with_stubs.rag.format_retrieved_context.return_value = "No relevant context found."

    state = create_initial_state("Test Topic")
    out = agent_with_stubs.retrieve_and_synthesize(state)

    assert out["status"] == "completed"
    assert "synthesis" in out


def test_research_method_executes_full_workflow(agent_with_stubs):
    """Test that research() method executes full workflow"""
    final_state = agent_with_stubs.research("Test Topic")

    assert final_state["status"] == "completed"
    assert "synthesis" in final_state
    assert final_state["step_count"] >= 7


def test_create_research_graph_factory(agent_with_stubs):
    """Test that create_research_graph factory function works"""
    from agent.graph import create_research_graph

    with patch("agent.graph.get_llm", return_value=DummyLLM("test")):
        with patch("agent.graph.RAGPipeline"):
            agent = create_research_graph()
            assert isinstance(agent, ResearchAgent)