"""Tests for agent.state module"""

import uuid

from agent.state import create_initial_state


def test_create_initial_state() -> None:
    """Test creating initial state with topic"""
    state = create_initial_state("Test Topic")
    assert state["topic"] == "Test Topic"
    assert state["status"] == "initialized"
    assert state["step_count"] == 0
    assert state["research_plan"] is None
    assert state["search_queries"] == []
    assert state["search_results"] == []
    assert state["scraped_content"] == []
    assert state["arxiv_papers"] == []
    assert state["pdf_content"] == []
    assert state["all_documents"] == []
    assert state["vector_store_id"] is None
    assert state["retrieved_chunks"] == []
    assert state["synthesis"] is None
    assert state["messages"] == []
    assert isinstance(state["request_id"], str)
    assert len(state["request_id"]) > 0


def test_create_initial_state_with_request_id() -> None:
    """Test creating initial state with custom request_id"""
    custom_id = "custom-request-123"
    state = create_initial_state("Test Topic", request_id=custom_id)
    assert state["request_id"] == custom_id
    assert state["topic"] == "Test Topic"


def test_create_initial_state_generates_uuid() -> None:
    """Test that request_id is generated as UUID when not provided"""
    state1 = create_initial_state("Topic 1")
    state2 = create_initial_state("Topic 2")

    # Both should have valid UUIDs
    assert state1["request_id"] != state2["request_id"]
    # Verify it's a valid UUID format
    uuid.UUID(state1["request_id"])
    uuid.UUID(state2["request_id"])


def test_research_state_structure() -> None:
    """Test that ResearchState has all required fields"""
    state = create_initial_state("Test")
    # Verify all TypedDict fields are present
    assert "topic" in state
    assert "request_id" in state
    assert "research_plan" in state
    assert "search_queries" in state
    assert "search_results" in state
    assert "scraped_content" in state
    assert "arxiv_papers" in state
    assert "pdf_content" in state
    assert "all_documents" in state
    assert "vector_store_id" in state
    assert "retrieved_chunks" in state
    assert "synthesis" in state
    assert "messages" in state
    assert "step_count" in state
    assert "status" in state


def test_create_initial_state_empty_topic() -> None:
    """Test creating initial state with empty topic"""
    state = create_initial_state("")
    assert state["topic"] == ""
    assert state["status"] == "initialized"


def test_create_initial_state_long_topic() -> None:
    """Test creating initial state with long topic"""
    long_topic = "A" * 1000
    state = create_initial_state(long_topic)
    assert state["topic"] == long_topic
